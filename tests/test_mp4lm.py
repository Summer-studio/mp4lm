from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from importlib import import_module, invalidate_caches
from pathlib import Path

from mp4lm import BundleBuilder, Mp4LMBundle, export_pytorch_state_dict, load_pytorch_state_dict, pack_pytorch_state_dict
from mp4lm.exceptions import FormatError


class Mp4LMRoundTripTests(unittest.TestCase):
    def _write_fake_torch(self, root: Path) -> Path:
        fake_torch = root / "torch.py"
        fake_torch.write_text(
            """
import pickle

__version__ = "0.0.fake"

class FakeDType:
    def __init__(self, name, itemsize):
        self.name = name
        self.itemsize = itemsize
    def __repr__(self):
        return f"torch.{self.name}"
    def __str__(self):
        return f"torch.{self.name}"

bool = FakeDType("bool", 1)
int8 = FakeDType("int8", 1)
uint8 = FakeDType("uint8", 1)
int16 = FakeDType("int16", 2)
short = int16
int32 = FakeDType("int32", 4)
int = int32
int64 = FakeDType("int64", 8)
long = int64
float16 = FakeDType("float16", 2)
half = float16
bfloat16 = FakeDType("bfloat16", 2)
float32 = FakeDType("float32", 4)
float = float32
float64 = FakeDType("float64", 8)
double = float64

class FakeArray:
    def __init__(self, data):
        self._data = bytes(data)
    def tobytes(self, order="C"):
        return self._data

class Tensor:
    def __init__(self, data, dtype, shape):
        self._data = bytes(data)
        self.dtype = dtype
        self.shape = tuple(shape)
    def detach(self):
        return self
    def cpu(self):
        return self
    def contiguous(self):
        return self
    def numpy(self):
        return FakeArray(self._data)
    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total
    def reshape(self, shape):
        return Tensor(self._data, self.dtype, shape)
    def clone(self):
        return Tensor(self._data, self.dtype, self.shape)
    def tobytes(self):
        return self._data

def frombuffer(buffer, dtype):
    data = bytes(buffer)
    if len(data) % dtype.itemsize:
        raise ValueError("buffer size is not aligned with dtype")
    return Tensor(data, dtype, (len(data) // dtype.itemsize,))

def save(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)

def load(path, map_location=None, weights_only=True):
    with open(path, "rb") as handle:
        return pickle.load(handle)
            """.strip()
            + "\n",
            encoding="utf-8",
        )
        return fake_torch

    def test_roundtrip_tensor_and_file_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "model.mp4lm"
            tensor_bytes = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

            builder = BundleBuilder(
                model={"architecture": "decoder-only-transformer", "name": "toy"},
                chunk_size=8,
                deterministic=True,
            )
            builder.add_tensor(
                "layer.weight",
                tensor_bytes,
                dtype="float32",
                shape=[2, 2],
                tensor_subtype="parameter",
            )
            builder.add_file("config.json", b'{"hidden_size":2}')
            builder.write(bundle_path)

            bundle = Mp4LMBundle.open(bundle_path)
            self.assertEqual(bundle.manifest()["spec"], "mp4lm/0.1")
            self.assertEqual(bundle.fetch_artifact("config.json"), b'{"hidden_size":2}')
            tensor = bundle.tensor("layer.weight")
            self.assertEqual(tensor.data, tensor_bytes)
            self.assertEqual(tensor.shape, [2, 2])

            report = bundle.verify()
            self.assertTrue(report["ok"])
            self.assertEqual(report["artifacts_verified"], 2)

    def test_deterministic_output_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "first.mp4lm"
            second = Path(tmp_dir) / "second.mp4lm"
            payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

            for target in (first, second):
                builder = BundleBuilder(model={"name": "toy"}, deterministic=True, chunk_size=7)
                builder.add_tensor("tensor", payload, dtype="float32", shape=[2, 2])
                builder.write(target)

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_invalid_tensor_length_is_rejected(self) -> None:
        with self.assertRaises(FormatError):
            BundleBuilder().add_tensor("broken", b"\x00" * 6, dtype="float32", shape=[2, 1])

    def test_cli_pack_manifest_verify_and_cat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tensor_path = root / "weights.bin"
            note_path = root / "notes.txt"
            spec_path = root / "bundle.json"
            bundle_path = root / "bundle.mp4lm"
            unpack_dir = root / "unpacked"
            tensor_bytes = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

            tensor_path.write_bytes(tensor_bytes)
            note_path.write_text("hello mp4lm", encoding="utf-8")
            spec_path.write_text(
                json.dumps(
                    {
                        "deterministic": True,
                        "model": {"name": "toy"},
                        "artifacts": [
                            {
                                "name": "layer.weight",
                                "path": "weights.bin",
                                "object_type": "tensor",
                                "dtype": "float32",
                                "shape": [2, 2],
                            },
                            {
                                "name": "notes.txt",
                                "path": "notes.txt",
                                "object_type": "file",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path.cwd() / "src")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "pack",
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(bundle_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            manifest_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "manifest",
                    str(bundle_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(manifest_result.stdout)
            self.assertEqual(len(manifest["artifacts"]), 2)

            verify_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "verify",
                    str(bundle_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            verify_report = json.loads(verify_result.stdout)
            self.assertTrue(verify_report["ok"])

            cat_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "cat",
                    str(bundle_path),
                    "layer.weight",
                ],
                check=True,
                env=env,
                capture_output=True,
            )
            self.assertEqual(cat_result.stdout, tensor_bytes)

            unpack_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "unpack",
                    str(bundle_path),
                    str(unpack_dir),
                    "--artifact",
                    "notes.txt",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            unpack_report = json.loads(unpack_result.stdout)
            self.assertTrue(unpack_report["ok"])
            self.assertEqual(
                (unpack_dir / "artifacts" / "file" / "notes.txt").read_text(encoding="utf-8"),
                "hello mp4lm",
            )
            self.assertTrue((unpack_dir / "bundle.manifest.json").exists())

    def test_unpack_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "unsafe.mp4lm"
            unpack_dir = Path(tmp_dir) / "unpacked"

            builder = BundleBuilder(deterministic=True)
            builder.add_file("../escape.txt", b"unsafe")
            builder.write(bundle_path)

            bundle = Mp4LMBundle.open(bundle_path)
            with self.assertRaises(FormatError):
                bundle.unpack(unpack_dir)

    def test_pytorch_bridge_roundtrip_with_fake_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_fake_torch(root)
            sys.path.insert(0, str(root))
            invalidate_caches()
            try:
                sys.modules.pop("torch", None)
                torch = import_module("torch")

                checkpoint_path = root / "model.pt"
                bundle_path = root / "model.mp4lm"
                restored_path = root / "restored.pt"
                original_state_dict = {
                    "embed.weight": torch.Tensor(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0), torch.float32, (2, 2)),
                    "lm_head.weight": torch.Tensor(struct.pack("<2q", 7, 9), torch.int64, (2,)),
                }
                torch.save(original_state_dict, checkpoint_path)

                pack_pytorch_state_dict(checkpoint_path, bundle_path, deterministic=True, chunk_size=9)

                bundle = Mp4LMBundle.open(bundle_path)
                manifest = bundle.manifest()
                self.assertEqual(manifest["compatibility"]["pytorch"], "0.0.fake")
                self.assertEqual(manifest["model"]["parameter_count"], 6)

                roundtrip_state_dict = load_pytorch_state_dict(bundle)
                self.assertEqual(roundtrip_state_dict["embed.weight"].tobytes(), original_state_dict["embed.weight"].tobytes())
                self.assertEqual(roundtrip_state_dict["lm_head.weight"].tobytes(), original_state_dict["lm_head.weight"].tobytes())

                export_pytorch_state_dict(bundle, restored_path)
                restored = torch.load(restored_path)
                self.assertEqual(restored["embed.weight"].tobytes(), original_state_dict["embed.weight"].tobytes())
                self.assertEqual(restored["lm_head.weight"].tobytes(), original_state_dict["lm_head.weight"].tobytes())
            finally:
                sys.modules.pop("torch", None)
                sys.path.pop(0)

    def test_cli_pack_from_and_unpack_to_pytorch_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fake_torch_dir = root / "fake_torch"
            fake_torch_dir.mkdir()
            self._write_fake_torch(fake_torch_dir)

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(fake_torch_dir), str(Path.cwd() / "src")])

            checkpoint_path = root / "model.pt"
            bundle_path = root / "model.mp4lm"
            restored_path = root / "restored.pt"

            create_checkpoint = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import struct, torch; "
                        "state_dict = {"
                        "'embed.weight': torch.Tensor(struct.pack('<4f', 1.0, 2.0, 3.0, 4.0), torch.float32, (2, 2)), "
                        "'lm_head.weight': torch.Tensor(struct.pack('<2q', 11, 13), torch.int64, (2,))}; "
                        f"torch.save(state_dict, r'{checkpoint_path}')"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(create_checkpoint.stdout, "")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "pack",
                    "--input",
                    str(checkpoint_path),
                    "--input-format",
                    "pytorch-state-dict",
                    "--output",
                    str(bundle_path),
                    "--deterministic",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            unpack_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mp4lm.cli",
                    "unpack",
                    str(bundle_path),
                    str(restored_path),
                    "--as",
                    "pytorch-state-dict",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            unpack_report = json.loads(unpack_result.stdout)
            self.assertTrue(unpack_report["ok"])
            self.assertEqual(unpack_report["output_format"], "pytorch-state-dict")

            sys.path.insert(0, str(fake_torch_dir))
            invalidate_caches()
            try:
                sys.modules.pop("torch", None)
                torch = import_module("torch")
                restored = torch.load(restored_path)
                self.assertEqual(
                    restored["embed.weight"].tobytes(),
                    struct.pack("<4f", 1.0, 2.0, 3.0, 4.0),
                )
                self.assertEqual(
                    restored["lm_head.weight"].tobytes(),
                    struct.pack("<2q", 11, 13),
                )
            finally:
                sys.modules.pop("torch", None)
                sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
