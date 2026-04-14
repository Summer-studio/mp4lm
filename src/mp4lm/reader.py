from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ._utils import sha256_hex
from .constants import (
    DTYPE_BYTE_WIDTHS,
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    INDEX_HEADER_SIZE,
    INDEX_HEADER_STRUCT,
    INDEX_MAGIC,
    MAGIC,
    MAX_INDEX_SIZE,
    MAX_MANIFEST_SIZE,
    SPEC_VERSION,
)
from .exceptions import FormatError, IntegrityError
from .models import TensorPayload


def _expected_tensor_length(dtype: str | None, shape: list[int] | None) -> int | None:
    if dtype is None or shape is None:
        return None
    width = DTYPE_BYTE_WIDTHS.get(dtype)
    if width is None:
        return None
    total = width
    for dim in shape:
        total *= int(dim)
    return total


def _artifact_output_path(root: Path, artifact: dict[str, Any]) -> Path:
    name = artifact["name"]
    raw_name = Path(name)
    if raw_name.is_absolute():
        raise FormatError(f"artifact name must not be absolute during unpack: {name!r}")
    if any(part in {"", ".", ".."} for part in raw_name.parts):
        raise FormatError(f"artifact name is unsafe for unpack: {name!r}")

    relative_path = Path("artifacts") / Path(*artifact["object_type"].split(".")) / raw_name
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise FormatError(f"artifact path escapes the output directory: {name!r}")
    return candidate


class Mp4LMBundle:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._size = self.path.stat().st_size
        self._header = self._read_header()
        self._manifest_bytes = self._read_range(
            self._header["manifest_offset"],
            self._header["manifest_length"],
        )
        self._index_bytes = self._read_range(
            self._header["index_offset"],
            self._header["index_length"],
        )
        self._footer = self._read_footer()
        self._validate_control_plane()
        self._manifest = json.loads(self._manifest_bytes.decode("utf-8"))
        self._validate_manifest()
        self._artifacts_by_id = {artifact["id"]: artifact for artifact in self._manifest["artifacts"]}
        self._artifacts_by_name = {artifact["name"]: artifact for artifact in self._manifest["artifacts"]}
        self._chunks_by_id = {chunk["id"]: chunk for chunk in self._manifest["chunks"]}

    @classmethod
    def open(cls, source: str | Path) -> "Mp4LMBundle":
        return cls(source)

    def _read_range(self, offset: int, length: int) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)

    def _read_header(self) -> dict[str, int]:
        with self.path.open("rb") as handle:
            raw = handle.read(HEADER_SIZE)
        if len(raw) != HEADER_SIZE:
            raise FormatError("file is too small to contain an Mp4LM header")
        (
            magic,
            major,
            minor,
            flags,
            manifest_offset,
            manifest_length,
            index_offset,
            index_length,
            footer_offset,
            _reserved,
        ) = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise FormatError("invalid Mp4LM magic bytes")
        return {
            "major": major,
            "minor": minor,
            "flags": flags,
            "manifest_offset": manifest_offset,
            "manifest_length": manifest_length,
            "index_offset": index_offset,
            "index_length": index_length,
            "footer_offset": footer_offset,
        }

    def _read_footer(self) -> dict[str, Any]:
        offset = self._header["footer_offset"]
        if offset + FOOTER_SIZE > self._size:
            raise FormatError("footer offset points outside the bundle")
        raw = self._read_range(offset, FOOTER_SIZE)
        (
            magic,
            major,
            minor,
            manifest_offset,
            manifest_length,
            index_offset,
            index_length,
            manifest_sha256,
            index_sha256,
            flags,
            _reserved,
        ) = FOOTER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise FormatError("invalid footer magic bytes")
        return {
            "major": major,
            "minor": minor,
            "manifest_offset": manifest_offset,
            "manifest_length": manifest_length,
            "index_offset": index_offset,
            "index_length": index_length,
            "manifest_sha256": manifest_sha256.hex(),
            "index_sha256": index_sha256.hex(),
            "flags": flags,
        }

    def _validate_control_plane(self) -> None:
        if self._header["manifest_length"] > MAX_MANIFEST_SIZE:
            raise FormatError("manifest exceeds the allowed size limit")
        if self._header["index_length"] > MAX_INDEX_SIZE:
            raise FormatError("index exceeds the allowed size limit")
        if self._header["manifest_offset"] != HEADER_SIZE:
            raise FormatError("manifest must begin immediately after the fixed header in v0.1")
        if self._header["index_offset"] != self._header["manifest_offset"] + self._header["manifest_length"]:
            raise FormatError("index offset does not follow manifest")
        if self._footer["manifest_offset"] != self._header["manifest_offset"]:
            raise FormatError("footer manifest offset does not match header")
        if self._footer["manifest_length"] != self._header["manifest_length"]:
            raise FormatError("footer manifest length does not match header")
        if self._footer["index_offset"] != self._header["index_offset"]:
            raise FormatError("footer index offset does not match header")
        if self._footer["index_length"] != self._header["index_length"]:
            raise FormatError("footer index length does not match header")
        if sha256_hex(self._manifest_bytes) != self._footer["manifest_sha256"]:
            raise IntegrityError("manifest checksum does not match footer")
        if sha256_hex(self._index_bytes) != self._footer["index_sha256"]:
            raise IntegrityError("index checksum does not match footer")

        (
            index_magic,
            _major,
            _minor,
            _artifact_count,
            _chunk_count,
            _chunk_ref_count,
            _flags,
            artifact_records_offset,
            chunk_records_offset,
            chunk_refs_offset,
            _reserved,
        ) = INDEX_HEADER_STRUCT.unpack(self._index_bytes[:INDEX_HEADER_SIZE])
        if index_magic != INDEX_MAGIC:
            raise FormatError("invalid index magic bytes")
        if not (
            artifact_records_offset <= chunk_records_offset <= chunk_refs_offset <= len(self._index_bytes)
        ):
            raise FormatError("index section offsets are invalid")

    def _validate_manifest(self) -> None:
        if self._manifest.get("spec") != SPEC_VERSION:
            raise FormatError(f"unsupported spec version: {self._manifest.get('spec')!r}")
        artifacts = self._manifest.get("artifacts", [])
        chunks = self._manifest.get("chunks", [])
        if len({artifact["id"] for artifact in artifacts}) != len(artifacts):
            raise FormatError("artifact ids must be unique")
        if len({artifact["name"] for artifact in artifacts}) != len(artifacts):
            raise FormatError("artifact names must be unique")
        if len({chunk["id"] for chunk in chunks}) != len(chunks):
            raise FormatError("chunk ids must be unique")

        payload_region_end = self._header["footer_offset"]
        ordered_chunks = sorted(chunks, key=lambda chunk: (chunk["offset"], chunk["length"], chunk["id"]))
        previous_end = self._header["index_offset"] + self._header["index_length"]
        for chunk in ordered_chunks:
            offset = int(chunk["offset"])
            length = int(chunk["length"])
            if length < 0:
                raise FormatError(f"chunk {chunk['id']} has a negative length")
            if offset < previous_end:
                raise FormatError(f"chunk {chunk['id']} overlaps a control plane or prior chunk")
            if offset + length > payload_region_end:
                raise FormatError(f"chunk {chunk['id']} extends past the footer")
            previous_end = offset + length

        chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        for artifact in artifacts:
            payload = artifact.get("payload", {})
            chunk_ids = payload.get("chunk_ids", [])
            for chunk_id in chunk_ids:
                if chunk_id not in chunks_by_id:
                    raise FormatError(f"artifact {artifact['id']} references unknown chunk {chunk_id}")
            if artifact["object_type"] == "tensor":
                expected = _expected_tensor_length(artifact.get("dtype"), artifact.get("shape"))
                byte_length = payload.get("byte_length")
                if expected is not None and expected != byte_length:
                    raise FormatError(
                        f"tensor {artifact['name']!r} declares {byte_length} bytes but dtype/shape require {expected}"
                    )

    def header(self) -> dict[str, int]:
        return dict(self._header)

    def footer(self) -> dict[str, Any]:
        return dict(self._footer)

    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_bytes.decode("utf-8"))

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [dict(artifact) for artifact in self._manifest["artifacts"]]

    def artifact(self, identifier: str) -> dict[str, Any]:
        artifact = self._artifacts_by_name.get(identifier) or self._artifacts_by_id.get(identifier)
        if artifact is None:
            raise KeyError(identifier)
        return artifact

    def read_chunk(self, chunk_id: str) -> bytes:
        chunk = self._chunks_by_id[chunk_id]
        return self._read_range(chunk["offset"], chunk["length"])

    def fetch_artifact(self, name: str) -> bytes:
        artifact = self.artifact(name)
        payload = artifact["payload"]
        data = b"".join(self.read_chunk(chunk_id) for chunk_id in payload["chunk_ids"])
        if sha256_hex(data) != payload["checksum"]["value"]:
            raise IntegrityError(f"artifact checksum mismatch for {artifact['name']}")
        return data

    def tensor(self, name: str) -> TensorPayload:
        artifact = self.artifact(name)
        if artifact["object_type"] != "tensor":
            raise FormatError(f"artifact {name!r} is not a tensor")
        return TensorPayload(
            name=artifact["name"],
            data=self.fetch_artifact(name),
            dtype=artifact["dtype"],
            shape=list(artifact["shape"]),
            layout=artifact["layout"],
            tensor_subtype=artifact.get("tensor_subtype"),
            endianness=artifact["encoding"]["endianness"],
            quantization=artifact.get("quantization"),
        )

    def verify(self) -> dict[str, Any]:
        chunk_count = 0
        for chunk in self._manifest["chunks"]:
            payload = self.read_chunk(chunk["id"])
            if sha256_hex(payload) != chunk["checksum"]["value"]:
                raise IntegrityError(f"chunk checksum mismatch for {chunk['id']}")
            chunk_count += 1

        artifact_count = 0
        for artifact in self._manifest["artifacts"]:
            self.fetch_artifact(artifact["name"])
            artifact_count += 1

        return {
            "artifacts_verified": artifact_count,
            "bundle": str(self.path),
            "chunks_verified": chunk_count,
            "ok": True,
        }

    def unpack(
        self,
        output_dir: str | Path,
        *,
        artifacts: Iterable[str] | None = None,
        include_manifest: bool = True,
    ) -> list[Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        if include_manifest:
            manifest_path = root / "bundle.manifest.json"
            manifest_path.write_text(
                json.dumps(self.manifest(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            written.append(manifest_path)

        selected = (
            [self.artifact(identifier) for identifier in artifacts]
            if artifacts is not None
            else list(self._manifest["artifacts"])
        )
        for artifact in selected:
            output_path = _artifact_output_path(root, artifact)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(self.fetch_artifact(artifact["id"]))
            written.append(output_path)

        return written


class Mp4LMLoader:
    def open(self, source: str | Path) -> Mp4LMBundle:
        return Mp4LMBundle.open(source)
