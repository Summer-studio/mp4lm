from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from sys import byteorder
from typing import Any

from ._utils import parse_chunk_size
from .constants import DEFAULT_CHUNK_SIZE
from .exceptions import FormatError
from .reader import Mp4LMBundle
from .writer import BundleBuilder

_TORCH_DTYPE_ALIASES = {
    "bool": ("bool",),
    "int8": ("int8",),
    "uint8": ("uint8",),
    "int16": ("int16", "short"),
    "int32": ("int32", "int"),
    "int64": ("int64", "long"),
    "float16": ("float16", "half"),
    "bfloat16": ("bfloat16",),
    "float32": ("float32", "float"),
    "float64": ("float64", "double"),
}


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise FormatError("PyTorch bridge requires torch to be installed") from exc


def _is_tensor(value: Any) -> bool:
    return (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "contiguous")
    )


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise FormatError("PyTorch checkpoint must deserialize to a mapping or contain a state_dict mapping")

    candidate = payload.get("state_dict")
    if isinstance(candidate, Mapping) and all(_is_tensor(value) for value in candidate.values()):
        return candidate
    if all(_is_tensor(value) for value in payload.values()):
        return payload
    raise FormatError("PyTorch checkpoint does not contain a valid state_dict")


def _torch_dtype_to_mp4lm(torch: Any, dtype: Any) -> str:
    for mp4lm_dtype, aliases in _TORCH_DTYPE_ALIASES.items():
        for alias in aliases:
            torch_dtype = getattr(torch, alias, None)
            if torch_dtype is not None and dtype == torch_dtype:
                return mp4lm_dtype

    dtype_name = str(dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.")
    for mp4lm_dtype, aliases in _TORCH_DTYPE_ALIASES.items():
        if dtype_name in aliases:
            return mp4lm_dtype
    raise FormatError(f"unsupported torch dtype: {dtype!r}")


def _mp4lm_dtype_to_torch(torch: Any, dtype_name: str) -> Any:
    aliases = _TORCH_DTYPE_ALIASES.get(dtype_name)
    if aliases is None:
        raise FormatError(f"unsupported Mp4LM dtype for PyTorch export: {dtype_name!r}")
    for alias in aliases:
        torch_dtype = getattr(torch, alias, None)
        if torch_dtype is not None:
            return torch_dtype
    raise FormatError(f"PyTorch runtime does not expose dtype {dtype_name!r}")


def _tensor_subtype(name: str) -> str:
    lowered = name.lower()
    if "lm_head" in lowered:
        return "lm_head"
    if any(token in lowered for token in ("embed", "embedding", "wte", "wpe")):
        return "embedding"
    if any(token in lowered for token in ("norm", "layernorm", ".ln", "ln_", "rmsnorm")):
        return "norm"
    return "parameter"


def _tensor_numel(tensor: Any) -> int:
    if hasattr(tensor, "numel"):
        return int(tensor.numel())
    total = 1
    for dim in tuple(tensor.shape):
        total *= int(dim)
    return total


def _tensor_bytes(tensor: Any) -> bytes:
    materialized = tensor.detach().cpu().contiguous()
    array = materialized.numpy()
    return array.tobytes(order="C")


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Mapping[str, Any]:
    torch = _require_torch()
    path = Path(checkpoint_path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    return _extract_state_dict(payload)


def builder_from_pytorch_state_dict(
    state_dict: Mapping[str, Any],
    *,
    model: dict[str, Any] | None = None,
    compatibility: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    chunk_size: int | str = DEFAULT_CHUNK_SIZE,
    deterministic: bool = False,
) -> BundleBuilder:
    torch = _require_torch()
    normalized_model = dict(model or {})
    normalized_compatibility = dict(compatibility or {})
    normalized_provenance = dict(provenance or {})
    normalized_provenance.setdefault("source_format", "pytorch-state-dict")
    normalized_compatibility.setdefault("pytorch", getattr(torch, "__version__", "unknown"))

    parameter_count = 0
    builder = BundleBuilder(
        model=normalized_model,
        compatibility=normalized_compatibility,
        provenance=normalized_provenance,
        chunk_size=parse_chunk_size(chunk_size),
        deterministic=deterministic,
    )

    for name, tensor in sorted(_extract_state_dict(state_dict).items()):
        dtype_name = _torch_dtype_to_mp4lm(torch, tensor.dtype)
        shape = [int(dim) for dim in tuple(tensor.shape)]
        builder.add_tensor(
            name,
            _tensor_bytes(tensor),
            dtype=dtype_name,
            shape=shape,
            tensor_subtype=_tensor_subtype(name),
            endianness=byteorder,
        )
        parameter_count += _tensor_numel(tensor)

    builder.model.setdefault("parameter_count", parameter_count)
    return builder


def pack_pytorch_state_dict(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    model: dict[str, Any] | None = None,
    compatibility: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    chunk_size: int | str = DEFAULT_CHUNK_SIZE,
    deterministic: bool = False,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Path:
    checkpoint = Path(checkpoint_path)
    normalized_provenance = dict(provenance or {})
    normalized_provenance.setdefault("source_artifact", checkpoint.name)
    builder = builder_from_pytorch_state_dict(
        load_checkpoint_state_dict(checkpoint, map_location=map_location, weights_only=weights_only),
        model=model,
        compatibility=compatibility,
        provenance=normalized_provenance,
        chunk_size=chunk_size,
        deterministic=deterministic,
    )
    return builder.write(output_path)


def load_pytorch_state_dict(
    bundle_or_path: str | Path | Mp4LMBundle,
    *,
    artifacts: Iterable[str] | None = None,
    clone_tensors: bool = True,
) -> "OrderedDict[str, Any]":
    torch = _require_torch()
    if not hasattr(torch, "frombuffer"):
        raise FormatError("PyTorch export requires torch.frombuffer support")

    bundle = bundle_or_path if isinstance(bundle_or_path, Mp4LMBundle) else Mp4LMBundle.open(bundle_or_path)
    identifiers = (
        list(artifacts)
        if artifacts is not None
        else [artifact["id"] for artifact in bundle.list_artifacts() if artifact["object_type"] == "tensor"]
    )

    state_dict: OrderedDict[str, Any] = OrderedDict()
    for identifier in identifiers:
        artifact = bundle.artifact(identifier)
        if artifact["object_type"] != "tensor":
            raise FormatError(f"artifact {artifact['name']!r} is not a tensor and cannot be exported to state_dict")
        tensor = torch.frombuffer(
            bytearray(bundle.fetch_artifact(artifact["id"])),
            dtype=_mp4lm_dtype_to_torch(torch, artifact["dtype"]),
        ).reshape(tuple(artifact["shape"]))
        if clone_tensors and hasattr(tensor, "clone"):
            tensor = tensor.clone()
        state_dict[artifact["name"]] = tensor
    return state_dict


def export_pytorch_state_dict(
    bundle_or_path: str | Path | Mp4LMBundle,
    output_path: str | Path,
    *,
    artifacts: Iterable[str] | None = None,
) -> Path:
    torch = _require_torch()
    target = Path(output_path)
    torch.save(load_pytorch_state_dict(bundle_or_path, artifacts=artifacts), target)
    return target
