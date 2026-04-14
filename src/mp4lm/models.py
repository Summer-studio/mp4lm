from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import DTYPE_BYTE_WIDTHS
from .exceptions import FormatError


def _normalize_shape(shape: list[int] | tuple[int, ...] | None) -> list[int] | None:
    if shape is None:
        return None
    normalized = [int(dim) for dim in shape]
    if any(dim < 0 for dim in normalized):
        raise FormatError("tensor shapes must be non-negative")
    return normalized


@dataclass(slots=True)
class ArtifactInput:
    name: str
    data: bytes
    object_type: str
    tensor_subtype: str | None = None
    dtype: str | None = None
    shape: list[int] | tuple[int, ...] | None = None
    layout: str = "row-major"
    endianness: str = "little"
    hints: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise FormatError("artifact name must not be empty")
        self.shape = _normalize_shape(self.shape)
        if self.object_type == "tensor":
            if self.dtype is None or self.shape is None:
                raise FormatError("tensor artifacts require dtype and shape")
            expected = self.expected_byte_length()
            if expected is not None and expected != len(self.data):
                raise FormatError(
                    f"tensor {self.name!r} byte length mismatch: expected {expected}, got {len(self.data)}"
                )

    @property
    def artifact_id(self) -> str:
        return f"{self.object_type}:{self.name}"

    def expected_byte_length(self) -> int | None:
        if self.dtype is None or self.shape is None:
            return None
        width = DTYPE_BYTE_WIDTHS.get(self.dtype)
        if width is None:
            return None
        total = width
        for dim in self.shape:
            total *= dim
        return total


@dataclass(slots=True)
class TensorPayload:
    name: str
    data: bytes
    dtype: str
    shape: list[int]
    layout: str
    tensor_subtype: str | None
    endianness: str
    quantization: dict[str, Any] | None
