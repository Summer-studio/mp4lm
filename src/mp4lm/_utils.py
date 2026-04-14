from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash16(data: str) -> bytes:
    return hashlib.sha256(data.encode("utf-8")).digest()[:16]


def utc_now_rfc3339() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_chunk_size(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("chunk size must be positive")
        return value

    text = value.strip().lower()
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    for suffix, factor in suffixes.items():
        if text.endswith(suffix):
            number = float(text[: -len(suffix)])
            return int(number * factor)
    return int(text)


def rel_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()
