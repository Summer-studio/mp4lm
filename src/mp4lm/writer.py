from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ._utils import canonical_json_bytes, hash16, parse_chunk_size, rel_posix, sha256_hex, utc_now_rfc3339
from .constants import (
    ARTIFACT_RECORD_STRUCT,
    CHUNK_RECORD_STRUCT,
    CHUNK_REF_RECORD_STRUCT,
    DEFAULT_CHUNK_SIZE,
    FLAG_DETERMINISTIC,
    FIXED_CREATED_AT,
    FOOTER_SIZE,
    FOOTER_STRUCT,
    HEADER_SIZE,
    HEADER_STRUCT,
    INDEX_HEADER_STRUCT,
    INDEX_HEADER_SIZE,
    INDEX_MAGIC,
    MAGIC,
    MAJOR_VERSION,
    MAX_LAYOUT_PASSES,
    MINOR_VERSION,
    SPEC_VERSION,
)
from .exceptions import FormatError
from .models import ArtifactInput


def _normalize_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _compute_bundle_id(artifacts: list[ArtifactInput], deterministic: bool) -> str:
    if not deterministic:
        return str(uuid.uuid4())
    digest_source = []
    for artifact in artifacts:
        digest_source.append(artifact.artifact_id)
        digest_source.append(sha256_hex(artifact.data))
    return sha256_hex("\n".join(digest_source).encode("utf-8"))


def _build_artifact_manifest_entry(
    artifact: ArtifactInput,
    chunk_ids: list[str],
    payload_checksum: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": artifact.artifact_id,
        "name": artifact.name,
        "object_type": artifact.object_type,
        "payload": {
            "byte_length": len(artifact.data),
            "checksum": {"algorithm": "sha256", "value": payload_checksum},
            "chunk_ids": chunk_ids,
        },
    }
    if artifact.object_type == "tensor":
        entry["dtype"] = artifact.dtype
        entry["shape"] = list(artifact.shape or [])
        entry["layout"] = artifact.layout
        entry["encoding"] = {"kind": "raw", "endianness": artifact.endianness}
        if artifact.tensor_subtype is not None:
            entry["tensor_subtype"] = artifact.tensor_subtype
        if artifact.quantization is not None:
            entry["quantization"] = artifact.quantization
    if artifact.hints:
        entry["hints"] = artifact.hints
    if artifact.media_type is not None:
        entry["media_type"] = artifact.media_type
    return entry


def _empty_manifest(
    *,
    bundle_id: str,
    created_at: str,
    producer: dict[str, Any],
    model: dict[str, Any],
    compatibility: dict[str, Any],
    provenance: dict[str, Any],
    license_info: dict[str, Any],
    artifacts: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifacts": artifacts,
        "bundle_id": bundle_id,
        "chunks": chunks,
        "compatibility": compatibility,
        "created_at": created_at,
        "license": license_info,
        "model": model,
        "preview": [],
        "producer": producer,
        "provenance": provenance,
        "signatures": [],
        "spec": SPEC_VERSION,
    }


def _artifact_entry_offsets(manifest_bytes: bytes, artifacts: list[dict[str, Any]]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for artifact in artifacts:
        encoded = canonical_json_bytes(artifact)
        position = manifest_bytes.find(encoded, cursor)
        if position < 0:
            position = manifest_bytes.find(encoded)
        if position < 0:
            raise FormatError(f"could not locate manifest bytes for artifact {artifact['id']}")
        offsets[artifact["id"]] = position
        cursor = position + len(encoded)
    return offsets


def _build_index_bytes(manifest_bytes: bytes, manifest: dict[str, Any]) -> bytes:
    artifacts = manifest["artifacts"]
    chunks = manifest["chunks"]
    artifact_offsets = _artifact_entry_offsets(manifest_bytes, artifacts)

    artifact_records_offset = INDEX_HEADER_SIZE
    chunk_records_offset = artifact_records_offset + (len(artifacts) * ARTIFACT_RECORD_STRUCT.size)
    total_chunk_refs = sum(len(artifact["payload"]["chunk_ids"]) for artifact in artifacts)
    chunk_refs_offset = chunk_records_offset + (len(chunks) * CHUNK_RECORD_STRUCT.size)

    header = INDEX_HEADER_STRUCT.pack(
        INDEX_MAGIC,
        MAJOR_VERSION,
        MINOR_VERSION,
        len(artifacts),
        len(chunks),
        total_chunk_refs,
        0,
        artifact_records_offset,
        chunk_records_offset,
        chunk_refs_offset,
        b"\x00" * 12,
    )

    chunk_index_by_id = {chunk["id"]: index for index, chunk in enumerate(chunks)}

    artifact_records = bytearray()
    chunk_ref_records = bytearray()
    next_chunk_ref_offset = chunk_refs_offset
    for artifact in artifacts:
        chunk_ids = artifact["payload"]["chunk_ids"]
        artifact_records.extend(
            ARTIFACT_RECORD_STRUCT.pack(
                hash16(artifact["id"]),
                artifact_offsets[artifact["id"]],
                next_chunk_ref_offset if chunk_ids else 0,
                len(chunk_ids),
            )
        )
        for chunk_id in chunk_ids:
            chunk = chunks[chunk_index_by_id[chunk_id]]
            chunk_ref_records.extend(
                CHUNK_REF_RECORD_STRUCT.pack(
                    chunk_index_by_id[chunk_id],
                    0,
                    0,
                    chunk["length"],
                )
            )
            next_chunk_ref_offset += CHUNK_REF_RECORD_STRUCT.size

    chunk_records = bytearray()
    for chunk in chunks:
        chunk_records.extend(
            CHUNK_RECORD_STRUCT.pack(
                hash16(chunk["id"]),
                chunk["offset"],
                chunk["length"],
                0,
            )
        )

    return bytes(header + artifact_records + chunk_records + chunk_ref_records)


class BundleBuilder:
    def __init__(
        self,
        *,
        model: dict[str, Any] | None = None,
        producer_name: str = "mp4lm",
        producer_version: str = "0.1.0",
        compatibility: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        license_info: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        deterministic: bool = False,
        bundle_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.model = _normalize_mapping(model)
        self.producer = {"name": producer_name, "version": producer_version}
        self.compatibility = _normalize_mapping(compatibility)
        self.provenance = _normalize_mapping(provenance)
        self.license_info = _normalize_mapping(license_info)
        self.chunk_size = int(chunk_size)
        self.deterministic = deterministic
        self.bundle_id = bundle_id
        self.created_at = created_at
        self._artifacts: list[ArtifactInput] = []

    def add_artifact(self, artifact: ArtifactInput) -> ArtifactInput:
        if any(existing.name == artifact.name for existing in self._artifacts):
            raise FormatError(f"duplicate artifact name: {artifact.name}")
        self._artifacts.append(artifact)
        return artifact

    def add_tensor(
        self,
        name: str,
        data: bytes,
        *,
        dtype: str,
        shape: list[int] | tuple[int, ...],
        tensor_subtype: str | None = None,
        layout: str = "row-major",
        endianness: str = "little",
        hints: dict[str, Any] | None = None,
        quantization: dict[str, Any] | None = None,
    ) -> ArtifactInput:
        return self.add_artifact(
            ArtifactInput(
                name=name,
                data=data,
                object_type="tensor",
                tensor_subtype=tensor_subtype,
                dtype=dtype,
                shape=shape,
                layout=layout,
                endianness=endianness,
                hints=dict(hints or {}),
                quantization=dict(quantization) if quantization else None,
            )
        )

    def add_file(
        self,
        name: str,
        data: bytes,
        *,
        object_type: str = "file",
        media_type: str | None = None,
        hints: dict[str, Any] | None = None,
    ) -> ArtifactInput:
        return self.add_artifact(
            ArtifactInput(
                name=name,
                data=data,
                object_type=object_type,
                media_type=media_type,
                hints=dict(hints or {}),
            )
        )

    def write(self, output_path: str | Path) -> Path:
        if not self._artifacts:
            raise FormatError("cannot build an empty bundle")
        if self.chunk_size <= 0:
            raise FormatError("chunk size must be positive")

        ordered_artifacts = sorted(self._artifacts, key=lambda artifact: artifact.name)
        bundle_id = self.bundle_id or _compute_bundle_id(ordered_artifacts, self.deterministic)
        created_at = self.created_at or (FIXED_CREATED_AT if self.deterministic else utc_now_rfc3339())

        artifact_entries: list[dict[str, Any]] = []
        chunk_entries: list[dict[str, Any]] = []
        payload_chunks: list[bytes] = []

        next_chunk_number = 1
        for artifact in ordered_artifacts:
            payload_checksum = sha256_hex(artifact.data)
            chunk_ids: list[str] = []
            for start in range(0, len(artifact.data), self.chunk_size):
                chunk_bytes = artifact.data[start : start + self.chunk_size]
                chunk_id = f"chunk_{next_chunk_number:06d}"
                next_chunk_number += 1
                payload_chunks.append(chunk_bytes)
                chunk_ids.append(chunk_id)
                chunk_entries.append(
                    {
                        "checksum": {
                            "algorithm": "sha256",
                            "value": sha256_hex(chunk_bytes),
                        },
                        "compression": {"kind": "none"},
                        "id": chunk_id,
                        "kind": "payload",
                        "length": len(chunk_bytes),
                        "offset": 0,
                    }
                )
            artifact_entries.append(_build_artifact_manifest_entry(artifact, chunk_ids, payload_checksum))

        manifest = _empty_manifest(
            bundle_id=bundle_id,
            created_at=created_at,
            producer=self.producer,
            model=self.model,
            compatibility=self.compatibility,
            provenance=self.provenance,
            license_info=self.license_info,
            artifacts=artifact_entries,
            chunks=chunk_entries,
        )

        manifest_bytes = canonical_json_bytes(manifest)
        for _ in range(MAX_LAYOUT_PASSES):
            # Chunk offsets live inside the manifest, so layout is solved as a small fixed-point loop.
            tentative_index_bytes = _build_index_bytes(manifest_bytes, manifest)
            payload_offset = HEADER_SIZE + len(manifest_bytes) + len(tentative_index_bytes)
            next_offset = payload_offset
            for chunk_entry, chunk_bytes in zip(chunk_entries, payload_chunks, strict=True):
                chunk_entry["offset"] = next_offset
                chunk_entry["length"] = len(chunk_bytes)
                next_offset += len(chunk_bytes)
            next_manifest_bytes = canonical_json_bytes(manifest)
            if next_manifest_bytes == manifest_bytes:
                manifest_bytes = next_manifest_bytes
                break
            manifest_bytes = next_manifest_bytes
        else:
            raise FormatError("bundle layout did not stabilize")

        index_bytes = _build_index_bytes(manifest_bytes, manifest)
        manifest_offset = HEADER_SIZE
        index_offset = manifest_offset + len(manifest_bytes)
        footer_offset = index_offset + len(index_bytes) + sum(len(chunk) for chunk in payload_chunks)
        flags = FLAG_DETERMINISTIC if self.deterministic else 0

        header_bytes = HEADER_STRUCT.pack(
            MAGIC,
            MAJOR_VERSION,
            MINOR_VERSION,
            flags,
            manifest_offset,
            len(manifest_bytes),
            index_offset,
            len(index_bytes),
            footer_offset,
            b"\x00" * 68,
        )
        footer_bytes = FOOTER_STRUCT.pack(
            MAGIC,
            MAJOR_VERSION,
            MINOR_VERSION,
            manifest_offset,
            len(manifest_bytes),
            index_offset,
            len(index_bytes),
            bytes.fromhex(sha256_hex(manifest_bytes)),
            bytes.fromhex(sha256_hex(index_bytes)),
            flags,
            b"\x00" * 12,
        )
        if len(header_bytes) != HEADER_SIZE:
            raise AssertionError("header size mismatch")
        if len(footer_bytes) != FOOTER_SIZE:
            raise AssertionError("footer size mismatch")

        output = Path(output_path)
        with output.open("wb") as handle:
            handle.write(header_bytes)
            handle.write(manifest_bytes)
            handle.write(index_bytes)
            for chunk in payload_chunks:
                handle.write(chunk)
            handle.write(footer_bytes)
        return output


def build_from_spec(spec: dict[str, Any], *, chunk_size: int | str = DEFAULT_CHUNK_SIZE) -> BundleBuilder:
    builder = BundleBuilder(
        model=spec.get("model"),
        compatibility=spec.get("compatibility"),
        provenance=spec.get("provenance"),
        license_info=spec.get("license"),
        chunk_size=parse_chunk_size(chunk_size),
        deterministic=bool(spec.get("deterministic", False)),
        bundle_id=spec.get("bundle_id"),
        created_at=spec.get("created_at"),
    )
    base_dir = Path(spec.get("_base_dir", "."))
    for artifact in spec.get("artifacts", []):
        source_path = base_dir / artifact["path"]
        data = source_path.read_bytes()
        object_type = artifact.get("object_type", "file")
        if object_type == "tensor":
            builder.add_tensor(
                artifact["name"],
                data,
                dtype=artifact["dtype"],
                shape=artifact["shape"],
                tensor_subtype=artifact.get("tensor_subtype"),
                layout=artifact.get("layout", "row-major"),
                endianness=artifact.get("endianness", "little"),
                hints=artifact.get("hints"),
                quantization=artifact.get("quantization"),
            )
        else:
            builder.add_file(
                artifact["name"],
                data,
                object_type=object_type,
                media_type=artifact.get("media_type"),
                hints=artifact.get("hints"),
            )
    return builder


def build_from_spec_file(spec_path: str | Path, *, chunk_size: int | str = DEFAULT_CHUNK_SIZE) -> BundleBuilder:
    path = Path(spec_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["_base_dir"] = str(path.parent)
    return build_from_spec(spec, chunk_size=chunk_size)


def pack_directory(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int | str = DEFAULT_CHUNK_SIZE,
    deterministic: bool = False,
    model: dict[str, Any] | None = None,
) -> Path:
    root = Path(input_dir)
    builder = BundleBuilder(
        model=model,
        chunk_size=parse_chunk_size(chunk_size),
        deterministic=deterministic,
    )
    for path in sorted(file for file in root.rglob("*") if file.is_file()):
        builder.add_file(rel_posix(path, root), path.read_bytes())
    return builder.write(output_path)
