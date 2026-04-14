from __future__ import annotations

from struct import Struct

SPEC_VERSION = "mp4lm/0.1"
MAJOR_VERSION = 0
MINOR_VERSION = 1

MAGIC = b"MP4LM\x00\x01\x00"
INDEX_MAGIC = b"MP4LMI\x00\x01"

# The spec text says "target 128 bytes" but the field table sums to 132.
# The reference implementation fixes the header at 128 bytes with 68 reserved bytes.
HEADER_STRUCT = Struct("<8sHHQQQQQQ68s")
FOOTER_STRUCT = Struct("<8sHHQQQQ32s32sQ12s")
INDEX_HEADER_STRUCT = Struct("<8sHHIIIIQQQ12s")
ARTIFACT_RECORD_STRUCT = Struct("<16sQQQ")
CHUNK_RECORD_STRUCT = Struct("<16sQQQ")
CHUNK_REF_RECORD_STRUCT = Struct("<IIQQ")

HEADER_SIZE = HEADER_STRUCT.size
FOOTER_SIZE = FOOTER_STRUCT.size
INDEX_HEADER_SIZE = INDEX_HEADER_STRUCT.size
ARTIFACT_RECORD_SIZE = ARTIFACT_RECORD_STRUCT.size
CHUNK_RECORD_SIZE = CHUNK_RECORD_STRUCT.size
CHUNK_REF_RECORD_SIZE = CHUNK_REF_RECORD_STRUCT.size

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
MAX_LAYOUT_PASSES = 8
MAX_MANIFEST_SIZE = 64 * 1024 * 1024
MAX_INDEX_SIZE = 64 * 1024 * 1024

FLAG_DETERMINISTIC = 1 << 0

FIXED_CREATED_AT = "1970-01-01T00:00:00Z"

DTYPE_BYTE_WIDTHS = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
