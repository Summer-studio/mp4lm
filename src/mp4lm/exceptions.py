from __future__ import annotations


class Mp4LMError(Exception):
    """Base exception for Mp4LM errors."""


class FormatError(Mp4LMError):
    """Raised when a bundle is structurally invalid."""


class IntegrityError(Mp4LMError):
    """Raised when hashes or byte ranges do not verify."""
