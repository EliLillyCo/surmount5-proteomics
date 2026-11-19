"""SHA-256 verification of pathway inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from analysis.external.pinned import sha256_file


def _mismatch(label: str, expected: str, actual: str) -> ValueError:
    return ValueError(
        f"SHA-256 mismatch for {label}: expected {expected}, actual {actual}. "
        "The input differs from the pinned snapshot; stage the pinned file "
        "or update the digest in analysis/pathway/constants.py."
    )


def verify_sha256(path: Path | str, expected: str) -> None:
    """Raise ValueError naming the file, expected and actual digest on mismatch."""
    actual = sha256_file(Path(path))
    if actual != expected:
        raise _mismatch(str(path), expected, actual)


def verify_bytes(data: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise _mismatch(label, expected, actual)
