"""SHA-256 pinned downloads: a cached copy is re-verified on every use, a fresh one before it lands.

Shared by the external inputs that carry a single pinned digest (the STEP supplementary
workbook, the Hormone Cell Atlas pancreas h5ad, the QC reference workbooks and the published
cis-pQTL tables). The pattern is that of
``analysis.external.hpa_disease_atlas``: a download is streamed to a ``.part`` file beside
the destination and renamed only after it verifies, so a mismatched file never lands in the
cache.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha256(path: Path, expected: str) -> None:
    """Raise when ``path`` does not match ``expected``.

    Raises:
        RuntimeError: Naming the file and both digests.
    """
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"{path}: sha256 {digest} differs from pinned {expected}")


def fetch_pinned(url: str, path: Path, sha256: str, timeout: int = 300) -> Path:
    """The checksum-verified file at ``path``, downloaded from ``url`` first when absent.

    A cached file is re-verified on every call and never re-downloaded.

    Args:
        url: Where to download from when ``path`` does not exist.
        path: Cache location.
        sha256: Expected hex digest of the file's bytes.
        timeout: Seconds for the connection and for each read.

    Returns:
        ``path``.

    Raises:
        RuntimeError: When the cached or downloaded bytes do not match ``sha256``.
    """
    if path.exists():
        check_sha256(path, sha256)
        return path
    logger.info(f"Downloading {url} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with part.open("wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
    try:
        check_sha256(part, sha256)
    except RuntimeError:
        part.unlink()
        raise
    part.rename(path)
    return path
