"""Gene-set cache: downloads Enrichr libraries once, persists to disk."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import requests

from .constants import CACHE_DIR

logger = logging.getLogger(__name__)

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"


def _download_enrichr_library(name: str) -> dict:
    """Download a gene-set library from Enrichr, returning {set_name: [genes]}.

    Bypasses gseapy's download_library which has a Python 3.14 / requests
    compatibility bug (Enrichr returns no Content-Type header, so
    response.encoding is None and iter_lines yields bytes instead of str).
    """
    url = f"{ENRICHR_URL}/geneSetLibrary?mode=text&libraryName={name}"
    response = requests.get(url, timeout=60, stream=True)
    if not response.ok:
        raise RuntimeError(
            f"Failed to download gene set library '{name}' from Enrichr "
            f"(HTTP {response.status_code})"
        )
    # Force UTF-8 decoding since Enrichr omits Content-Type header
    response.encoding = "utf-8"
    genesets: dict = {}
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        parts = line.strip().split("\t")
        set_name = parts[0]
        # Column 1 is description (often empty); genes start at column 2
        genes = [g.split(",")[0] for g in parts[2:] if g]
        genesets[set_name] = genes
    return genesets


class GeneSetCache:
    """Downloads gene-set libraries once, caching to disk and memory.

    On first request, downloads from Enrichr and saves a JSON file under
    *cache_dir*.  Subsequent calls (even across process restarts) load
    from disk.  Use ``--clear-cache`` or delete the cache directory to
    force a fresh download.
    """

    DEFAULT_DIR = CACHE_DIR

    def __init__(self, cache_dir: Path | None = None):
        self._mem: dict[str, dict] = {}
        self._dir = cache_dir or self.DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self._dir / f"{safe}.json"

    def get(self, name: str, *, download_missing: bool = False) -> dict:
        """Return a cached gene-set library, optionally downloading it on demand."""
        # 1. memory
        if name in self._mem:
            return self._mem[name]

        # 2. disk
        path = self._disk_path(name)
        if path.exists():
            logger.info(f"Loading gene set library from cache: {name}")
            with open(path) as f:
                lib = json.load(f)
            self._mem[name] = lib
            logger.info(f"  -> {len(lib)} sets loaded from {path.name}")
            return lib

        if not download_missing:
            raise FileNotFoundError(
                f"Cached gene-set library '{name}' not found at {path}. "
                "Run the command again with --download-missing to fetch and cache it."
            )

        # 3. download & persist
        logger.info(f"Downloading gene set library: {name}")
        lib = _download_enrichr_library(name)
        logger.info(f"  -> {len(lib)} sets downloaded")
        with open(path, "w") as f:
            json.dump(lib, f)
        logger.info(f"  -> cached to {path}")
        self._mem[name] = lib
        return lib

    def prefetch(self, names: list[str], *, download_missing: bool = False):
        """Ensure all requested libraries are available."""
        for name in names:
            self.get(name, download_missing=download_missing)

    def clear(self):
        """Remove all cached files."""
        if self._dir.exists():
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)
        self._mem.clear()
        logger.info("Gene set cache cleared")


# Global cache instance (shared within the process)
_gene_set_cache = GeneSetCache()
