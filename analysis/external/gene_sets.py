"""Fetch the Enrichr gene-set libraries the pathway stage reads (stage ``gene_sets``).

    pixi run python analysis/external/gene_sets.py [--out-dir <dir>]

Each library of ``analysis.pathway.constants.DEFAULT_GENE_SETS`` is written as ``{set_name: [genes]}``
JSON to ``<out-dir>/<library>.json`` (default ``data/external/gene_sets/``, the pathway CLI's
``--cache-dir`` default) and checked against its SHA-256 in ``GENE_SET_SHA256``. A library already
present with the pinned digest is not downloaded again; a present file with another digest raises.
A download is verified before it is written, so a mismatched download raises and writes nothing.
Enrichr revises libraries in place, so a download passes only while Enrichr serves the pinned bytes.

Source: https://maayanlab.cloud/Enrichr (Kuleshov MV, et al. Nucleic Acids Res 2016).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.pathway.cache import _download_enrichr_library
from analysis.pathway.constants import CACHE_DIR, DEFAULT_GENE_SETS, GENE_SET_SHA256
from analysis.pathway.verify import verify_bytes, verify_sha256


def fetch_libraries(out_dir: Path = CACHE_DIR, names: list[str] = DEFAULT_GENE_SETS) -> list[Path]:
    """The verified library JSONs in ``out_dir``, each downloaded from Enrichr first when absent.

    Args:
        out_dir: Where the ``<library>.json`` files are kept.
        names: Enrichr library names, each pinned in ``GENE_SET_SHA256``.

    Returns:
        The library paths, in ``names`` order.

    Raises:
        KeyError: When a library has no pinned digest.
        ValueError: When a present file or a download does not match its pinned digest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        pin = GENE_SET_SHA256[name]
        path = out_dir / f"{name}.json"
        if path.exists():
            verify_sha256(path, pin)
            print(f"[ok] {path} (present, pinned digest)")
        else:
            payload = json.dumps(_download_enrichr_library(name)).encode()
            verify_bytes(payload, pin, f"downloaded Enrichr library {name}")
            part = path.with_name(path.name + ".part")
            part.write_bytes(payload)
            part.rename(path)
            print(f"[ok] {path} (downloaded, pinned digest)")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Fetch (or verify) the pinned gene-set libraries."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=CACHE_DIR, help="where the library JSONs are kept")
    args = ap.parse_args(argv)
    fetch_libraries(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
