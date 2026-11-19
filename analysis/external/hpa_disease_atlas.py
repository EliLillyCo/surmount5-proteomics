"""Human Protein Atlas pan-disease blood atlas: disease-versus-control differential expression.

    pixi run python -m analysis.external.hpa_disease_atlas

Fetches the HPA blood disease differential-expression table (Olink Explore 1536 plasma
proteomics) on first use, verifies its SHA-256 and caches it under
``data/external/hpa/`` (gitignored). The disease-signature reversal of
Fig 3b (``analysis/disease_reversal.py``) reads it through ``read_disease_de``. Only this
file is needed: it carries the disease ``Class`` column.

Citation
--------
Álvez MB, Bergström S, Kenrick J, et al. "A human pan-disease blood atlas of the circulating
proteome." Science 390(6779), eadx2678 (2025). https://doi.org/10.1126/science.adx2678
PMID: 41066540

Source
------
https://v25.proteinatlas.org/download/tsv/blood_pea_disease_de.tsv.zip (the release's own
host, so the pinned checksum still matches after HPA's next release).
Human Protein Atlas version 25.1 (Last-Modified 2025-11-06), described at
https://www.proteinatlas.org/humanproteome/blood/data ("Blood disease data (PEA)").
SHA-256 of the zip: 6f843164366d6863e63896c01e07d844e8a862c4c9bedcbc4a2cd015e4b23992

License
-------
The Human Protein Atlas is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0, https://www.proteinatlas.org/about/licence). HPA asks
that downloaded data be cited with a primary publication and the data version, as
"data available from v25.proteinatlas.org".

Columns
-------
Gene, ENSG ID, Disease, Class, Control (Healthy / Class / All other diseases), logFC,
p-value adjusted, and the classifier importance columns (Seeds, Mean/SD/Max/Min importance).
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

from analysis.external.pinned import sha256_file

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
HPA_VERSION = "25.1"
URL = "https://v25.proteinatlas.org/download/tsv/blood_pea_disease_de.tsv.zip"
SHA256 = "6f843164366d6863e63896c01e07d844e8a862c4c9bedcbc4a2cd015e4b23992"
MEMBER = "blood_pea_disease_de.tsv"
CACHE_PATH = ROOT / "data" / "external" / "hpa" / "blood_pea_disease_de.tsv.zip"


def _check(path: Path) -> None:
    """Raise when ``path`` does not match the pinned ``SHA256``."""
    digest = sha256_file(path)
    if digest != SHA256:
        raise RuntimeError(f"{path}: sha256 {digest[:16]} differs from pinned {SHA256[:16]}")


def fetch(path: Path = CACHE_PATH) -> Path:
    """The checksum-verified zip at ``path``, downloaded first when absent.

    A download is written to a ``.part`` file beside ``path`` and renamed only after it
    verifies, so a mismatched file never lands in the cache; a cached file is re-verified on
    every call.

    Raises:
        RuntimeError: When the cached or downloaded bytes do not match ``SHA256``.
    """
    if path.exists():
        _check(path)
        return path
    logger.info(f"Downloading HPA v{HPA_VERSION} blood disease DE table -> {path}")
    with requests.get(URL, timeout=300) as r:
        r.raise_for_status()
        data = r.content
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    try:
        _check(part)
    except RuntimeError:
        part.unlink()
        raise
    part.rename(path)
    return path


def read_disease_de(path: Path = CACHE_PATH) -> pl.DataFrame:
    """The differential-expression table, every column as text (callers cast what they use)."""
    with zipfile.ZipFile(fetch(path)) as zf:
        raw = zf.read(MEMBER)
    return pl.read_csv(io.BytesIO(raw), separator="\t", infer_schema_length=0)


def main(argv: list[str] | None = None) -> int:
    """Fetch and verify the HPA table, then report its size."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=CACHE_PATH, help="Where the zip is cached.")
    args = ap.parse_args(argv)
    de = read_disease_de(args.path)
    print(f"[ok] {args.path} ({de.height} rows, {de['Disease'].n_unique()} diseases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
