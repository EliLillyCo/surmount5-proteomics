"""Stage the third-party reference inputs the QC reads, checksum-verified.

    pixi run python analysis/external/qc_reference.py --pvar-dir <PGEN directory> --pvar-pattern <PVAR family>

The QC's sex and age concordance models read two published supplementary tables, and
its genetic identity check scores tubes against published cis-pQTL catalogs. Only the
Fenland workbook is a manual download (below); the rest are downloaded.

* The sex-association (Koprulu 2025, Nat Commun, Table S2) and ProtAge (Argentieri 2024,
  Nat Med, Table S1) tables are downloaded from the publisher and written as the TSVs
  the QC readers parse, into ``analysis.qc_paths.QC_REFERENCE_DIR``.
* The cis-pQTL catalogs are fetched from their publications and parsed into
  ``PQTL_CATALOG_DIR``: fourteen parquets, the thirteen catalogs of the QC and Tables S1-S2
  (five Olink, eight SomaScan; ``ckb_soma`` is the CKB Atlas) and the 2025 CKB SomaScan
  catalog, which supplies UniProt to MESA SomaScan. The Fenland catalog's allele check reads
  the trial PVARs of ``--pvar-pattern`` (``paths.yaml`` ``surmount5_fenland_pvar_pattern``)
  in ``--pvar-dir``. Its source, the
  Science supplementary workbook, cannot be fetched by script: it is read from
  ``data/vendor/`` or ``--fenland-xlsx`` (data/external/README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import polars as pl

from analysis.external.pinned import fetch_pinned
from analysis.external.pqtl_catalogs import FENLAND_XLSX, RAW_DIR, build_catalogs
from analysis.qc_paths import PQTL_CATALOG_DIR, PROTAGE_TABLE, SEX_TABLE

# (URL, workbook SHA-256, sheet, 0-based header row, columns the QC reader needs, output).
REFERENCE_TABLES = (
    (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41467-025-59034-4/MediaObjects/41467_2025_59034_MOESM3_ESM.xlsx",
        "cad181e8293b370cfcd971666a3aa1aba1de8649b5cc73cf96b279796de1e028",
        "S. Data2",
        1,
        ("SeqId", "pval.SL", "pval.OL", "UniProt"),
        SEX_TABLE,
    ),
    (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41591-024-03164-7/MediaObjects/41591_2024_3164_MOESM3_ESM.xlsx",
        "89d8a380108d6e8dc12e19dcaba73c2faf6121204fbf4b2a1dbcaa2d23682eee",
        "TS1. ProtAge APs",
        0,
        ("UniProt ID",),
        PROTAGE_TABLE,
    ),
)


def read_sheet(xlsx: Path, sheet: str, header_row: int) -> pl.DataFrame:
    """Read one worksheet into an all-string DataFrame, rows above the header dropped."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    header = [str(h) for h in rows[header_row]]
    body = [r for r in rows[header_row + 1 :] if any(v is not None for v in r)]
    # Cells are written as text: the QC readers re-parse the TSV with their own schema
    # inference.
    records = [
        {h: (None if v is None else str(v)) for h, v in zip(header, r, strict=False)} for r in body
    ]
    return pl.DataFrame(records, schema={h: pl.Utf8 for h in header})


def stage_reference_tables(cache: Path) -> None:
    """Download both workbooks into ``cache`` and write the TSVs the QC checks read."""
    cache.mkdir(parents=True, exist_ok=True)
    for url, digest, sheet, header_row, needed, out in REFERENCE_TABLES:
        xlsx = fetch_pinned(url, cache / url.rsplit("/", 1)[1], digest)
        table = read_sheet(xlsx, sheet, header_row)
        missing = [c for c in needed if c not in table.columns]
        if missing:
            raise ValueError(f"{xlsx.name} sheet {sheet!r} lacks {missing}")
        out.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(out, separator="\t")
        print(f"[ok] wrote {out} ({table.height} rows)")


def main(argv: list[str] | None = None) -> int:
    """Stage the reference tables and the parsed catalogs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pvar-dir", type=Path, required=True, help="The trial's PGEN directory.")
    ap.add_argument(
        "--pvar-pattern",
        required=True,
        help="PVAR file name without .pvar, {chrom} for the chromosome (the Fenland allele check).",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "external" / "qc_reference",
        help="Where the downloaded workbooks are kept.",
    )
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Where the published pQTL tables are kept.")
    ap.add_argument(
        "--fenland-xlsx",
        type=Path,
        default=FENLAND_XLSX,
        help="science.abj1541_tables_s1_to_s9.xlsx, downloaded from science.org (data/external/README.md).",
    )
    args = ap.parse_args(argv)
    stage_reference_tables(args.cache)
    build_catalogs(PQTL_CATALOG_DIR, args.pvar_dir, args.pvar_pattern, args.raw_dir, args.fenland_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
