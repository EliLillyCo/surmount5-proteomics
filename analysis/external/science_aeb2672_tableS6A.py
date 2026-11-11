"""Produce ``data/external/science_aeb2672_tableS6A.parquet`` from the Science supplementary workbook.

    pixi run python analysis/external/science_aeb2672_tableS6A.py --xlsx <science.aeb2672_tables_s1_to_s6.xlsx>

Table S6A of the Hormone Cell Atlas (Fei, Huang-Doran et al., Science 2026,
doi:10.1126/science.aeb2672), "Hormone receiving strengths at the fine-grained cell-type level
across all 47 tissues", is read by ``figures/_internal/_gip_panel.py`` for the GIP-receiving
cell-type panel of Fig 5a. The workbook (``science.aeb2672_tables_s1_to_s6.xlsx``, 22 MB) comes
from the article's supplementary materials on science.org; download it in a browser and pass
it with ``--xlsx``. It is SHA-256 pinned (``XLSX_SHA256``)
and verified before it is read.

The parquet keeps four columns of sheet "Table S6A", in sheet order: ``Hormone_short``,
``Tissue``, the fine-grained cell-type column (``CELLTYPE_COLUMN``) and ``Strength``. The
dropped columns are ``Tier``, ``Source Assay`` and the two cell counts. Every column is text.
``Strength`` is the sheet's float rounded to nine decimals and written without trailing zeros.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import polars as pl

from analysis.external.pinned import check_sha256

ROOT = Path(__file__).resolve().parents[2]


# SHA-256 of science.aeb2672_tables_s1_to_s6.xlsx.
XLSX_SHA256 = "c235a13f04ac71e7066e1e082221766f66785d13e7fd940b0b7eaef130aa6077"
SHEET = "Table S6A"
CELLTYPE_COLUMN = "Tissue level fine-grained cell type (celltype_level2)"
COLUMNS = ("Hormone_short", "Tissue", CELLTYPE_COLUMN, "Strength")
STRENGTH_DECIMALS = 9
OUTPUT_PATH = ROOT / "data" / "external" / "science_aeb2672_tableS6A.parquet"


def _strength_text(value: float) -> str:
    """The strength rounded to ``STRENGTH_DECIMALS`` decimals, trailing zeros dropped."""
    return f"{value:.{STRENGTH_DECIMALS}f}".rstrip("0").rstrip(".")


def read_table(xlsx: Path) -> pl.DataFrame:
    """The four kept columns of sheet ``Table S6A``, all text.

    Raises:
        RuntimeError: When ``xlsx`` does not match ``XLSX_SHA256``.
        ValueError: When the sheet lacks one of ``COLUMNS``.
    """
    check_sha256(xlsx, XLSX_SHA256)
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb[SHEET].iter_rows(values_only=True))
    wb.close()
    header_at = next(i for i, row in enumerate(rows) if row and row[0] == COLUMNS[0])
    header = [str(h).strip() for h in rows[header_at]]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{xlsx} sheet {SHEET} is missing columns {missing}")
    index = [header.index(c) for c in COLUMNS]
    records = [
        [str(row[i]) if c != "Strength" else _strength_text(float(row[i])) for i, c in zip(index, COLUMNS)]
        for row in rows[header_at + 1 :]
        if any(v is not None for v in row)
    ]
    return pl.DataFrame(records, schema=list(COLUMNS), orient="row")


def main(argv: list[str] | None = None) -> int:
    """Write the parquet from the workbook."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xlsx", type=Path, required=True, help="science.aeb2672_tables_s1_to_s6.xlsx")
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args(argv)
    table = read_table(args.xlsx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out)
    print(f"[ok] {args.out} ({table.height} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
