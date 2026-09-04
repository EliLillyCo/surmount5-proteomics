"""Download and extract QC reference tables for sex and age concordance.

Two publicly available supplementary tables are required by the sex- and
age-concordance QC checks.  This script fetches the Excel workbooks from
the publisher, extracts the required sheet from each, and writes them as
TSVs at the locations expected by ``analysis/qc/sex_concordance.py`` and
``analysis/qc/age_concordance.py``.

Output files
------------
analysis/qc/data/41467_2025_59034_MOESM3_ESM.Table_S2.tsv
    Sex-associated proteins (Koprulu et al. 2025, Nat Commun 16:4001).
    Sheet: "S. Data2" (header at row 2 due to legend row).

analysis/qc/data/41591_2024_3164_MOESM3_ESM.Table_S1.tsv
    ProtAge 204 age-associated proteins (Argentieri et al. 2024, Nat Med 30:2450).
    Sheet: "TS1. ProtAge APs".

Sources
-------
Koprulu M, et al.  "Sex differences in the genetic regulation of the human
plasma proteome."  Nat Commun 16, 4001 (2025).
DOI: 10.1038/s41467-025-59034-4

Argentieri MA, et al.  "Proteomic aging clock predicts mortality and risk of
common age-related diseases in diverse populations."  Nat Med 30, 2450–2460
(2024).  DOI: 10.1038/s41591-024-03164-7

Run
---
    pixi shell
    uv run python analysis/scripts/qc_reference_tables.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import fastexcel
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
QC_DATA_DIR = ROOT / "analysis" / "qc" / "data"

# ---------------------------------------------------------------------------
# Sex-associated proteins (Koprulu et al. 2025)
# ---------------------------------------------------------------------------
SEX_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-025-59034-4/MediaObjects/"
    "41467_2025_59034_MOESM3_ESM.xlsx"
)
SEX_XLSX_CACHE = QC_DATA_DIR / "41467_2025_59034_MOESM3_ESM.xlsx"
SEX_TSV_OUTPUT = QC_DATA_DIR / "41467_2025_59034_MOESM3_ESM.Table_S2.tsv"
SEX_SHEET = "S. Data2"
SEX_HEADER_ROW = 1  # legend row on top → real header at row 1

# ---------------------------------------------------------------------------
# ProtAge 204 proteins (Argentieri et al. 2024)
# ---------------------------------------------------------------------------
AGE_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41591-024-03164-7/MediaObjects/"
    "41591_2024_3164_MOESM3_ESM.xlsx"
)
AGE_XLSX_CACHE = QC_DATA_DIR / "41591_2024_3164_MOESM3_ESM.xlsx"
AGE_TSV_OUTPUT = QC_DATA_DIR / "41591_2024_3164_MOESM3_ESM.Table_S1.tsv"
AGE_SHEET = "TS1. ProtAge APs"


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Fetch a supplementary Excel workbook, caching on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  cached: {dest.name}")
        return dest
    print(f"  downloading → {dest.name}")
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_tables(force: bool = False) -> None:
    """Download both workbooks and extract the required sheets to TSV."""
    QC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Sex table ---
    print("Sex-associated proteins (Koprulu et al. 2025, Nat Commun):")
    xlsx_path = _download(SEX_URL, SEX_XLSX_CACHE, force=force)
    xl = fastexcel.read_excel(str(xlsx_path))
    sex_df = xl.load_sheet(SEX_SHEET, header_row=SEX_HEADER_ROW).to_polars()
    sex_df.write_csv(SEX_TSV_OUTPUT, separator="\t")
    print(f"  wrote: {SEX_TSV_OUTPUT.name}  ({sex_df.shape[0]} rows, {sex_df.shape[1]} cols)")

    # --- ProtAge table ---
    print("ProtAge 204 proteins (Argentieri et al. 2024, Nat Med):")
    xlsx_path = _download(AGE_URL, AGE_XLSX_CACHE, force=force)
    xl2 = fastexcel.read_excel(str(xlsx_path))
    age_df = xl2.load_sheet(AGE_SHEET).to_polars()
    age_df.write_csv(AGE_TSV_OUTPUT, separator="\t")
    print(f"  wrote: {AGE_TSV_OUTPUT.name}  ({age_df.shape[0]} rows, {age_df.shape[1]} cols)")

    # --- Validation ---
    print("\nValidation:")
    assert {"SeqId", "pval.SL", "pval.OL", "UniProt"} <= set(sex_df.columns), (
        f"Sex table missing expected columns; got: {sex_df.columns}"
    )
    print(f"  sex table OK: {sex_df.shape}")

    assert "UniProt ID" in age_df.columns, (
        f"ProtAge table missing 'UniProt ID'; got: {age_df.columns}"
    )
    print(f"  protage table OK: {age_df.shape}")


if __name__ == "__main__":
    fetch_tables()
