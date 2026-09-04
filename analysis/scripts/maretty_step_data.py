"""Maretty et al. 2025 STEP 1 / STEP 2 SomaScan supplementary tables.

Pulls the Nature Medicine supplementary Excel directly from the publisher,
caches the raw workbook under ``data/external/``, and materializes one
canonical long-form parquet at ``analysis/outputs/maretty_step_data.parquet``.

That parquet is the single analysis-side STEP/Maretty output used by:
- `figures/supp_fig9_cross_study.py`
- `figures/supp_fig10_cross_study_pcbl.py`
- `figures/fig5_ppy.py`
- `figures/supp_fig15_reg4.py`

Citation
--------
Maretty L, Schmiegelow MD, Suderman M, et al.
"Plasma proteome response to semaglutide treatment of obesity."
Nat Med 31, 1–11 (2025).  https://doi.org/10.1038/s41591-024-03355-2
PMID: 39753963

Source URL
----------
https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-024-03355-2/
MediaObjects/41591_2024_3355_MOESM3_ESM.xlsx

Sheet → (trial, model) mapping
------------------------------
S2_tx_STEP1          → (`step1`, `primary`)
S3_tx_STEP2          → (`step2`, `primary`)
S6_tx_blocked_STEP1  → (`step1`, `blocked`)
S7_tx_blocked_STEP2  → (`step2`, `blocked`)

The "blocked" tables are the paper's sensitivity analysis with the treatment
effect adjusted for %ΔWeight and %ΔHbA1c.

Columns preserved (renamed to match our conventions)
----------------------------------------------------
- trial          — `step1` or `step2`
- model          — `primary` or `blocked`
- SeqId          — SomaScan aptamer ID, dot-form normalized to dash-form
                   (e.g. ``seq.4588.1`` → ``4588-1``)
- gene_symbol    — HGNC symbol (paper's ``EntrezGeneSymbol``)
- target         — SomaLogic target short name
- target_full    — SomaLogic target full name
- uniprot        — UniProt accession
- effect_size    — log relative change (SEMA 2.4 mg vs PBO)
- std_error      — SE of effect_size
- test_statistic — t/Wald statistic
- pvalue         — raw p
- pvalue_adj     — Bonferroni-style adjusted p (capped at 1.0)
- qvalue         — Storey q-value (paper's FDR)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import openpyxl
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "maretty_step_data.parquet"
DATA_DIR = ROOT / "data" / "external"

SUPP_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41591-024-03355-2/MediaObjects/"
    "41591_2024_3355_MOESM3_ESM.xlsx"
)
XLSX_CACHE = DATA_DIR / "maretty_2025_supp.xlsx"

SHEET_SPECS = [
    ("step1", "primary", "S2_tx_STEP1"),
    ("step2", "primary", "S3_tx_STEP2"),
    ("step1", "blocked", "S6_tx_blocked_STEP1"),
    ("step2", "blocked", "S7_tx_blocked_STEP2"),
]

# Column rename: paper → our convention.
COL_RENAME = {
    "ANALYTEID": "_analyte_id",  # dropped after SeqId derivation
    "STUDYID": "_study_id",
    "term": "_term",
    "effect_size": "effect_size",
    "std_error": "std_error",
    "test_statistic": "test_statistic",
    "pvalue": "pvalue",
    "pvalue_adj": "pvalue_adj",
    "qvalue": "qvalue",
    "Target": "target",
    "TargetFullName": "target_full",
    "UniProt": "uniprot",
    "EntrezGeneID": "_entrez_id",
    "EntrezGeneSymbol": "gene_symbol",
}


def _download_supp(force: bool = False) -> Path:
    """Fetch the Nature Medicine supplementary xlsx, caching under data/external/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if XLSX_CACHE.exists() and not force:
        return XLSX_CACHE
    print(f"Downloading Maretty 2025 supplementary tables → {XLSX_CACHE}")
    urllib.request.urlretrieve(SUPP_URL, XLSX_CACHE)
    return XLSX_CACHE


def _normalize_seqid(value: str | None) -> str | None:
    """Normalize SeqId from dot-form (seq.4588.1) to dash-form (4588-1)."""
    if value is None:
        return None
    return value.replace("seq.", "").replace(".", "-")


def _read_sheet(xlsx_path: Path, trial: str, model: str, sheet_name: str) -> pl.DataFrame:
    """Read one sheet via openpyxl (read-only) and return a long-form frame."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    records = [dict(zip(header, r)) for r in rows]
    wb.close()
    df = pl.DataFrame(records)

    # Rename and prune, then normalise SeqId to dash-form (matches surmount5_soma).
    df = df.rename({k: v for k, v in COL_RENAME.items() if k in df.columns})
    df = df.with_columns(
        pl.lit(trial).alias("trial"),
        pl.lit(model).alias("model"),
        pl.col("_analyte_id")
        .map_elements(_normalize_seqid, return_dtype=pl.String)
        .alias("SeqId"),
    )
    keep_cols = [
        "trial",
        "model",
        "SeqId",
        "gene_symbol",
        "target",
        "target_full",
        "uniprot",
        "effect_size",
        "std_error",
        "test_statistic",
        "pvalue",
        "pvalue_adj",
        "qvalue",
    ]
    return df.select(keep_cols)


def build_table(force_download: bool = False) -> Path:
    """Download (if needed) and write the canonical long-form STEP parquet."""
    xlsx = _download_supp(force=force_download)
    frames = [
        _read_sheet(xlsx, trial=trial, model=model, sheet_name=sheet_name)
        for trial, model, sheet_name in SHEET_SPECS
    ]
    combined = pl.concat(frames, how="vertical")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT_PATH)
    print(f"Saved → {OUTPUT_PATH}")
    for trial, model, _ in SHEET_SPECS:
        n_rows = combined.filter(
            (pl.col("trial") == trial) & (pl.col("model") == model)
        ).height
        print(f"  {trial:5s} {model:7s}  ({n_rows} rows)")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_table()
