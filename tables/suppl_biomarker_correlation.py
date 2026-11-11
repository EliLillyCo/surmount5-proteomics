"""Proteomics–clinical lab biomarker correlations (Spearman) at all timepoints.

Each row is one (platform, assay, lab test, timepoint) combination. The pairs, loaders,
visit matching and the fit are ``analysis.biomarker_correlation``, shared with Extended
Data Fig 3. Baseline proteomics is paired with screening labs; week 24 and week 72
proteomics with labs of the same visit.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.biomarker_correlation import (
    OLINK_PAIRS,
    SOMA_PAIRS,
    TIMEPOINTS,
    PairSpec,
    fit_pair,
    load_lab_values,
    load_proteomics,
    load_subjects,
)
from analysis.qc_contract import long_table_path
from paths import PHENO_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLINK_NPX = long_table_path("olink")
SOMA_RFU = long_table_path("soma")
ADLB = PHENO_DIR / "adlb.parquet"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_platform(
    pairs: list[PairSpec],
    qc_platform: str,
    parquet_path: Path,
    sample_col: str,
    assay_col: str,
    value_col: str,
    log2_transform: bool,
) -> pl.DataFrame:
    rows = []
    assay_ids = [p.assay_id for p in pairs]

    for prot_visitnum, lab_visitnum, prot_week, lab_week in TIMEPOINTS:
        subjects = load_subjects(qc_platform, sample_col, prot_visitnum)
        if len(subjects) == 0:
            continue
        prot_df = load_proteomics(parquet_path, sample_col, assay_col, value_col, subjects, assay_ids)
        lab_df = load_lab_values(ADLB, list(dict.fromkeys(p.paramcd for p in pairs)), lab_visitnum)

        for spec in pairs:
            fit = fit_pair(spec, prot_df, lab_df, assay_col, value_col, log2_transform)
            if fit is None:
                continue
            rows.append(
                {
                    assay_col: spec.assay_id,
                    "Assay": spec.gene,
                    "Lab_Test": spec.lab_name,
                    "Proteomics_Week": prot_week,
                    "Lab_Week": lab_week,
                    "Spearman_Rho": fit.rho,
                    "Spearman_P": fit.pvalue,
                    "N": fit.n,
                }
            )

    return pl.DataFrame(rows).sort(
        "Proteomics_Week", "Spearman_Rho", descending=[False, True]
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the Olink and SomaScan biomarker correlations.

    Reads the QC long tables (Olink NPX; SomaScan RFU, log2-transformed) and the clinical
    labs, and fits every assay-lab pair at each timepoint through
    ``analysis.biomarker_correlation``.
    """
    olink_df = _build_platform(
        OLINK_PAIRS,
        "olink",
        OLINK_NPX,
        "SampleID",
        "OlinkID",
        "NPX",
        log2_transform=False,
    )
    soma_df = _build_platform(
        SOMA_PAIRS,
        "soma",
        SOMA_RFU,
        "SampleId",
        "SeqId",
        "RFU",
        log2_transform=True,
    )
    return [
        ("Biomarker Corr (Olink)", olink_df),
        ("Biomarker Corr (SomaScan)", soma_df),
    ]


if __name__ == "__main__":
    for name, df in build():
        print(f"\n=== {name} ===")
        print(df)
