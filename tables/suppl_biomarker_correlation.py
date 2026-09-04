"""Proteomics–clinical lab biomarker correlations (Spearman) at all timepoints.

Each row is one (platform, assay, lab test, timepoint) combination.
Correlations use the nearest matching lab VISITNUM for each proteomics timepoint:
  - Wk0  (VISITNUM 2 proteomics) ↔ VISITNUM 1 labs (screening)
  - Wk24 (VISITNUM 8 proteomics) ↔ VISITNUM 8 labs
  - Wk72 (VISITNUM 20 proteomics) ↔ VISITNUM 20 labs
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PHENO_DIR, QC_OLINK_DIR, QC_SOMA_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLINK_NPX = QC_OLINK_DIR / "surmount5_olink.npx_long.parquet"
SOMA_RFU = QC_SOMA_DIR / "surmount5_soma.rfu_long.parquet"
OLINK_SAMPLE_QC = QC_OLINK_DIR / "surmount5_olink.sample_qc_summary.tsv"
SOMA_SAMPLE_QC = QC_SOMA_DIR / "surmount5_soma.sample_qc_summary.tsv"
ADLB = PHENO_DIR / "adlb.parquet"

# ---------------------------------------------------------------------------
# Timepoint mapping: (proteomics_visitnum, lab_visitnum, proteomics_week, lab_week)
# ---------------------------------------------------------------------------

TIMEPOINTS = [
    (2, 1, 0, -2),
    (8, 8, 24, 24),
    (20, 20, 72, 72),
]

# ---------------------------------------------------------------------------
# Protein-to-lab mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    gene: str
    assay_id: str
    lab_name: str
    paramcd: str


OLINK_PAIRS = [
    PairSpec("INS-CPEPTIDE", "OID44722", "Insulin", "INSUP28S"),
    PairSpec("GGT1", "OID45132", "GGT", "GGTE17S"),
    PairSpec("GOT1", "OID45135", "AST", "ASTE01S"),
    PairSpec("CST3", "OID45345", "Cystatin C", "CYSTZB5S"),
    PairSpec("PNLIP", "OID45398", "Lipase", "LIPAE09T"),
    PairSpec("CALCA", "OID43443", "Calcitonin", "CLCTGB7S"),
    PairSpec("TSHB", "OID45001", "TSH", "TSHQ04S"),
    PairSpec("FSHB", "OID43729", "FSH", "FSHQ06S"),
]

SOMA_PAIRS = [
    PairSpec("INS", "4883-56", "Insulin", "INSUP28S"),
    PairSpec("GPT", "3709-4", "ALT", "ALTE03S"),
    PairSpec("GOT1", "4912-17", "AST", "ASTE01S"),
    PairSpec("ALPL", "16926-44", "ALP", "ALP47E"),
    PairSpec("CK-MM", "2670-67", "Creatine Kinase", "CKE12T"),
    PairSpec("CK-MB", "3714-49", "Creatine Kinase", "CKE12T"),
    PairSpec("ALB", "18380-78", "Albumin", "ALBF11S"),
    PairSpec("CST3", "2609-59", "Cystatin C", "CYSTZB5S"),
    PairSpec("PNLIP", "15613-16", "Lipase", "LIPAE09T"),
    PairSpec("CALCA", "29468-2", "Calcitonin", "CLCTGB7S"),
    PairSpec("TSH", "3521-16", "TSH", "TSHQ04S"),
    PairSpec("AMY2A", "18917-53", "Amylase", "AMYLE56S"),
    PairSpec("FSH", "3032-11", "FSH", "FSHQ06S"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_subjects(qc_path: Path, sample_col: str, visitnum: int) -> pl.DataFrame:
    return (
        pl.read_csv(qc_path, separator="\t")
        .filter((pl.col("VISITNUM") == visitnum) & (pl.col("Status") != "FAIL"))
        .select(sample_col, "USUBJID")
    )


def _load_proteomics(
    parquet_path: Path,
    sample_col: str,
    assay_col: str,
    value_col: str,
    valid_samples: pl.DataFrame,
    assay_ids: list[str],
) -> pl.DataFrame:
    sample_ids = valid_samples[sample_col].to_list()
    df = (
        pl.scan_parquet(parquet_path)
        .filter(
            pl.col(sample_col).is_in(sample_ids) & pl.col(assay_col).is_in(assay_ids)
        )
        .select(sample_col, assay_col, value_col)
        .collect()
    )
    return df.join(valid_samples, on=sample_col).select("USUBJID", assay_col, value_col)


def _load_lab_values(paramcds: list[str], lab_visitnum: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(ADLB)
        .filter(
            pl.col("PARAMCD").is_in(paramcds)
            & (pl.col("VISITNUM") == lab_visitnum)
            & pl.col("AVAL").is_not_null()
            & (pl.col("AVAL") > 0)
        )
        .group_by("USUBJID", "PARAMCD")
        .agg(pl.col("AVAL").mean())
        .collect()
    )


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------


def _compute_one(
    spec: PairSpec,
    prot_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    assay_col: str,
    value_col: str,
    log2_transform: bool,
) -> dict | None:
    prot = prot_df.filter(pl.col(assay_col) == spec.assay_id).select(
        "USUBJID", value_col
    )
    lab = lab_df.filter(pl.col("PARAMCD") == spec.paramcd).select("USUBJID", "AVAL")
    merged = prot.join(lab, on="USUBJID").drop_nulls()

    if len(merged) < 10:
        return None

    x = merged[value_col].to_numpy().astype(float)
    y = merged["AVAL"].to_numpy().astype(float)

    if log2_transform:
        valid = (x > 0) & np.isfinite(x)
        x, y = x[valid], y[valid]
        x = np.log2(x)

    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        return None

    rho, pvalue = stats.spearmanr(x, y)
    return {"Spearman_Rho": rho, "Spearman_P": pvalue, "N": len(x)}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_platform(
    platform: str,
    pairs: list[PairSpec],
    qc_path: Path,
    parquet_path: Path,
    sample_col: str,
    assay_col: str,
    value_col: str,
    log2_transform: bool,
) -> pl.DataFrame:
    rows = []
    assay_ids = [p.assay_id for p in pairs]

    for prot_visitnum, lab_visitnum, prot_week, lab_week in TIMEPOINTS:
        subjects = _load_subjects(qc_path, sample_col, prot_visitnum)
        if len(subjects) == 0:
            continue
        prot_df = _load_proteomics(
            parquet_path, sample_col, assay_col, value_col, subjects, assay_ids
        )
        all_paramcds = list({p.paramcd for p in pairs})
        lab_df = _load_lab_values(all_paramcds, lab_visitnum)

        for spec in pairs:
            result = _compute_one(
                spec, prot_df, lab_df, assay_col, value_col, log2_transform
            )
            if result is None:
                continue
            rows.append(
                {
                    assay_col: spec.assay_id,
                    "Gene": spec.gene,
                    "Lab_Test": spec.lab_name,
                    "Lab_PARAMCD": spec.paramcd,
                    "Proteomics_Week": prot_week,
                    "Lab_Week": lab_week,
                    **result,
                }
            )

    return pl.DataFrame(rows).sort(
        "Proteomics_Week", "Spearman_Rho", descending=[False, True]
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    olink_df = _build_platform(
        "Olink",
        OLINK_PAIRS,
        OLINK_SAMPLE_QC,
        OLINK_NPX,
        "SampleID",
        "OlinkID",
        "PCNormalizedNPX",
        log2_transform=False,
    )
    soma_df = _build_platform(
        "SomaScan",
        SOMA_PAIRS,
        SOMA_SAMPLE_QC,
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
