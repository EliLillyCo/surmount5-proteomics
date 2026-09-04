"""Weight-mediation supplementary table.

Restricted to proteins with a significant across-treatment contrast
(Across_Treatment_FDR < 0.05) — mediation decomposition is only
meaningful when there is a treatment effect to partition.

For each platform, emits one row per (protein × visit). Each row carries:
  - Annotation (UniProt / Gene_Symbol / Assay or Target / platform ID)
  - Visit (Wk24, Wk72)
  - ACME (indirect, weight-mediated): estimate, 95% bootstrap CI, P-value, FDR
    (BH-adjusted within across-treatment significant proteins at each visit)
  - ADE (direct, weight-independent): estimate, 95% bootstrap CI, P-value, FDR
    (BH-adjusted within across-treatment significant proteins at each visit)
  - Total effect (= ACME + ADE): estimate, 95% bootstrap CI, P-value
  - Prop_Mediated (point estimate only)
  - Across_Treatment_FDR (BH-adjusted across-treatment FDR at this visit,
    from the same MMRM contrast reported in the across-treatment table)
  - Mediation_Class — one of fully_mediated, partial, direct_only,
    inconsistent, unresolved. "inconsistent" = ACME and ADE both
    significant but opposite signs; "unresolved" = neither ACME nor
    ADE individually significant.

Two sheets: Olink, SomaScan. Contrast is TZP 15 mg (or MTD) vs SEMA 2.4 mg
(or MTD), same as the across-treatment supplement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import COVAR, MMRM_OLINK_DIR, MMRM_SOMA_DIR, mmrm_dir  # noqa: E402

from _annotations import build_marker_annotations  # noqa: E402

VISIT_TO_WEEK = {8: 24, 20: 72}
ACME_FDR_THRESHOLD = 0.05
ADE_FDR_THRESHOLD = 0.05

MED_FILES = {
    "olink": MMRM_OLINK_DIR / "mediationRes" / f"surmount5_{COVAR}_mediation_PCTCHG_WGT.csv",
    "soma": MMRM_SOMA_DIR / "mediationRes" / f"surmount5_{COVAR}_mediation_PCTCHG_WGT.csv",
}

ACTRT_FILE = (
    "surmount5_{covar}_proteomics_olinkAnalysis_acrossTrts_"
    "resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@{week}_py.csv"
)


def _load_across_treatment_fdr(platform: str) -> pl.DataFrame:
    """Load Across_Treatment_FDR keyed by (marker, Week) for Wk24 and Wk72."""
    parts = []
    for w in (24, 72):
        path = (
            mmrm_dir(platform)
            / "finalRes"
            / "acTrt"
            / ACTRT_FILE.format(covar=COVAR, week=w)
        )
        parts.append(
            pl.read_csv(path).select(
                "marker",
                pl.lit(w).cast(pl.Int16).alias("Week"),
                pl.col("fdr").alias("Across_Treatment_FDR"),
            )
        )
    return pl.concat(parts, how="vertical")


ACTRT_FDR_THRESHOLD = 0.05


def _classify(df: pl.DataFrame) -> pl.DataFrame:
    """Add Mediation_Class column. Classes are only assigned to proteins
    whose Across_Treatment_FDR is significant (i.e. there is a treatment
    effect to decompose); otherwise the class is "unresolved". This
    matches the figure scatter-panel legend terminology."""
    cls = (
        pl.when(pl.col("Across_Treatment_FDR") >= ACTRT_FDR_THRESHOLD)
        .then(pl.lit("unresolved"))
        .when(
            (pl.col("ACME_FDR") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_FDR") < ADE_FDR_THRESHOLD)
            & (pl.col("ACME") * pl.col("ADE") < 0)
        )
        .then(pl.lit("inconsistent"))
        .when(
            (pl.col("ACME_FDR") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_FDR") < ADE_FDR_THRESHOLD)
        )
        .then(pl.lit("partial"))
        .when(
            (pl.col("ACME_FDR") < ACME_FDR_THRESHOLD)
            & (pl.col("ADE_FDR") >= ADE_FDR_THRESHOLD)
        )
        .then(pl.lit("fully_mediated"))
        .when(
            (pl.col("ADE_FDR") < ADE_FDR_THRESHOLD)
            & (pl.col("ACME_FDR") >= ACME_FDR_THRESHOLD)
        )
        .then(pl.lit("direct_only"))
        .otherwise(pl.lit("unresolved"))
    )
    return df.with_columns(cls.alias("Mediation_Class"))


def _load_mediation(platform: str) -> pl.DataFrame:
    """Load mediation CSV, restrict to analyzed visits, recompute ACME_FDR
    and ADE_FDR (BH within visit, restricted to across-treatment significant
    proteins), and assign Mediation_Class."""
    df = pl.read_csv(MED_FILES[platform]).filter(
        pl.col("visit").is_in(list(VISIT_TO_WEEK.keys()))
    )

    # Rename pipeline columns to the table's user-facing names.
    df = df.rename(
        {
            "ACME_estimate": "ACME",
            "ACME_ci_lower": "ACME_CI_lower",
            "ACME_ci_upper": "ACME_CI_upper",
            "ACME_pvalue": "ACME_P-value",
            "ADE_estimate": "ADE",
            "ADE_ci_lower": "ADE_CI_lower",
            "ADE_ci_upper": "ADE_CI_upper",
            "ADE_pvalue": "ADE_P-value",
            "total_estimate": "Total",
            "total_ci_lower": "Total_CI_lower",
            "total_ci_upper": "Total_CI_upper",
            "total_pvalue": "Total_P-value",
            "prop_mediated": "Prop_Mediated",
        }
    )
    # Drop pipeline ACME_fdr (computed across all proteins); we recompute.
    if "ACME_fdr" in df.columns:
        df = df.drop("ACME_fdr")

    week_expr = pl.col("visit").replace_strict(VISIT_TO_WEEK, return_dtype=pl.Int16)
    df = df.with_columns(week_expr.alias("Week"))

    # Bring in across-treatment FDR BEFORE computing FDRs so BH can
    # be restricted to the across-treatment significant subset — mediation
    # decomposition is only meaningful when there is a total treatment
    # effect to partition.
    actrt = _load_across_treatment_fdr(platform)
    df = df.join(actrt, on=["marker", "Week"], how="left")

    # Compute ACME_FDR and ADE_FDR per visit (BH restricted to proteins
    # with Across_Treatment_FDR < 0.05).  Non-significant proteins get NaN.
    fdr_parts = []
    for v in sorted(df["visit"].unique().to_list()):
        sub = df.filter(pl.col("visit") == v)
        is_sig = sub["Across_Treatment_FDR"] < ACTRT_FDR_THRESHOLD
        sig_idx = np.where(is_sig.to_numpy())[0]
        sig_sub = sub.filter(is_sig)
        acme_q_sig = false_discovery_control(
            sig_sub["ACME_P-value"].to_numpy(), method="bh"
        )
        ade_q_sig = false_discovery_control(
            sig_sub["ADE_P-value"].to_numpy(), method="bh"
        )
        acme_fdr_full = np.full(sub.height, np.nan)
        ade_fdr_full = np.full(sub.height, np.nan)
        acme_fdr_full[sig_idx] = acme_q_sig
        ade_fdr_full[sig_idx] = ade_q_sig
        fdr_parts.append(
            sub.with_columns(
                pl.Series("ACME_FDR", acme_fdr_full),
                pl.Series("ADE_FDR", ade_fdr_full),
            )
        )
    df = pl.concat(fdr_parts, how="vertical")

    df = _classify(df)
    return df


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    med = _load_mediation(platform)
    id_col = "OlinkID" if platform == "olink" else "SeqId"
    vendor_col = "Assay" if platform == "olink" else "Target"

    round_cols = [
        "ACME",
        "ACME_CI_lower",
        "ACME_CI_upper",
        "ADE",
        "ADE_CI_lower",
        "ADE_CI_upper",
        "Total",
        "Total_CI_lower",
        "Total_CI_upper",
        "Prop_Mediated",
    ]

    return (
        annot.join(med, on="marker", how="inner")
        .rename({"marker": id_col})
        .filter(pl.col("Across_Treatment_FDR") < ACTRT_FDR_THRESHOLD)
        .with_columns(pl.col(round_cols).round(4))
        .select(
            id_col,
            "UniProt",
            "Gene_Symbol",
            vendor_col,
            "Week",
            "ACME",
            "ACME_CI_lower",
            "ACME_CI_upper",
            "ACME_P-value",
            "ACME_FDR",
            "ADE",
            "ADE_CI_lower",
            "ADE_CI_upper",
            "ADE_P-value",
            "ADE_FDR",
            "Total",
            "Total_CI_lower",
            "Total_CI_upper",
            "Total_P-value",
            "Prop_Mediated",
            "Mediation_Class",
        )
        .sort(["Gene_Symbol", id_col, "Week"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
