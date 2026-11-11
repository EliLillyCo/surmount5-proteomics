"""Weight-mediation supplementary table.

For each platform, one row per (protein x visit) over every protein in the mediation
run. FDR is the unconditional Benjamini-Hochberg of the analytic (Sobel/Wald)
p-values over all proteins of the platform and visit, one family each for the indirect,
direct and total effect; the class is gated on the total-effect FDR afterward. Columns:
  - Annotation (UniProt / Gene_Symbol / Assay or Target / platform ID)
  - Week (24, 72)
  - ACME (indirect, weight-mediated): estimate, 95% bootstrap CI, analytic P-value, FDR
  - ADE (direct, weight-independent): the same
  - Total effect (= ACME + ADE): the same
  - Prop_Mediated (PM = ACME / Total) with its 95% bootstrap percentile CI (Prop_Mediated_CI_lower,
    Prop_Mediated_CI_upper); PM is unstable when the total effect is near zero
  - Mediation_Class: ns (Total_FDR >= 0.05), fully_mediated, partial, direct_only,
    inconsistent (ACME and ADE both significant, opposite signs) or unresolved (Total_FDR
    < 0.05 but neither ACME nor ADE significant)

Two sheets: Olink, SomaScan. Contrast is TZP 15 mg (or MTD) vs SEMA 2.4 mg (or MTD).
The classification is ``analysis.mediation.results.load_platform``, the same one the
figures read.
"""

from __future__ import annotations


import polars as pl

from analysis.mediation.constants import DEFAULT_SIMS
from analysis.mediation.results import load_platform
from analysis.design import POST_BASELINE_VISITS, VISIT_LABELS

from tables._annotations import build_marker_annotations, join_annotation

RENAME = {
    "ACME_estimate": "ACME",
    "ACME_ci_lower": "ACME_CI_lower",
    "ACME_ci_upper": "ACME_CI_upper",
    "ACME_pvalue_analytic": "ACME_P-value",
    "acme_fdr": "ACME_FDR",
    "ADE_estimate": "ADE",
    "ADE_ci_lower": "ADE_CI_lower",
    "ADE_ci_upper": "ADE_CI_upper",
    "ADE_pvalue_analytic": "ADE_P-value",
    "ade_fdr": "ADE_FDR",
    "total_estimate": "Total",
    "total_ci_lower": "Total_CI_lower",
    "total_ci_upper": "Total_CI_upper",
    "total_pvalue_analytic": "Total_P-value",
    "total_fdr": "Total_FDR",
    "prop_mediated": "Prop_Mediated",
    "prop_mediated_ci_lower": "Prop_Mediated_CI_lower",
    "prop_mediated_ci_upper": "Prop_Mediated_CI_upper",
    "mclass": "Mediation_Class",
}
ROUND_COLS = [
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
    "Prop_Mediated_CI_lower",
    "Prop_Mediated_CI_upper",
]
VALUE_COLS = [
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
    "Total_FDR",
    "Prop_Mediated",
    "Prop_Mediated_CI_lower",
    "Prop_Mediated_CI_upper",
    "Mediation_Class",
]


def _load_mediation(platform: str) -> pl.DataFrame:
    """Every visit of a platform's classified mediation results, with the week and table names.

    Raises:
        ValueError: If the results were not all drawn with ``DEFAULT_SIMS`` bootstrap draws,
            the count the table's footnote prints.
    """
    frames = [
        load_platform(platform, visit).with_columns(
            pl.lit(VISIT_LABELS[visit]).cast(pl.Int16).alias("Week")
        )
        for visit in POST_BASELINE_VISITS
    ]
    med = pl.concat(frames, how="vertical")
    draws = sorted(med["sims"].unique().to_list())
    if draws != [DEFAULT_SIMS]:
        raise ValueError(
            f"{platform}: the mediation results carry sims={draws}; "
            f"the table footnote prints {DEFAULT_SIMS:,} draws"
        )
    return med.rename(RENAME)


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    med = _load_mediation(platform)
    id_col = "OlinkID" if platform == "olink" else "SeqId"
    vendor_col = "Assay" if platform == "olink" else "Target"
    return (
        join_annotation(annot, med.drop("Assay", strict=False), "mediation")
        .rename({"marker": id_col})
        .with_columns(pl.col(ROUND_COLS).round(4))
        .select(id_col, "UniProt", "Gene_Symbol", vendor_col, "Week", *VALUE_COLS)
        .sort(["Gene_Symbol", id_col, "Week"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the Olink and SomaScan weight-mediation tables.

    Reads every post-baseline visit of each platform's classified mediation results
    (``analysis.mediation.results.load_platform``) and joins the marker annotations.
    """
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
