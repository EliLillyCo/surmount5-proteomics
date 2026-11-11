"""Baseline (Week 0) treatment-arm comparison results (TZP 15mg vs SEMA 2.4mg).

Adjusted for age, sex. Tests whether randomization yielded balanced proteomic
profiles before any treatment exposure. Both platforms model on the log2 scale
(Olink NPX is log2; SomaScan RFU is log2-transformed before MMRM), so the
``diff`` column is equivalent to log2 fold change.

Two sheets: Olink, SomaScan.
"""

from __future__ import annotations


import polars as pl

from analysis.mmrm_outputs import baseline_diff_csv

from tables._annotations import build_marker_annotations, join_annotation


def _load_baseline(platform: str) -> pl.DataFrame:
    path = baseline_diff_csv(platform)
    return pl.read_csv(path).select(
        "marker",
        pl.col("diff").alias("log2FC"),
        pl.col("StdErr").alias("log2FC_SE"),
        pl.col("pVal").alias("P-value"),
        pl.col("fdr").alias("FDR"),
    )


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    diff = _load_baseline(platform)
    id_col = "OlinkID" if platform == "olink" else "SeqId"
    round_cols = ["log2FC", "log2FC_SE"]
    return (
        join_annotation(annot, diff, "baseline difference")
        .rename({"marker": id_col})
        .with_columns(pl.col(round_cols).round(4))
        .sort(id_col)
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the baseline tables."""
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
