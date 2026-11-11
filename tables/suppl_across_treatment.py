"""Between-treatment proteomics results (TZP 15mg vs SEMA 2.4mg).

Adjusted for age, sex, baseline protein level. Reports the TZP-vs-SEMA
contrast only — within-arm PCBL values live in the trajectory table.
Two sheets: Olink, SomaScan.
"""

from __future__ import annotations


import polars as pl

from analysis.mmrm_outputs import across_treatment_csv, chg_diff_csv, with_contrast_stderr

from analysis.design import CONTROL_VALUE, POST_BASELINE_WEEKS, TREAT_VALUE

from tables._annotations import build_marker_annotations, join_annotation


def _load_actrt(platform: str, week: int, raw: pl.DataFrame) -> pl.DataFrame:
    path = across_treatment_csv(platform, week)
    # The SE is the contrast's own MMRM StdErr (log2 scale) from rawOutput.
    return (
        with_contrast_stderr(
            pl.read_csv(path), raw, trt=TREAT_VALUE, _trt=CONTROL_VALUE, time=week, _time=week
        )
        .with_columns(
            pl.lit(week).cast(pl.Int16).alias("Week"),
            (pl.col("FC").log(base=2)).alias("log2FC"),
            pl.col("StdErr").alias("log2FC_SE"),
        )
        .select(
            "marker",
            "Week",
            "log2FC",
            "log2FC_SE",
            pl.col("pVal").alias("P-value"),
            pl.col("fdr").alias("FDR"),
        )
    )


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    raw = pl.read_csv(chg_diff_csv(platform))
    combined = pl.concat([_load_actrt(platform, w, raw) for w in POST_BASELINE_WEEKS])

    id_col = "OlinkID" if platform == "olink" else "SeqId"
    round_cols = ["log2FC", "log2FC_SE"]
    return (
        join_annotation(annot, combined, "between-treatment")
        .rename({"marker": id_col})
        .with_columns(pl.col(round_cols).round(4))
        .sort([id_col, "Week"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the between-treatment tables."""
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
