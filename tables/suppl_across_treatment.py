"""Across-treatment proteomics results (TZP 15mg vs SEMA 2.4mg).

Adjusted for age, sex, baseline protein level. Reports the TZP-vs-SEMA
contrast only — within-arm PCBL values live in the trajectory table.
Two sheets: Olink, SomaScan.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import COVAR, mmrm_dir  # noqa: E402

from _annotations import build_marker_annotations  # noqa: E402

_LN2 = math.log(2)


def _load_actrt(platform: str, week: int) -> pl.DataFrame:
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "acTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@{week}_py.csv"
    )
    # SE-FC is on the linear-FC scale; OlinkAnalyze reports FC = 2^diff with
    # SE-FC = FC · ln(2) · SE(log2FC). Invert via the delta method.
    return (
        pl.read_csv(path)
        .with_columns(
            pl.lit(week).cast(pl.Int16).alias("Week"),
            (pl.col("FC").log(base=2)).alias("log2FC"),
            (pl.col("SE-FC") / (pl.col("FC") * _LN2)).alias("log2FC_SE"),
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
    combined = pl.concat([_load_actrt(platform, w) for w in (24, 72)])

    id_col = "OlinkID" if platform == "olink" else "SeqId"
    round_cols = ["log2FC", "log2FC_SE"]
    return (
        annot.join(combined, on="marker", how="inner")
        .rename({"marker": id_col})
        .with_columns(pl.col(round_cols).round(4))
        .sort([id_col, "Week"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the across-treatment tables."""
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
