"""Analytes whose within-arm responses to TZP and SEMA oppose each other (Supplementary Fig. 6).

Criterion: the between-treatment (TZP − SEMA) difference at the late visit has
FDR < ``FDR_THRESHOLD``, and the within-arm percent change from baseline has opposite signs
in the two arms at both the early and the late visit. The figure shows the ``TOP_K``
opposing analytes with the lowest late-visit between-treatment FDR on each platform, fewer
when fewer qualify.

``EXCLUDED_MARKERS`` holds the assays set aside before ranking (the Olink assays in
``OLINK_EXCLUDED_ASSAYS``).
"""

from __future__ import annotations

import polars as pl

from analysis.mmrm_outputs import across_treatment_csv
from analysis.design import ARMS, EARLY_WEEK, LATE_WEEK
from analysis.output_layout import trajectory_path
from analysis.platform_inputs import EXCLUDED_ASSAYS as EXCLUDED_MARKERS
from analysis.statistics import FDR_THRESHOLD

TOP_K = 4


def opposing_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ``significant`` (late between-treatment FDR) and ``opposing`` (also opposite signs at both visits).

    ``frame`` needs ``marker``, ``fdr_between_late`` and ``pcbl_{TZP,SEMA}_{early,late}``
    (within-arm percent change from baseline at the early and late visit).
    """
    return frame.with_columns(
        (pl.col("fdr_between_late") < FDR_THRESHOLD).alias("significant")
    ).with_columns(
        (
            pl.col("significant")
            & (pl.col("pcbl_TZP_early") * pl.col("pcbl_SEMA_early") < 0)
            & (pl.col("pcbl_TZP_late") * pl.col("pcbl_SEMA_late") < 0)
        ).alias("opposing")
    )


def select_opposing(
    frame: pl.DataFrame, *, top_k: int = TOP_K, excluded: tuple[str, ...] = ()
) -> pl.DataFrame:
    """The ``top_k`` opposing, non-excluded analytes by ascending late between-treatment FDR."""
    return (
        opposing_flags(frame)
        .filter(pl.col("opposing") & ~pl.col("marker").is_in(excluded))
        .sort("fdr_between_late", "marker")
        .head(top_k)
    )


def load_opposing_frame(platform: str) -> pl.DataFrame:
    """Per-marker within-arm changes (trajectory parquet) joined to the late between-treatment FDR."""
    trajectory = pl.read_parquet(trajectory_path(platform))
    between = pl.read_csv(across_treatment_csv(platform, LATE_WEEK)).select(
        "marker", pl.col("fdr").alias("fdr_between_late")
    )
    return trajectory.select(
        "marker",
        *(
            pl.col(f"PCBL_{arm}_{week}").alias(f"pcbl_{arm}_{name}")
            for arm in tuple(ARMS)
            for name, week in (("early", EARLY_WEEK), ("late", LATE_WEEK))
        ),
    ).join(between, on="marker")


def opposing_summary(platform: str) -> dict:
    """Counts and chosen markers for one platform: significant, opposing, and the figure's picks."""
    flagged = opposing_flags(load_opposing_frame(platform))
    chosen = select_opposing(flagged, excluded=EXCLUDED_MARKERS[platform])
    return {
        "n_significant": int(flagged["significant"].sum()),
        "n_opposing": int(flagged["opposing"].sum()),
        "n_total": flagged.height,
        "chosen": chosen["marker"].to_list(),
    }
