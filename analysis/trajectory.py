"""Within-arm trajectory taxonomy of every protein's change-from-baseline response.

The class map and the decision rules below are applied to each arm of the active-comparator
trial (TZP and SEMA) on the log2 within-arm change estimates of the MMRM.

Inputs per arm, per protein: the within-arm LSmean change from baseline with its StdErr
and DF at the early and late visit (weeks 24 and 72), and the formal within-arm
late-minus-early (week 72 - week 24) contrast ``diff`` with its StdErr and df. P-values
are the two-sided t-test of estimate / StdErr at the stated df; the Benjamini-Hochberg
FDR is taken genome-wide within each (arm, window): the early change, the late change and
the late-vs-early contrast each form one family.

Decision tree (signs from the log2 estimates):

    sig_e, sig_l   = FDR(early) < 0.05, FDR(late) < 0.05
    dir_e, dir_l   = sign(LSmean_early), sign(LSmean_late)
    sig_within     = FDR(late - early contrast) < 0.05
    growth_up      = sig_within and diff > 0   (the level rose from week 24 to week 72)
    growth_down    = sig_within and diff < 0   (the level fell further)

    neither significant                                        -> Null
    sig_e only                                                 -> Transient up/down (dir_e)
    sig_l only                                                 -> Late-onset up/down (dir_l)
    both, opposite directions                                  -> Reversal up-down / down-up
    both, same direction (+) and growth_up                     -> Progressive up
    both, same direction (-) and growth_down                   -> Progressive down
    both, same direction, otherwise                            -> Sustained up/down
"""

from __future__ import annotations

import polars as pl
from scipy.stats import t as t_dist

from analysis.statistics import FDR_THRESHOLD, bh_fdr

# Class of a protein with no significant trajectory.
NULL_CLASS = "Null"

# Class keys (as the rules below name them) -> the display labels the figures order and
# color by.
TRAJECTORY_CLASS_LABELS: dict[str, str] = {
    "Progressive_up": "Progressive ↑",
    "Progressive_down": "Progressive ↓",
    "Sustained_up": "Sustained ↑",
    "Sustained_down": "Sustained ↓",
    "Transient_up": "Transient ↑",
    "Transient_down": "Transient ↓",
    "LateOnset_up": "Late-onset ↑",
    "LateOnset_down": "Late-onset ↓",
    "Reversal_up_down": "Reversal ↑↓",
    "Reversal_down_up": "Reversal ↓↑",
    NULL_CLASS: NULL_CLASS,
}


def _t_pvalue(estimate: pl.Series, se: pl.Series, df: pl.Series) -> pl.Series:
    """Two-sided t-test p-values of ``estimate / se`` at ``df`` degrees of freedom."""
    t_stat = (estimate / se).abs().to_numpy()
    return pl.Series(2 * t_dist.sf(t_stat, df=df.to_numpy()))


def _with_fdr(df: pl.DataFrame, estimate: str, se: str, dof: str, name: str) -> pl.DataFrame:
    """Add ``<name>_p`` (t-test) and ``<name>_fdr`` (BH over all rows of ``df``)."""
    p = _t_pvalue(df[estimate], df[se], df[dof])
    return df.with_columns(p.alias(f"{name}_p"), pl.Series(f"{name}_fdr", bh_fdr(p.to_numpy())))


def classify_arm(
    early: pl.DataFrame, late: pl.DataFrame, contrast: pl.DataFrame
) -> pl.DataFrame:
    """Classify the trajectory of every protein for one arm.

    Args:
        early: ``marker, LSmean, StdErr, DF`` of the arm's change at the early visit.
        late: Same at the late visit.
        contrast: ``marker, diff, StdErr, df`` of the arm's late-minus-early contrast.

    Returns:
        One row per marker present in all three inputs, sorted by marker: ``marker``,
        ``chg_early``, ``fdr_early``, ``chg_late``, ``fdr_late``, ``chg_late_vs_early``,
        ``fdr_late_vs_early`` and ``trajectory`` (a display label).
    """
    e = _with_fdr(early, "LSmean", "StdErr", "DF", "early").select(
        "marker", pl.col("LSmean").alias("chg_early"), "early_fdr"
    )
    l = _with_fdr(late, "LSmean", "StdErr", "DF", "late").select(
        "marker", pl.col("LSmean").alias("chg_late"), "late_fdr"
    )
    c = _with_fdr(contrast, "diff", "StdErr", "df", "within").select(
        "marker", pl.col("diff").alias("chg_late_vs_early"), "within_fdr"
    )
    combined = (
        e.join(l, on="marker", how="inner").join(c, on="marker", how="inner").sort("marker")
    )

    sig_e = pl.col("early_fdr") < FDR_THRESHOLD
    sig_l = pl.col("late_fdr") < FDR_THRESHOLD
    pos_e = pl.col("chg_early") > 0
    pos_l = pl.col("chg_late") > 0
    same_dir = pos_e == pos_l
    sig_within = pl.col("within_fdr") < FDR_THRESHOLD
    growth_up = sig_within & (pl.col("chg_late_vs_early") > 0)
    growth_down = sig_within & (pl.col("chg_late_vs_early") < 0)
    both = sig_e & sig_l

    key = (
        # Reversal precedes the same-direction classes: it needs opposite signs.
        pl.when(both & pos_e & ~pos_l)
        .then(pl.lit("Reversal_up_down"))
        .when(both & ~pos_e & pos_l)
        .then(pl.lit("Reversal_down_up"))
        .when(both & same_dir & growth_up & pos_e)
        .then(pl.lit("Progressive_up"))
        .when(both & same_dir & growth_down & ~pos_e)
        .then(pl.lit("Progressive_down"))
        .when(both & same_dir & pos_e)
        .then(pl.lit("Sustained_up"))
        .when(both & same_dir & ~pos_e)
        .then(pl.lit("Sustained_down"))
        .when(sig_e & ~sig_l & pos_e)
        .then(pl.lit("Transient_up"))
        .when(sig_e & ~sig_l & ~pos_e)
        .then(pl.lit("Transient_down"))
        .when(~sig_e & sig_l & pos_l)
        .then(pl.lit("LateOnset_up"))
        .when(~sig_e & sig_l & ~pos_l)
        .then(pl.lit("LateOnset_down"))
        .otherwise(pl.lit(NULL_CLASS))
    )
    return combined.with_columns(
        key.replace_strict(TRAJECTORY_CLASS_LABELS, return_dtype=pl.String).alias("trajectory")
    ).select(
        "marker",
        "chg_early",
        pl.col("early_fdr").alias("fdr_early"),
        "chg_late",
        pl.col("late_fdr").alias("fdr_late"),
        "chg_late_vs_early",
        pl.col("within_fdr").alias("fdr_late_vs_early"),
        "trajectory",
    )


def _pcbl(log2_change: pl.Expr) -> pl.Expr:
    """Percent change from baseline of a log2 change (matches the MMRM PCBL tables)."""
    return (2**log2_change - 1) * 100


def classify_platform(
    lsmeans: pl.DataFrame,
    diff: pl.DataFrame,
    annotation: pl.DataFrame,
    arms: dict[str, str],
    early: int,
    late: int,
) -> pl.DataFrame:
    """Classify every arm of one platform and merge into one wide table.

    Args:
        lsmeans: MMRM ``rawOutput/*_chg_lsmeans.csv`` (``marker, time, trt, LSmean, StdErr, DF``).
        diff: MMRM ``rawOutput/*_chg_diff.csv`` (``marker, time, trt, _time, _trt, diff, StdErr, df``).
        annotation: ``marker, Assay`` (the assay label).
        arms: Short arm name -> TRT01A level, in output order (e.g. ``{"TZP": ..., "SEMA": ...}``).
        early: Early visit label (week 24).
        late: Late visit label (week 72).

    Returns:
        One row per marker: ``marker``, ``Assay``, then for each arm
        ``trajectory_<arm>``, ``PCBL_<arm>_<early>``, ``PCBL_<arm>_<late>``, followed
        by ``fdr_<arm>_<early>``, ``fdr_<arm>_<late>`` and the late-vs-early log2
        contrast and FDR (``chg_<arm>_<late>vs<early>``, ``fdr_<arm>_<late>vs<early>``).

    Raises:
        ValueError: When an arm lacks exactly one late-minus-early contrast per marker.
    """
    lsm = lsmeans.with_columns(pl.col("time").cast(pl.Int64))
    dif = diff.with_columns(pl.col("time").cast(pl.Int64), pl.col("_time").cast(pl.Int64))
    result = annotation.select("marker", "Assay").sort("marker")
    for arm, trt in arms.items():
        arm_lsm = lsm.filter(pl.col("trt") == trt)
        contrast = dif.filter(
            (pl.col("trt") == trt)
            & (pl.col("_trt") == trt)
            & (pl.col("time") == late)
            & (pl.col("_time") == early)
        )
        n_markers = arm_lsm["marker"].n_unique()
        if contrast.height != n_markers or contrast["marker"].n_unique() != n_markers:
            raise ValueError(
                f"{arm} ({trt}): expected one week-{late} - week-{early} contrast per marker "
                f"({n_markers}), found {contrast.height}; regenerate the MMRM tree "
                "(the within-arm contrast is late minus early)"
            )
        classified = classify_arm(
            arm_lsm.filter(pl.col("time") == early),
            arm_lsm.filter(pl.col("time") == late),
            contrast,
        )
        result = result.join(
            classified.select(
                "marker",
                pl.col("trajectory").alias(f"trajectory_{arm}"),
                _pcbl(pl.col("chg_early")).alias(f"PCBL_{arm}_{early}"),
                _pcbl(pl.col("chg_late")).alias(f"PCBL_{arm}_{late}"),
                pl.col("fdr_early").alias(f"fdr_{arm}_{early}"),
                pl.col("fdr_late").alias(f"fdr_{arm}_{late}"),
                pl.col("chg_late_vs_early").alias(f"chg_{arm}_{late}vs{early}"),
                pl.col("fdr_late_vs_early").alias(f"fdr_{arm}_{late}vs{early}"),
            ),
            on="marker",
            how="inner",
        )

    lead = ["marker", "Assay"]
    per_arm = [
        c
        for a in arms
        for c in (
            f"trajectory_{a}",
            f"PCBL_{a}_{early}",
            f"PCBL_{a}_{late}",
        )
    ]
    extra = [
        c
        for a in arms
        for c in (
            f"fdr_{a}_{early}",
            f"fdr_{a}_{late}",
            f"chg_{a}_{late}vs{early}",
            f"fdr_{a}_{late}vs{early}",
        )
    ]
    return result.select(lead + per_arm + extra)
