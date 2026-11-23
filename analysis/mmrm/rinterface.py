"""R interface utilities for MMRM analysis.

Handles conversion between Python/Polars data and R objects, factor releveling,
and emmeans extraction.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from scipy.stats import t as t_dist

from analysis.rinterface import get_r_bindings

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def recalc_pvalues(df: pl.DataFrame, estimate_col: str = "LSmean") -> pl.DataFrame:
    """Recalculate p-values using the t-distribution.

    Args:
        df: Frame with ``StdErr``, ``df`` and the estimate column.
        estimate_col: Column holding the estimate tested against zero.

    Returns:
        ``df`` with a two-sided ``pvalue`` column from the t-distribution at ``df``.
    """
    tvals = (df[estimate_col] / df["StdErr"]).abs().to_numpy()
    dfs = df["df"].to_numpy()
    return df.with_columns(pl.Series("pvalue", 2 * t_dist.sf(tvals, df=dfs)))


# ============================================================================
# EMMEANS EXTRACTION
# ============================================================================


def extract_emmeans_summary(
    emm,
    protein: str,
    is_baseline: bool = False,
    visit_labels: dict[int, int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract lsmeans and contrasts from emmeans object."""
    bindings = get_r_bindings()
    r_summary = bindings["summary"]
    r_as_character = bindings["as_character"]
    r_transform = bindings["transform"]
    r_as_df = bindings["as_df"]
    r_pairs = bindings["pairs"]
    # LSMeans
    emm_summary = r_summary(emm, infer=ro.BoolVector([True, True]))

    # Convert factors to characters
    if is_baseline:
        trt_char = r_as_character(emm_summary.rx2("TRT01A"))
        emm_summary = r_transform(emm_summary, TRT01A=trt_char)
        del trt_char
    else:
        trt_char = r_as_character(emm_summary.rx2("TRT01A"))
        visit_char = r_as_character(emm_summary.rx2("VISITNUM"))
        emm_summary = r_transform(
            emm_summary,
            TRT01A=trt_char,
            VISITNUM=visit_char,
        )
        del trt_char, visit_char

    with pandas2ri.converter.context():
        lsmeans_pd = pandas2ri.rpy2py(emm_summary)
        lsmeans = pl.from_pandas(lsmeans_pd)
    del emm_summary, lsmeans_pd

    lsmeans = (
        lsmeans.rename(
            {
                "p.value": "Probt",
                "emmean": "LSmean",
                "SE": "StdErr",
                "t.ratio": "tValue",
                "lower.CL": "Lower",
                "upper.CL": "Upper",
            }
        )
        .pipe(recalc_pvalues)
        .rename({"pvalue": "pVal", "df": "DF"})
        .with_columns(
            [
                (2 ** pl.col("LSmean")).alias("LSM"),
                (np.log(2) * 2 ** pl.col("LSmean") * pl.col("StdErr")).alias("SE-LSM"),
                pl.lit(protein).alias("marker"),
            ]
        )
    )

    if not is_baseline:
        lsmeans = lsmeans.rename({"VISITNUM": "Time"})

    # Reverse pairwise contrasts: treatment minus control and later visit minus earlier visit.
    contrast_obj = r_pairs(emm, adjust="none", reverse=True)
    contrast_summary = (
        r_summary(contrast_obj) if is_baseline else r_as_df(r_summary(contrast_obj))
    )
    contrast_char = r_as_character(contrast_summary.rx2("contrast"))
    contrast_summary = r_transform(contrast_summary, contrast=contrast_char)
    del contrast_char, contrast_obj

    with pandas2ri.converter.context():
        contrasts_pd = pandas2ri.rpy2py(contrast_summary)
        contrasts = pl.from_pandas(contrasts_pd)
    del contrast_summary, contrasts_pd

    contrasts = (
        contrasts.rename(
            {
                "p.value": "Probt",
                "estimate": "diff",
                "SE": "StdErr",
                "t.ratio": "tValue",
            }
        )
        .pipe(recalc_pvalues, "diff")
        .rename({"pvalue": "pVal"})
        .with_columns(pl.lit(protein).alias("marker"))
    )

    # Parse contrast column and rename to final column names
    if is_baseline:
        contrasts = contrasts.with_columns(
            [
                pl.col("contrast").str.split(" - ").list.get(0).alias("trt"),
                pl.col("contrast").str.split(" - ").list.get(1).alias("_trt"),
            ]
        ).with_columns(
            [
                pl.col("trt").str.replace_all(r"[()]", ""),
                pl.col("_trt").str.replace_all(r"[()]", ""),
            ]
        )
        lsmeans = lsmeans.rename({"TRT01A": "trt"})
    else:
        contrasts = (
            contrasts.with_columns(
                [
                    pl.col("contrast").str.split(" - ").list.get(0).alias("_part1"),
                    pl.col("contrast").str.split(" - ").list.get(1).alias("_part2"),
                ]
            )
            .with_columns(
                [
                    pl.col("_part1").str.extract(r"\S+$", 0).alias("time"),
                    pl.col("_part1").str.extract(r"^(.+)\s+\S+$", 1).alias("trt"),
                    pl.col("_part2").str.extract(r"\S+$", 0).alias("_time"),
                    pl.col("_part2").str.extract(r"^(.+)\s+\S+$", 1).alias("_trt"),
                ]
            )
            .drop(["_part1", "_part2"])
            .with_columns(
                [
                    pl.col("trt").str.replace_all(r"[()]", ""),
                    pl.col("_trt").str.replace_all(r"[()]", ""),
                    pl.col("time").str.replace_all(r"[()]", ""),
                    pl.col("_time").str.replace_all(r"[()]", ""),
                ]
            )
        )
        lsmeans = lsmeans.rename({"TRT01A": "trt", "Time": "time"})

        # Map visit numbers to labels if provided
        if visit_labels:
            visit_map_str = {}
            for k, v in visit_labels.items():
                visit_map_str[str(k)] = str(v)
                visit_map_str[f"VISITNUM{k}"] = str(v)

            lsmeans = lsmeans.with_columns(
                pl.col("time").replace(visit_map_str, default=pl.col("time"))
            )
            contrasts = contrasts.with_columns(
                [
                    pl.col("time").replace(visit_map_str, default=pl.col("time")),
                    pl.col("_time").replace(visit_map_str, default=pl.col("_time")),
                ]
            )

    return lsmeans, contrasts
