"""R interface utilities for MMRM analysis.

Handles conversion between Python/Polars data and R objects, factor releveling,
and emmeans extraction.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

_R_BINDINGS: dict[str, Any] | None = None


def get_r_bindings() -> dict[str, Any]:
    """Lazily import R packages and functions."""
    global _R_BINDINGS
    if _R_BINDINGS is None:
        r_base = importr("base")
        _R_BINDINGS = {
            "mmrm": importr("mmrm"),
            "emmeans": importr("emmeans"),
            "gc": r_base.gc,
            "factor": ro.r["factor"],
            "unique": ro.r["unique"],
            "relevel": ro.r["relevel"],
            "summary": ro.r["summary"],
            "as_character": ro.r["as.character"],
            "transform": ro.r["transform"],
            "as_df": ro.r["as.data.frame"],
            "lm": ro.r["lm"],
            "pairs": ro.r["pairs"],
        }
    return _R_BINDINGS


def r_gc(*args, **kwargs):
    """Run R garbage collection."""
    return get_r_bindings()["gc"](*args, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def recalc_pvalues(df: pl.DataFrame, estimate_col: str = "LSmean") -> pl.DataFrame:
    """Recalculate p-values using t-distribution."""
    from scipy.stats import t as t_dist

    return df.with_columns(
        [
            pl.struct([estimate_col, "StdErr", "df"])
            .map_elements(
                lambda x: 2 * t_dist.sf(abs(x[estimate_col] / x["StdErr"]), df=x["df"]),
                return_dtype=pl.Float64,
            )
            .alias("pvalue")
        ]
    )


# ============================================================================
# FACTOR CONVERSION
# ============================================================================


def to_r_factors(data_pd, cat_covs: list[str]):
    """Convert pandas DataFrame to R with proper factor levels."""
    bindings = get_r_bindings()
    r_factor = bindings["factor"]
    r_unique = bindings["unique"]
    r_relevel = bindings["relevel"]
    required_factors = {"TRT01A", "VISITNUM", "USUBJID"}
    all_factors = set(cat_covs) | required_factors

    for col in all_factors:
        if col in data_pd.columns and data_pd[col].dtype == object:
            data_pd[col] = data_pd[col].astype("category")

    with pandas2ri.converter.context():
        r_data = pandas2ri.py2rpy(data_pd)

    colnames = list(r_data.names)
    for col in all_factors:
        if col in colnames:
            col_idx = colnames.index(col)
            col_data = r_data.rx2(col)
            r_data[col_idx] = r_factor(col_data, levels=r_unique(col_data))
            del col_data

    # Relevel for emmeans contrasts
    if "TRT01A" in colnames:
        trt_col = r_data.rx2("TRT01A")
        trt_levels = list(trt_col.levels)

        def extract_dose(trt_name):
            numbers = re.findall(r"(\d+)", trt_name)
            if numbers:
                return int(numbers[0])
            return float("inf")

        def is_placebo(trt_name):
            return "placebo" in trt_name.lower()

        # Set reference: Placebo first, then lowest dose
        placebo_level = next((l for l in trt_levels if is_placebo(l)), None)
        if placebo_level:
            ref_level = placebo_level
        else:
            ref_level = min(trt_levels, key=extract_dose)

        trt_idx = colnames.index("TRT01A")
        r_data[trt_idx] = r_relevel(trt_col, ref=ref_level)
        del trt_col

    if "VISITNUM" in colnames:
        visit_col = r_data.rx2("VISITNUM")
        visit_levels = list(visit_col.levels)
        if visit_levels:
            latest_visit = str(max(int(v) for v in visit_levels))
            visit_idx = colnames.index("VISITNUM")
            r_data[visit_idx] = r_relevel(visit_col, ref=latest_visit)
        del visit_col

    return r_data


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

    # Contrasts — use pairs() instead of contrast(method="pairwise")
    # because only pairs() supports the reverse parameter
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
