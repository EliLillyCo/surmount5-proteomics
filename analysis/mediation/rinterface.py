"""R interface helpers for manuscript mediation analysis."""

from __future__ import annotations

import re
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects import conversion, pandas2ri
from rpy2.robjects.packages import importr

_R_BINDINGS: dict[str, Any] | None = None


def get_r_bindings() -> dict[str, Any]:
    """Lazily import R packages and helper functions."""
    global _R_BINDINGS
    if _R_BINDINGS is None:
        r_base = importr("base")
        _R_BINDINGS = {
            "mediation": importr("mediation"),
            "gc": r_base.gc,
            "summary": ro.r["summary"],
            "as_df": ro.r["as.data.frame"],
            "factor": ro.r["factor"],
            "unique": ro.r["unique"],
            "relevel": ro.r["relevel"],
            "rownames": ro.r["rownames"],
            "lm": ro.r["lm"],
        }
    return _R_BINDINGS


def r_gc(*args, **kwargs):
    """Run R garbage collection."""
    return get_r_bindings()["gc"](*args, **kwargs)


def r_lm(*args, **kwargs):
    """Call R's lm()."""
    return get_r_bindings()["lm"](*args, **kwargs)


def r_mediate(*args, **kwargs):
    """Call mediation::mediate()."""
    return get_r_bindings()["mediation"].mediate(*args, **kwargs)


def _default_reference_level(levels: list[str]) -> str:
    def extract_dose(name: str) -> float:
        numbers = re.findall(r"(\d+)", name)
        if numbers:
            return int(numbers[0])
        return float("inf")

    placebo = next((level for level in levels if "placebo" in level.lower()), None)
    if placebo is not None:
        return placebo
    return min(levels, key=extract_dose)


def to_r_factors(
    data_pd,
    cat_covs: list[str],
    *,
    treat_col: str = "TRT01A",
    treat_ref: str | None = None,
):
    """Convert a pandas frame to R, preserving factor levels and treatment reference."""
    bindings = get_r_bindings()
    r_factor = bindings["factor"]
    r_unique = bindings["unique"]
    r_relevel = bindings["relevel"]

    required_factors = {treat_col, "USUBJID"}
    all_factors = set(cat_covs) | required_factors

    for col in all_factors:
        if col in data_pd.columns and data_pd[col].dtype == object:
            data_pd[col] = data_pd[col].astype("category")

    with pandas2ri.converter.context():
        r_data = pandas2ri.py2rpy(data_pd)

    colnames = list(r_data.names)
    for col in all_factors:
        if col not in colnames:
            continue
        idx = colnames.index(col)
        col_data = r_data.rx2(col)
        r_data[idx] = r_factor(col_data, levels=r_unique(col_data))

    if treat_col in colnames:
        treat = r_data.rx2(treat_col)
        ref_level = treat_ref or _default_reference_level(list(treat.levels))
        idx = colnames.index(treat_col)
        r_data[idx] = r_relevel(treat, ref=ref_level)

    return r_data


def coefficient_table(model):
    """Return the lm summary coefficient table as a pandas DataFrame."""
    bindings = get_r_bindings()
    coef_df = bindings["as_df"](bindings["summary"](model).rx2("coefficients"))
    row_names = list(bindings["rownames"](coef_df))
    with (ro.default_converter + pandas2ri.converter).context():
        out = conversion.get_conversion().rpy2py(coef_df)
    out.index = row_names
    return out


def coefficient_stats(model, term: str) -> dict[str, float]:
    """Extract estimate, standard error, and p-value for one term from an lm fit."""
    coef_df = coefficient_table(model)
    if term not in coef_df.index:
        raise KeyError(
            f"Coefficient '{term}' not found; available terms: {list(coef_df.index)}"
        )
    row = coef_df.loc[term]
    return {
        "estimate": float(row["Estimate"]),
        "std_error": float(row["Std. Error"]),
        "p_value": float(row["Pr(>|t|)"]),
    }
