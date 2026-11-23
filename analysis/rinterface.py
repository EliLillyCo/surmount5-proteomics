"""Shared rpy2 plumbing: lazy R bindings and pandas -> R factor conversion.

The MMRM engine (``analysis.mmrm``) and the R ``mediation`` package comparison
hand a pandas frame to R with categorical columns as factors and the
treatment reference level fixed. That conversion, and the R package bindings, live
here once; the MMRM engine wraps the conversion with its own required factor
columns. Bindings are resolved lazily so importing an engine does not start R.
"""

from __future__ import annotations

from typing import Any

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

_R_BINDINGS: dict[str, Any] | None = None


def get_r_bindings() -> dict[str, Any]:
    """Lazily import the R packages and base functions the engines call.

    Returns:
        Cached mapping of short names to R packages and functions.
    """
    global _R_BINDINGS
    if _R_BINDINGS is None:
        r_base = importr("base")
        emmeans = importr("emmeans")
        _R_BINDINGS = {
            "mmrm": importr("mmrm"),
            "emmeans": emmeans,
            "gc": r_base.gc,
            "factor": ro.r["factor"],
            "relevel": ro.r["relevel"],
            "summary": ro.r["summary"],
            "as_character": ro.r["as.character"],
            "transform": ro.r["transform"],
            "as_df": ro.r["as.data.frame"],
            "rownames": ro.r["rownames"],
            "lm": ro.r["lm"],
            "pairs": ro.r["pairs"],
            "contrast": emmeans.contrast,
        }
    return _R_BINDINGS


def r_gc(*args, **kwargs):
    """Run R garbage collection.

    Args:
        *args: Passed to R's ``gc``.
        **kwargs: Passed to R's ``gc``.

    Returns:
        The R ``gc`` result.
    """
    return get_r_bindings()["gc"](*args, **kwargs)


def to_r_factors(data_pd, factors: dict[str, str | None]):
    """Convert a pandas frame to an R data.frame with the named columns as factors.

    Levels are the sorted unique values; when a reference level is given it is moved
    first, so it fixes contrast direction and model coding independent of row order.

    Args:
        data_pd: pandas DataFrame.
        factors: column -> reference level (``None`` keeps plain sorted levels).
            ``TRT01A`` must carry an explicit reference.

    Returns:
        The converted R data.frame.

    Raises:
        ValueError: If ``TRT01A`` has no reference, or a reference is not a level.
    """
    if "TRT01A" in factors and factors["TRT01A"] is None:
        raise ValueError("TRT01A needs an explicit reference level; there is no default.")
    bindings = get_r_bindings()
    r_factor = bindings["factor"]

    with pandas2ri.converter.context():
        r_data = pandas2ri.py2rpy(data_pd)

    colnames = list(r_data.names)
    for col, reference in factors.items():
        levels = sorted(str(v) for v in data_pd[col].dropna().unique().tolist())
        if reference is not None:
            if reference not in levels:
                raise ValueError(f"Reference {reference!r} is not a level of {col}: {levels}")
            levels = [reference] + [v for v in levels if v != reference]
        col_idx = colnames.index(col)
        r_data[col_idx] = r_factor(r_data.rx2(col), levels=ro.StrVector(levels))
    return r_data
