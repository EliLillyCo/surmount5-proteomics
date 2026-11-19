"""Statistical helpers and thresholds shared by the pipelines, tables and figures.

Kept free of the mount configuration (``paths``) so any module can import it,
including the figure theme and the unit tests that run without protected data.
"""

from __future__ import annotations

import numpy as np

# Benjamini-Hochberg false-discovery threshold used for every "significant" call.
FDR_THRESHOLD = 0.05
# The threshold as legends, axis labels and table footnotes print it.
FDR_LABEL = f"{FDR_THRESHOLD:g}"
# Every stochastic step in the repo seeds from this value.
SEED = 0


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR over the finite, non-negative entries of ``pvals``.

    NaN, infinite and negative entries are excluded from the family and come
    back as NaN. An input with no valid p-values returns all NaN.

    Args:
        pvals: Array-like of p-values.

    Returns:
        Adjusted p-values (q-values), same shape as ``pvals``.
    """
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p) & (p >= 0)
    if mask.sum() == 0:
        return out
    out[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return out


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_p(p: float, *, digits: int = 1) -> str:
    """Format a p-value for a panel label: ``P = 0.012`` or ``P = 3.2×10⁻⁵``.

    Scientific notation kicks in below 0.01 with ``digits`` mantissa decimals; the
    floor ``P < 10⁻⁹⁹`` avoids printing denormal exponents.

    Args:
        p: The p-value.
        digits: Mantissa decimals in scientific notation.

    Returns:
        The label string.
    """
    if p < 1e-99:
        return "P < 10⁻⁹⁹"
    exp = int(np.floor(np.log10(p)))
    if exp < -2:
        mant = p / 10**exp
        return f"P = {mant:.{digits}f}×10{str(exp).translate(_SUPERSCRIPT)}"
    return f"P = {p:.3f}"
