"""Two-component Gaussian mixture of the cross-platform Spearman correlations (Extended Data Fig 2).

``analysis/pipelines/platform_concordance.py`` fits it to label each Olink x SomaScan assay pair
concordant (posterior of the high component >= 0.5) and writes the labels to
``platform_concordance.parquet`` with the fit's weights, means, standard deviations and
threshold as constant columns (``FIT_COLUMNS``, written by ``fit_columns``). The ED2 figure draws
the component densities and the threshold from those columns (``read_fit``), so the curves and
the labels come from one fit. The fit is order-invariant within float rounding.

A fit that does not converge raises, and so does one whose posterior never crosses 0.5
between the component means.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats
from sklearn.mixture import GaussianMixture

from analysis.statistics import SEED

# GaussianMixture settings; reg_covar regularizes component covariance estimates.
GMM_SETTINGS: dict[str, object] = {
    "n_components": 2,
    "covariance_type": "full",
    "n_init": 10,
    "max_iter": 500,
    "reg_covar": 5e-3,
}
# Points of the grid the P(high) = 0.5 crossing is located on.
THRESHOLD_GRID_POINTS = 10_000
# Columns of platform_concordance.parquet that hold the fit, each constant over the rows:
# column -> (fit key, component index; None for the scalar threshold). Component 0 is the low
# (discordant) one, component 1 the high (concordant) one.
FIT_COLUMNS: dict[str, tuple[str, int | None]] = {
    "GMM_Weight_Low": ("weights", 0),
    "GMM_Weight_High": ("weights", 1),
    "GMM_Mean_Low": ("means", 0),
    "GMM_Mean_High": ("means", 1),
    "GMM_SD_Low": ("stds", 0),
    "GMM_SD_High": ("stds", 1),
    "GMM_Threshold": ("threshold", None),
}


def threshold_at_p_half(means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float:
    """The correlation where P(high component | rho) = 0.5, nearest the midpoint of the means.

    Args:
        means: Component means, low component first.
        stds: Component standard deviations, same order.
        weights: Mixture weights, same order.

    Returns:
        The crossing on a grid spanning the means plus half their gap on each side.

    Raises:
        ValueError: When the posterior never crosses 0.5 on that grid.
    """
    margin = 0.5 * (means[1] - means[0])
    x_grid = np.linspace(means[0] - margin, means[1] + margin, THRESHOLD_GRID_POINTS)
    d0 = weights[0] * sp_stats.norm.pdf(x_grid, means[0], stds[0])
    d1 = weights[1] * sp_stats.norm.pdf(x_grid, means[1], stds[1])
    total = d0 + d1
    p_high = np.where(total > 0, d1 / total, 0)
    cross_idx = np.where(np.diff(np.sign(p_high - 0.5)))[0]
    if len(cross_idx) == 0:
        raise ValueError(
            f"P(high component) never crosses 0.5 between {x_grid[0]:.4f} and {x_grid[-1]:.4f} "
            f"(means {means.tolist()}, sds {stds.tolist()}, weights {weights.tolist()})"
        )
    midpoint = np.mean(means)
    best = cross_idx[np.argmin(np.abs(x_grid[cross_idx] - midpoint))]
    return float(x_grid[best])


def fit_concordance_gmm(spearman_values: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit the two-component mixture and each pair's posterior of the high component.

    Args:
        spearman_values: One Spearman correlation per assay pair.

    Returns:
        ``means``, ``stds`` and ``weights`` (low component first), ``threshold`` (see
        ``threshold_at_p_half``) and ``posteriors`` (P(high) per input value, input order).

    Raises:
        RuntimeError: When the best of the ``n_init`` fits did not converge in ``max_iter``
            iterations.
        ValueError: When the posterior never crosses 0.5 (``threshold_at_p_half``).
    """
    X = spearman_values.reshape(-1, 1)
    gmm = GaussianMixture(random_state=SEED, **GMM_SETTINGS)
    gmm.fit(X)
    if not gmm.converged_:
        raise RuntimeError(
            f"the concordance mixture did not converge in {GMM_SETTINGS['max_iter']} iterations "
            f"({len(spearman_values)} values)"
        )

    # Component 0 is the low (discordant) one, component 1 the high (concordant) one.
    order = np.argsort(gmm.means_.flatten())
    means = gmm.means_.flatten()[order]
    stds = np.sqrt(gmm.covariances_.flatten())[order]
    weights = gmm.weights_[order]
    posteriors = gmm.predict_proba(X)[:, order]

    return {
        "means": means,
        "stds": stds,
        "weights": weights,
        "threshold": threshold_at_p_half(means, stds, weights),
        "posteriors": posteriors[:, 1],
    }




def fit_columns(fit: dict[str, np.ndarray | float]) -> list[pl.Expr]:
    """The fit's parameters as the constant ``FIT_COLUMNS`` of the concordance table.

    Args:
        fit: The result of ``fit_concordance_gmm``.

    Returns:
        One Float64 literal per ``FIT_COLUMNS`` entry, in its order.
    """
    return [
        pl.lit(float(fit[key] if index is None else fit[key][index]), dtype=pl.Float64).alias(name)
        for name, (key, index) in FIT_COLUMNS.items()
    ]


def read_fit(pairs: pl.DataFrame) -> dict[str, np.ndarray | float]:
    """The stored fit of the concordance table.

    Args:
        pairs: ``platform_concordance.parquet``, with its ``FIT_COLUMNS``.

    Returns:
        ``weights``, ``means`` and ``stds`` (arrays, low component first) and ``threshold``, as
        ``fit_concordance_gmm`` returns them, without the posteriors.

    Raises:
        ValueError: When a fit column holds more than one value.
    """
    columns = pairs.select(list(FIT_COLUMNS))
    varying = [name for name in FIT_COLUMNS if columns[name].n_unique() != 1]
    if varying:
        raise ValueError(f"the fit columns {varying} are not constant over the rows")
    first = columns.row(0, named=True)
    out: dict[str, list[float] | float] = {}
    for name, (key, index) in FIT_COLUMNS.items():
        if index is None:
            out[key] = first[name]
        else:
            out.setdefault(key, []).append(first[name])
    return {key: np.array(v) if isinstance(v, list) else v for key, v in out.items()}
