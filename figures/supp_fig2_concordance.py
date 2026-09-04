"""Cross-platform concordance GMM histogram (supplemental figure).

Distribution of per-pair Spearman correlations between Olink (NPX) and
SomaScan (log2 RFU) baseline measurements, with the 2-component GMM
labelling pairs as concordant (high) vs discordant (low). The threshold τ
where P(concordant | rho) = 0.5 separates the two groups.

Source: analysis/outputs/platform_concordance.parquet
  - 585 baseline subjects (matched across platforms, sample QC pass)
  - 4,510 assay pairs
  - GMM means: μ₀=0.078 (discordant), μ₁=0.646 (concordant)
  - Threshold τ=0.348 → 1,425 / 4,510 (31.6%) concordant
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import ultraplot as uplt
from scipy import stats as sp_stats
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _internal._common import (  # noqa: E402
    ANALYSIS_DIR,
    FS_AUX,
    FS_EMPH,
    FS_LEGEND,
    c,
    init_figure_theme,
    save_figure,
)

DATA_PATH = ANALYSIS_DIR / "outputs" / "platform_concordance.parquet"


# ---------------------------------------------------------------------------
# GMM refit (matches analysis/scripts/platform_concordance.py exactly)
# ---------------------------------------------------------------------------


def refit_gmm(rho: np.ndarray) -> dict:
    """Refit the 2-component GMM to recover means/stds/weights for the curves.

    Mirrors fit_concordance_gmm() in analysis/scripts/platform_concordance.py so the
    overlay densities match the parquet's labels exactly.
    """
    X = rho.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=10,
        max_iter=500,
        reg_covar=5e-3,
    )
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    order = np.argsort(means)
    return {
        "means": means[order],
        "stds": stds[order],
        "weights": weights[order],
    }


def threshold_at_p_half(
    means: np.ndarray, stds: np.ndarray, weights: np.ndarray
) -> float:
    margin = 0.5 * (means[1] - means[0])
    x_grid = np.linspace(means[0] - margin, means[1] + margin, 10000)
    d0 = weights[0] * sp_stats.norm.pdf(x_grid, means[0], stds[0])
    d1 = weights[1] * sp_stats.norm.pdf(x_grid, means[1], stds[1])
    total = d0 + d1
    p_high = np.where(total > 0, d1 / total, 0)
    cross_idx = np.where(np.diff(np.sign(p_high - 0.5)))[0]
    if len(cross_idx) == 0:
        return float(np.mean(means))
    midpoint = np.mean(means)
    best = cross_idx[np.argmin(np.abs(x_grid[cross_idx] - midpoint))]
    return float(x_grid[best])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure():
    init_figure_theme()

    color_disc = "#BFC4C9"  # light gray for the low/discordant distribution
    color_disc_line = c["bold_grey"]
    color_conc = c["red"]
    color_thresh = c["black"]

    df = pl.read_parquet(DATA_PATH)
    rho = df["SpearmanRho"].to_numpy()
    concordant = df["Concordant"].to_numpy()
    rho_disc = rho[~concordant]
    rho_conc = rho[concordant]

    gmm = refit_gmm(rho)
    means, stds, weights = gmm["means"], gmm["stds"], gmm["weights"]
    tau = threshold_at_p_half(means, stds, weights)

    n_total = len(rho)
    n_conc = int(concordant.sum())

    fig, ax = uplt.subplots(journal="nat1", refaspect=1.4)

    # Histogram: stacked discordant + concordant on the same bin grid so the
    # bar tops add to the total counts and the colors clearly mark each
    # GMM-assigned subgroup.
    bins = np.linspace(rho.min() - 0.01, rho.max() + 0.01, 50)
    ax.hist(
        [rho_disc, rho_conc],
        bins=bins,
        stacked=True,
        color=[color_disc, color_conc],
        edgecolor="white",
        linewidth=0.3,
        label=[
            f"Discordant ($n$={n_total - n_conc:,})",
            f"Concordant ($n$={n_conc:,})",
        ],
        zorder=2,
    )

    # GMM component density curves — convert pdf → expected count per bin width
    x_grid = np.linspace(rho.min(), rho.max(), 600)
    bin_width = bins[1] - bins[0]
    pdf_disc = (
        weights[0] * sp_stats.norm.pdf(x_grid, means[0], stds[0]) * n_total * bin_width
    )
    pdf_conc = (
        weights[1] * sp_stats.norm.pdf(x_grid, means[1], stds[1]) * n_total * bin_width
    )

    ax.plot(x_grid, pdf_disc, color=color_disc_line, lw=1.0, ls="-", zorder=4)
    ax.plot(x_grid, pdf_conc, color=color_conc, lw=1.0, ls="-", zorder=4)

    # Component-mean and threshold lines — all solid
    ymax = ax.get_ylim()[1]
    ax.axvline(means[0], color=color_disc_line, lw=0.7, ls="-", alpha=0.85, zorder=3)
    ax.axvline(means[1], color=color_conc, lw=0.7, ls="-", alpha=0.85, zorder=3)
    ax.axvline(tau, color=color_thresh, lw=0.9, ls="-", zorder=5)

    # Vertical labels beside each line, anchored near the top of the plot
    xrange = (rho.max() + 0.02) - (rho.min() - 0.02)
    label_y = ymax * 0.95
    ax.text(
        means[0] + xrange * 0.012,
        label_y,
        f"$\\mu_0$={means[0]:.3f}",
        color=color_disc_line,
        fontsize=FS_AUX,
        ha="left",
        va="top",
        rotation=90,
    )
    ax.text(
        means[1] + xrange * 0.012,
        label_y,
        f"$\\mu_1$={means[1]:.3f}",
        color=color_conc,
        fontsize=FS_AUX,
        ha="left",
        va="top",
        rotation=90,
    )
    ax.text(
        tau + xrange * 0.012,
        label_y,
        f"$\\tau$={tau:.3f}",
        color=color_thresh,
        fontsize=FS_AUX,
        ha="left",
        va="top",
        rotation=90,
    )

    # Title + gray subtitle (replaces the in-axes pill)
    n_subj = 585  # from analysis pipeline; documented in module docstring
    ax.text(
        0.5,
        1.06,
        "Correlation between Olink NPX and SomaScan log$_2$ RFU",
        transform=ax.transAxes,
        fontsize=FS_EMPH,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        1.02,
        f"{n_total:,} assay pairs · {n_subj} matched baseline subjects",
        transform=ax.transAxes,
        fontsize=FS_AUX,
        color="#666666",
        ha="center",
        va="bottom",
    )

    ax.format(
        xlabel="Spearman $\\rho$",
        ylabel="Number of assay pairs",
        xlim=(rho.min() - 0.02, rho.max() + 0.02),
    )

    ax.legend(
        loc="upper left",
        ncols=1,
        fontsize=FS_LEGEND,
        handletextpad=0.4,
        handlelength=1.0,
        borderaxespad=0.4,
    )

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "supp_fig2_concordance"), formats=("pdf",))
    print("Wrote supp_fig2_concordance.pdf")
