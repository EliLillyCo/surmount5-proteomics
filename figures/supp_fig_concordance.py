"""Cross-platform concordance GMM histogram (supplemental figure).

Distribution of per-pair Spearman correlations between Olink (PCNormalizedNPX) and
SomaScan (log2 RFU) baseline measurements, with the 2-component GMM
labeling pairs as concordant (high) vs discordant (low). The threshold τ
where P(concordant | rho) = 0.5 separates the two groups.

Source: analysis/outputs/platform_concordance.parquet (``analysis/pipelines/platform_concordance.py``).
The component weights, means, standard deviations and the threshold τ are the stored parameters
of the fit that labeled the pairs (``analysis.concordance.read_fit``), drawn with the
concordant count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from figures._internal._common import (
    FS_AUX,
    FS_EMPH,
    FS_LEGEND,
    NAT_W1,
    c,
    init_figure_theme,
    panel_figure,
    panel_grid,
    save_figure,
)

from analysis.concordance import read_fit
from analysis.output_layout import PLATFORM_CONCORDANCE

DATA_PATH = PLATFORM_CONCORDANCE


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure():
    """Build Extended Data Fig 2: the distribution of baseline Olink-SomaScan Spearman correlations.

    Reads ``analysis/outputs/platform_concordance.parquet``, stacks the discordant and
    concordant pairs in one histogram and overlays the two GMM component densities and the
    concordance threshold from the stage's stored fit (``read_fit``).

    Returns:
        The figure.
    """
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

    # The stage's stored mixture parameters, for the density curves and the threshold.
    gmm = read_fit(df)
    means, stds, weights = gmm["means"], gmm["stds"], gmm["weights"]
    tau = gmm["threshold"]

    n_total = len(rho)
    n_conc = int(concordant.sum())

    # Single column (89 mm); the axes are 1.4 times as wide as tall.
    fig = panel_figure(NAT_W1, 2.9402)
    (ax,) = panel_grid(fig, [[1]], left=0.4007, right=0.0451, top=0.4026, bottom=0.3532)

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

    # Title and gray subtitle
    n_subj_min, n_subj_max = int(df["NSamples"].min()), int(df["NSamples"].max())
    n_subj = str(n_subj_min) if n_subj_min == n_subj_max else f"{n_subj_min}–{n_subj_max}"
    ax.text(
        0.5,
        1.06,
        "Correlation between Olink plate-control-normalized NPX and SomaScan log₂ RFU",
        transform=ax.transAxes,
        fontsize=FS_EMPH,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        1.02,
        f"{n_total:,} assay pairs · {n_subj} matched baseline subjects per pair",
        transform=ax.transAxes,
        fontsize=FS_AUX,
        color="#666666",
        ha="center",
        va="bottom",
    )

    ax.set_xlabel("Spearman $\\rho$")
    ax.set_ylabel("Number of assay pairs")
    ax.set_xlim(rho.min() - 0.02, rho.max() + 0.02)

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
    save_figure(fig, str(out_dir / "supp_fig_concordance"), formats=("pdf",))
    print("Wrote supp_fig_concordance.pdf")
