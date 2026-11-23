"""Supplemental figure: proteomics assay vs. clinical lab biomarker correlations.

Scatter plots of baseline protein values (NPX or log₂RFU) vs. matched clinical
lab measurements, with Spearman ρ annotations.  Layout: 5 rows × 5 columns —
Olink on top (rows 0–1, 8 pairs; last two cells of row 1 hidden), SomaScan
below (rows 2–4, 13 pairs ranked by |ρ|; last two cells of row 4 hidden).

Baseline proteomics is paired with screening labs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import polars as pl
from scipy import stats

from figures._internal._common import (
    BANNER_HEIGHT_IN,
    FS_AUX,
    FS_EMPH,
    FS_PANEL,
    NAT_W2,
    PHENO_DIR,
    c,
    init_figure_theme,
    panel_figure,
    panel_grid,
    save_figure,
)
from analysis.biomarker_correlation import (
    LAB_SCREENING_VISITNUM,
    OLINK_PAIRS,
    SOMA_PAIRS,
    PairSpec,
    fit_pair,
    load_lab_values,
    load_proteomics,
    load_subjects,
)
from analysis.design import BASELINE_VISITNUM
from analysis.qc_contract import long_table_path

# ---------------------------------------------------------------------------
# Lab-assay color mapping (consistent across platforms, all Lilly palette)
# ---------------------------------------------------------------------------


LAB_COLORS = {
    "FSH": "#5A9BD5",
    "GGT": c["bold_brown"],
    "Cystatin C": c["bold_green"],
    "Calcitonin": c["bold_grey"],
    "TSH": c["red"],
    "Lipase": c["vibrant_coral"],
    "Insulin": c["bold_blue"],
    "AST": c["vibrant_gold"],
    "ALT": "#E8873D",
    "ALP": "#7A5195",
    "Albumin": "#3CB371",
    "Creatine Kinase": c["black"],
    "Amylase": "#E07B9A",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLINK_NPX = long_table_path("olink")
SOMA_RFU = long_table_path("soma")
ADLB = PHENO_DIR / "adlb.parquet"

# ---------------------------------------------------------------------------
# Correlations: the pairs, loaders and Spearman fit are analysis.biomarker_correlation,
# shared with Tables S7/S8; the figure adds the log10(lab) regression line it draws.
# ---------------------------------------------------------------------------


@dataclass
class CorrResult:
    """One protein-lab pair's fit: Spearman statistics, the paired values and the regression line.

    Attributes:
        spec: The assay-lab pair.
        rho: Spearman correlation.
        pvalue: Its P value.
        n: Number of paired participants.
        protein_vals: Protein values (NPX or log2 RFU).
        lab_vals: Lab values (linear scale).
        slope: OLS slope of log10(lab) on the protein value (the drawn fit line).
        intercept: Its intercept.
    """
    spec: PairSpec
    rho: float
    pvalue: float
    n: int
    protein_vals: np.ndarray
    lab_vals: np.ndarray
    slope: float
    intercept: float


def compute_correlations(
    pairs: list[PairSpec],
    prot_df: pl.DataFrame,
    lab_df: pl.DataFrame,
    assay_col: str,
    value_col: str,
    log2_transform: bool = False,
) -> list[CorrResult]:
    """Fit every pair with ``analysis.biomarker_correlation.fit_pair`` and the log10(lab) ~ protein line.

    Raises ``ValueError`` for a listed pair that ``fit_pair`` cannot fit (it returns ``None``).

    Args:
        pairs: Assay-lab pairs to fit.
        prot_df: Long proteomics values at the timepoint.
        lab_df: Long lab values at the matched visit.
        assay_col: Assay key column (``OlinkID`` or ``SeqId``).
        value_col: Value column (``NPX`` or ``RFU``).
        log2_transform: Take log2 of the protein value first (SomaScan RFU).

    Returns:
        The fits, sorted by decreasing |rho|, then gene, assay and lab name.
    """
    results = []
    for spec in pairs:
        fit = fit_pair(spec, prot_df, lab_df, assay_col, value_col, log2_transform)
        if fit is None:
            raise ValueError(
                f"{spec.lab_name} vs {spec.assay_id}: too few subjects to fit a listed pair"
            )
        slope, intercept, *_ = stats.linregress(fit.protein_vals, np.log10(fit.lab_vals))
        results.append(
            CorrResult(
                spec=spec,
                rho=fit.rho,
                pvalue=fit.pvalue,
                n=fit.n,
                protein_vals=fit.protein_vals,
                lab_vals=fit.lab_vals,
                slope=slope,
                intercept=intercept,
            )
        )

    results.sort(
        key=lambda r: (-abs(r.rho), r.spec.gene, r.spec.assay_id, r.spec.lab_name)
    )
    return results


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

# Layout in inches, so the figure fits the 180 x 170 mm Extended Data box. The
# scatter panels are not forced square: 5 rows of 1.16 in wide panels are about
# 0.8 in tall.
FIG_H_IN = 6.62
PLATFORM_BANNER_GAP_IN = 0.15  # axes top -> platform banner (clears the subplot banner)
SUBPLOT_BANNER_H_IN = 0.12
PANEL_LABEL_GAP_IN = 0.04
ROW_GAP_IN = 0.33  # holds the x tick labels and the next row's subplot banner
BLOCK_GAP_IN = 0.84  # Olink x tick labels + "NPX" label + SomaScan banner stack


def _add_platform_banner(fig, ax_top_left, ax_top_right, label: str):
    """Full-width gray banner above a column group."""
    bb_left = ax_top_left.get_position()
    bb_right = ax_top_right.get_position()
    banner_height = BANNER_HEIGHT_IN / fig.get_figheight()
    banner_y = bb_left.y1 + PLATFORM_BANNER_GAP_IN / fig.get_figheight()
    banner = mpatches.FancyBboxPatch(
        (bb_left.x0, banner_y),
        bb_right.x1 - bb_left.x0,
        banner_height,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = (bb_left.x0 + bb_right.x1) / 2
    mid_y = banner_y + banner_height / 2
    fig.text(
        mid_x,
        mid_y,
        label,
        fontsize=FS_EMPH,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _add_subplot_banner(fig, ax, gene: str, assay_id: str):
    """Small gray banner above an individual subplot: bold gene · gray assay ID."""
    bb = ax.get_position()
    banner_height = SUBPLOT_BANNER_H_IN / fig.get_figheight()
    banner_y = bb.y1 + 0.001
    banner = mpatches.FancyBboxPatch(
        (bb.x0, banner_y),
        bb.x1 - bb.x0,
        banner_height,
        boxstyle="square,pad=0",
        facecolor="#F2F2F2",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = (bb.x0 + bb.x1) / 2
    mid_y = banner_y + banner_height / 2
    fig.text(
        mid_x - 0.002,
        mid_y,
        gene,
        fontsize=FS_AUX,
        fontweight="bold",
        ha="right",
        va="center",
    )
    fig.text(
        mid_x + 0.002,
        mid_y,
        f"· {assay_id}",
        fontsize=FS_AUX,
        color="#666666",
        ha="left",
        va="center",
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_panel(
    axs,
    results: list[CorrResult],
    x_label: str,
    bottom_row_indices: set[int],
) -> None:
    """Fill axes with scatter + regression for results, colored by lab assay."""
    for i, ax in enumerate(axs):
        if i >= len(results):
            ax.set_visible(False)
            continue

        r = results[i]
        color = LAB_COLORS[r.spec.lab_name]
        ax.scatter(
            r.protein_vals,
            r.lab_vals,
            s=8,
            alpha=0.5,
            color=color,
            edgecolors="white",
            linewidths=0.3,
        )

        x_fit = np.linspace(r.protein_vals.min(), r.protein_vals.max(), 100)
        y_fit = 10 ** (r.slope * x_fit + r.intercept)
        ax.plot(x_fit, y_fit, color="black", alpha=0.5, lw=1.0)
        ax.set_xlim(x_fit[0], x_fit[-1])  # the x range ends at the data, no margin

        ax.set_yscale("log")
        if i in bottom_row_indices:
            ax.set_xlabel(x_label, fontsize=FS_AUX)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5)

        # Lab name in bottom-right
        ax.text(
            0.96,
            0.04,
            r.spec.lab_name,
            transform=ax.transAxes,
            fontsize=FS_AUX,
            fontweight="bold",
            va="bottom",
            ha="right",
            color="#444444",
        )

        # Correlation annotation top-left
        ax.text(
            0.04,
            0.96,
            f"ρ = {r.rho:.2f}\nn = {r.n}",
            transform=ax.transAxes,
            fontsize=FS_AUX,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )


def make_figure():
    """Build Extended Data Fig 3: baseline protein vs clinical-lab scatters, Olink above SomaScan.

    Reads the baseline QC long tables and the screening labs through
    ``analysis.biomarker_correlation`` and draws the top 8 Olink and 13 SomaScan pairs by
    |rho|, each with its Spearman rho and the fitted log10(lab) ~ protein line.

    Returns:
        The figure.
    """
    init_figure_theme()

    # --- Load data ---
    olink_subjects = load_subjects("olink", "SampleID", BASELINE_VISITNUM)
    soma_subjects = load_subjects("soma", "SampleId", BASELINE_VISITNUM)

    olink_assay_ids = [p.assay_id for p in OLINK_PAIRS]
    soma_assay_ids = [p.assay_id for p in SOMA_PAIRS]

    olink_prot = load_proteomics(
        OLINK_NPX,
        "SampleID",
        "OlinkID",
        "NPX",
        olink_subjects,
        olink_assay_ids,
    )
    soma_prot = load_proteomics(
        SOMA_RFU, "SampleId", "SeqId", "RFU", soma_subjects, soma_assay_ids
    )

    all_paramcds = list(dict.fromkeys(p.paramcd for p in OLINK_PAIRS + SOMA_PAIRS))
    # Baseline proteomics is matched to the screening labs.
    lab_df = load_lab_values(ADLB, all_paramcds, LAB_SCREENING_VISITNUM)

    # --- Compute correlations ---
    olink_results = compute_correlations(
        OLINK_PAIRS,
        olink_prot,
        lab_df,
        "OlinkID",
        "NPX",
        log2_transform=False,
    )
    soma_results = compute_correlations(
        SOMA_PAIRS, soma_prot, lab_df, "SeqId", "RFU", log2_transform=True
    )

    olink_top = olink_results[:8]
    soma_top = soma_results[:13]

    # --- Create figure: 5 rows × 5 cols, panel height set by the row budget ---
    # Olink occupies rows 0-1 (2 rows × 5 cols = 10 cells, 8 used).
    # SomaScan occupies rows 2-4 (3 rows × 5 cols = 15 cells, 13 used).
    # Enlarged hspace between rows 1 and 2 keeps the SomaScan banner well clear
    # of the Olink bottom-row x-tick labels and "NPX" axis label.
    fig = panel_figure(NAT_W2, FIG_H_IN)
    axs = panel_grid(
        fig,
        np.arange(1, 26).reshape(5, 5),
        left=0.25,
        right=0.1,
        top=0.5,
        bottom=0.3,
        wspace=0.24,
        hspace=(ROW_GAP_IN, BLOCK_GAP_IN, ROW_GAP_IN, ROW_GAP_IN),
    )

    # Olink: rows 0-1, all 5 cols (10 cells; first 8 used, last 2 hidden)
    olink_axs = [axs[r * 5 + col] for r in range(2) for col in range(5)]
    # SomaScan: rows 2-4, all 5 cols (15 cells; first 13 used, last 2 hidden)
    soma_axs = [axs[r * 5 + col] for r in range(2, 5) for col in range(5)]

    # Olink bottom row = indices 5..7 (cells 8 and 9 are hidden, so 5,6,7 are
    # the visible bottom-most). SomaScan bottom-most visible per column:
    # cols 0-2 have 3 rows used (indices 10, 11, 12); cols 3-4 have 2 rows used
    # (indices 8, 9).
    olink_bottom = {5, 6, 7}
    soma_bottom = {8, 9, 10, 11, 12}

    _plot_panel(olink_axs, olink_top, "NPX", olink_bottom)
    _plot_panel(soma_axs, soma_top, "log₂(RFU)", soma_bottom)

    # --- Platform banners ---
    _add_platform_banner(fig, axs[0], axs[4], "Olink")
    _add_platform_banner(fig, axs[10], axs[14], "SomaScan")

    # --- Panel labels above platform banners ---
    bb_olink_l = axs[0].get_position()
    bb_soma_l = axs[10].get_position()
    label_rise = (
        PLATFORM_BANNER_GAP_IN + BANNER_HEIGHT_IN + PANEL_LABEL_GAP_IN
    ) / fig.get_figheight()
    olink_label_y = bb_olink_l.y1 + label_rise
    soma_label_y = bb_soma_l.y1 + label_rise
    fig.text(
        bb_olink_l.x0 - 0.01,
        olink_label_y,
        "a",
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    fig.text(
        bb_soma_l.x0 - 0.01,
        soma_label_y,
        "b",
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # --- Subplot banners ---
    for i, r in enumerate(olink_top):
        _add_subplot_banner(fig, olink_axs[i], r.spec.gene, r.spec.assay_id)

    for i, r in enumerate(soma_top):
        _add_subplot_banner(fig, soma_axs[i], r.spec.gene, r.spec.assay_id)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_path = Path(__file__).resolve().parent / "supp_fig_biomarker_correlation"
    save_figure(fig, str(out_path), formats=("pdf",))
