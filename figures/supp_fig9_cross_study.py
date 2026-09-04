"""Cross-study proteome comparison: SURMOUNT-5 TZP−SEMA differential vs STEP 1 and STEP 2.

Two-row supplementary figure (nat2, 2 rows × 2 cols):
  Row a: acTrt (TZP−SEMA) Wk24, Wk72 z-score vs STEP 1 semaglutide z-score
  Row b: acTrt (TZP−SEMA) Wk24, Wk72 z-score vs STEP 2 semaglutide z-score

Near-zero Spearman ρ supports that the GIP-incremental differential is orthogonal
to GLP-1R pharmacology as characterized in STEP 1 (obesity) and STEP 2 (obesity + T2D).

External data — STEP 1 and STEP 2 SomaScan results:
  Source paper: Maretty et al., Nat Med 31, 1–11 (2025)
                https://www.nature.com/articles/s41591-024-03355-2
  Supplementary Table 2 (xlsx, sheets S2_tx_STEP1 and S3_tx_STEP2):
                https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-024-03355-2/MediaObjects/41591_2024_3355_MOESM3_ESM.xlsx

  Prepared by analysis/scripts/maretty_step_data.py and read from
  analysis/outputs/maretty_step_data.parquet.

  Columns used:
    - ANALYTEID: `seq.XXXX.XX` — converted to `XXXX-XX` SomaScan SeqId for joining
    - test_statistic: t-statistic from linear mixed model (sema 2.4 mg vs placebo)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import ultraplot as uplt
from matplotlib.patheffects import withStroke
from scipy.stats import spearmanr, theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _internal._common import (  # noqa: E402
    BANNER_HEIGHT_IN,
    FS_BANNER_BOLD,
    FS_BANNER_TAIL,
    COVAR,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FDR_THRESHOLD,
    RESULTS_DIR,
    deterministic_adjust_text,
    init_figure_theme,
    load_marker_to_gene,
    save_figure,
)
from _internal._maretty_step import load_step_primary as load_step  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_LABEL_MAIN = 10
N_LABEL_DISCORDANT = 5
N_LABEL_CONCORDANT = 6
QUADRANT_Y_MIN = 2.0

# ---------------------------------------------------------------------------
# Gene symbol lookup
# ---------------------------------------------------------------------------


def build_soma_gene_lookup() -> dict[str, str]:
    """SeqId → HGNC gene symbol via uniprot_map.parquet."""
    return load_marker_to_gene("soma")


def load_actrt(week: int) -> pd.DataFrame:
    """Across-treatment (TZP vs SEMA) results with Wald z-score and FDR.

    OlinkAnalyze fits on log2 scale and reports FC = 2^β with SE-FC on the
    fold-change scale via the delta method: SE(FC) ≈ FC · ln(2) · SE(β).
    Inverting gives SE(β) = SE-FC / (FC · ln 2), and the Wald statistic is

        z = β / SE(β) = log₂(FC) · FC · ln(2) / SE-FC

    This is the actual t-statistic from the model (with df → large, t ≈ z)
    and does not round-trip through the p-value, so it stays well-defined
    at extremes where p would saturate.
    """
    path = (
        RESULTS_DIR
        / "surmount5_soma"
        / COVAR
        / "finalRes"
        / "acTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_"
        f"TZP15mgorMTDVSSEMA2.4mgorMTD@{week}_py.csv"
    )
    df = pd.read_csv(path)
    fc = df["FC"].values
    se_fc = df["SE-FC"].values
    df["z"] = np.log2(fc) * fc * np.log(2) / se_fc
    df["sig"] = df["fdr"] < FDR_THRESHOLD
    df["dir_up"] = df["FC"] > 1  # True = TZP > SEMA
    return df.rename(columns={"marker": "SeqId"})[
        ["SeqId", "z", "sig", "dir_up", "fdr"]
    ].dropna()


def build_panel_data(
    week: int,
    step: pd.DataFrame,
    gene_lookup: dict[str, str],
) -> pd.DataFrame:
    actrt = load_actrt(week)
    merged = (
        actrt.merge(step, on="SeqId", how="inner")
        .dropna(subset=["z", "z_step"])
        .sort_values(["SeqId", "z", "z_step"], kind="mergesort")
        .reset_index(drop=True)
    )
    merged["symbol"] = merged["SeqId"].map(gene_lookup).fillna(merged["SeqId"])
    return merged


# ---------------------------------------------------------------------------
# Single scatter panel
# ---------------------------------------------------------------------------


def plot_scatter_into(
    ax,
    df: pd.DataFrame,
    week: int,
    trial_label: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_x_direction_cues: bool = True,
):
    # x = SMT-5 acTrt z-score (TZP − SEMA); y = STEP trial z-score (sema vs placebo)
    x = df["z"].values
    y = df["z_step"].values
    sig = df["sig"].values
    dir_up = df["dir_up"].values

    ns = ~sig
    sig_up = sig & dir_up
    sig_dn = sig & ~dir_up

    # Scatter layers
    ax.scatter(x[ns], y[ns], s=3, c=COLOR_NS, alpha=0.45, edgecolors="none", zorder=1)
    ax.scatter(
        x[sig_dn],
        y[sig_dn],
        s=9,
        c=COLOR_DOWN,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )
    ax.scatter(
        x[sig_up],
        y[sig_up],
        s=9,
        c=COLOR_UP,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )

    # Reference lines — solid
    ax.axhline(0, color="#AAAAAA", lw=0.5, zorder=0)
    ax.axvline(0, color="#AAAAAA", lw=0.5, zorder=0)

    # Theil-Sen regression line — robust median-pairwise-slope estimator.
    # Matches the sign of Spearman ρ (which we report); OLS would be yanked
    # by single outliers (e.g. PNLIPRP1 at STEP 1 z=23) and could disagree
    # in direction with the rank correlation we annotate.
    slope, intercept, _, _ = theilslopes(y, x)
    ax.axline((0, intercept), slope=slope, color="#333333", lw=0.6, alpha=0.9, zorder=3)

    # Gene labels: one label per gene, arrows to all aptamers — matches volcano pattern.
    # Five pools, ensuring all four quadrants get representation:
    #   - main: top N_LABEL_MAIN FDR<0.05 hits by |z| in our data
    #   - top-left discordant     (x<0, y≥Y_MIN): SEMA > TZP but sema-up in STEP 1
    #   - bottom-right discordant (x>0, y≤-Y_MIN): TZP > SEMA but sema-down in STEP 1
    #   - bottom-left concordant  (x<0, y≤-Y_MIN): SEMA > TZP and sema-down in STEP 1
    #   - top-right concordant    (x>0, y≥Y_MIN): TZP > SEMA and sema-up in STEP 1
    halo = [withStroke(linewidth=3, foreground="white")]
    sig_df = df[df["sig"]].copy().assign(abs_z=lambda d: d["z"].abs())

    main_pool = sig_df.sort_values(
        ["abs_z", "symbol", "SeqId", "z", "z_step"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).head(N_LABEL_MAIN * 2)

    def _quadrant(x_sign: int, y_sign: int, n: int) -> pd.DataFrame:
        if x_sign > 0:
            xmask = sig_df["z"] > 0
        else:
            xmask = sig_df["z"] < 0
        if y_sign > 0:
            ymask = sig_df["z_step"] >= QUADRANT_Y_MIN
        else:
            ymask = sig_df["z_step"] <= -QUADRANT_Y_MIN
        pool = sig_df[xmask & ymask].copy()
        if pool.empty:
            return pool
        pool["dscore"] = pool["abs_z"] + pool["z_step"].abs()
        return pool.sort_values(
            ["dscore", "symbol", "SeqId", "z", "z_step"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        ).head(n * 2)

    top_left_pool = _quadrant(-1, +1, N_LABEL_DISCORDANT)
    bot_right_pool = _quadrant(+1, -1, N_LABEL_DISCORDANT)
    bot_left_pool = _quadrant(-1, -1, N_LABEL_CONCORDANT)
    top_right_pool = _quadrant(+1, +1, N_LABEL_CONCORDANT)

    candidates = pd.concat(
        [main_pool, top_left_pool, bot_right_pool, bot_left_pool, top_right_pool],
        ignore_index=True,
    ).drop_duplicates("SeqId")
    gene_points: dict[str, list[tuple[float, float]]] = {}
    for _, row in candidates.iterrows():
        gene_points.setdefault(row["symbol"], []).append(
            (float(row["z"]), float(row["z_step"]))
        )

    # Best point per gene = aptamer with highest |z|
    gene_best: dict[str, tuple[float, float]] = {
        gene: max(pts, key=lambda p: (abs(p[0]), abs(p[1]), p[0], p[1]))
        for gene, pts in gene_points.items()
    }

    # Pick which genes to label: union of main + discordant gene picks
    def _gene_pick(pool: pd.DataFrame, score_col: str, n: int) -> list[str]:
        pool_with_gene = pool.assign(symbol=pool["symbol"])
        # one row per gene with that gene's max score in the pool
        return (
            pool_with_gene.sort_values(
                [score_col, "symbol", "SeqId", "z", "z_step"],
                ascending=[False, True, True, True, True],
                kind="mergesort",
            )
            .drop_duplicates("symbol")
            .head(n)["symbol"]
            .tolist()
        )

    main_genes = _gene_pick(main_pool, "abs_z", N_LABEL_MAIN)
    tl_genes = (
        _gene_pick(top_left_pool, "dscore", N_LABEL_DISCORDANT)
        if len(top_left_pool)
        else []
    )
    br_genes = (
        _gene_pick(bot_right_pool, "dscore", N_LABEL_DISCORDANT)
        if len(bot_right_pool)
        else []
    )
    bl_genes = (
        _gene_pick(bot_left_pool, "dscore", N_LABEL_CONCORDANT)
        if len(bot_left_pool)
        else []
    )
    tr_genes = (
        _gene_pick(top_right_pool, "dscore", N_LABEL_CONCORDANT)
        if len(top_right_pool)
        else []
    )

    top_genes = sorted(
        dict.fromkeys(main_genes + tl_genes + br_genes + bl_genes + tr_genes),
        key=lambda gene: (gene_best[gene][0], gene_best[gene][1], gene),
    )

    # Place one text per gene at its best-aptamer position
    texts = []
    for gene in top_genes:
        bx, by = gene_best[gene]
        texts.append(
            ax.text(
                bx,
                by,
                gene,
                fontsize=5,
                ha="center",
                va="bottom",
                path_effects=halo,
                zorder=5,
            )
        )

    if texts:
        deterministic_adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
            expand=(1.4, 1.6),
        )

    # Secondary arrows: from the (repelled) label to every other aptamer of the same gene
    for i, gene in enumerate(top_genes):
        pts = gene_points[gene]
        if len(pts) <= 1:
            continue
        bx, by = gene_best[gene]
        lx, ly = texts[i].get_position()
        for px, py in pts:
            if (px, py) == (bx, by):
                continue
            ax.annotate(
                "",
                xy=(px, py),
                xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
                zorder=4,
            )

    # Spearman ρ — upper-right, single row, no frame
    r, p = spearmanr(x, y)
    if p == 0 or not np.isfinite(p):
        p_str = "p < 1×10⁻³⁰⁰"
    else:
        mantissa, exp = f"{p:.1e}".split("e")
        sup_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        exp_str = str(int(exp)).translate(sup_map)
        p_str = f"p = {mantissa} × 10{exp_str}"
    ax.text(
        0.97,
        0.97,
        f"ρ = {r:.3f} · {p_str}",
        transform=ax.transAxes,
        fontsize=5.5,
        ha="right",
        va="top",
        color="#444444",
        zorder=6,
    )

    # Week pill — top center
    ax.text(
        0.5,
        0.97,
        f"Week {week}",
        transform=ax.transAxes,
        fontsize=6,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#F0F0F0",
            edgecolor="#CCCCCC",
            linewidth=0.4,
        ),
        zorder=6,
    )

    # Direction cues on x-axis (SURMOUNT-5 acTrt). y=0.02 matches volcano spacing.
    if show_x_direction_cues:
        ax.text(
            0.03,
            0.02,
            "← SEMA > TZP",
            transform=ax.transAxes,
            fontsize=5.5,
            color=COLOR_DOWN,
            ha="left",
            va="bottom",
            path_effects=halo,
        )
        ax.text(
            0.97,
            0.02,
            "TZP > SEMA →",
            transform=ax.transAxes,
            fontsize=5.5,
            color=COLOR_UP,
            ha="right",
            va="bottom",
            path_effects=halo,
        )
    ax.format(
        xlabel="SURMOUNT-5 z-score (TZP vs SEMA)" if show_xlabel else "",
        ylabel=f"{trial_label} z-score (SEMA vs Placebo)" if show_ylabel else "",
        yticklabels=None if show_ylabel else [],
        xticklabels=None if show_xlabel else [],
    )
    if not show_ylabel:
        ax.tick_params(left=False)
    if not show_xlabel:
        ax.tick_params(bottom=False)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure():
    init_figure_theme()

    gene_lookup = build_soma_gene_lookup()
    step1 = load_step("step1")
    step2 = load_step("step2")
    panel_data = {
        ("step1", 24): build_panel_data(24, step1, gene_lookup),
        ("step1", 72): build_panel_data(72, step1, gene_lookup),
        ("step2", 24): build_panel_data(24, step2, gene_lookup),
        ("step2", 72): build_panel_data(72, step2, gene_lookup),
    }

    fig, axs = uplt.subplots(
        ncols=2,
        nrows=2,
        journal="nat2",
        share=False,
        refaspect=1.0,
        wspace=("1.5em",),
        hspace=("9em",),  # room for the row 2 banner above the bottom row
    )
    # axs flat order: [top-left, top-right, bottom-left, bottom-right]
    plot_scatter_into(
        axs[0],
        panel_data[("step1", 24)],
        week=24,
        trial_label="STEP 1",
        show_ylabel=True,
        show_xlabel=False,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[1],
        panel_data[("step1", 72)],
        week=72,
        trial_label="STEP 1",
        show_ylabel=False,
        show_xlabel=False,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[2],
        panel_data[("step2", 24)],
        week=24,
        trial_label="STEP 2",
        show_ylabel=True,
        show_xlabel=True,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[3],
        panel_data[("step2", 72)],
        week=72,
        trial_label="STEP 2",
        show_ylabel=False,
        show_xlabel=True,
        show_x_direction_cues=True,
    )

    # Shared x across all four panels (same SURMOUNT-5 z-scores anchor every column)
    all_xlims = [ax.get_xlim() for ax in axs]
    x0 = min(xl[0] for xl in all_xlims)
    x1 = max(xl[1] for xl in all_xlims)
    x_pad = (x1 - x0) * 0.05
    x0 -= x_pad
    x1 += x_pad

    # Shared y within each row (one trial per row)
    for row_axes in ([axs[0], axs[1]], [axs[2], axs[3]]):
        y0 = min(ax.get_ylim()[0] for ax in row_axes)
        y1 = max(ax.get_ylim()[1] for ax in row_axes)
        y0 -= (y1 - y0) * 0.10
        for ax in row_axes:
            ax.format(xlim=(x0, x1), ylim=(y0, y1))

    axs.format(abc=False)

    # One banner per row + a/b panel labels above each banner
    fig.canvas.draw()
    _add_banner(
        fig,
        axs[0],
        axs[1],
        label="SomaScan",
        sublabel="· SURMOUNT-5 and STEP 1 comparison · adj. age, sex, baseline protein",
    )
    _add_banner(
        fig,
        axs[2],
        axs[3],
        label="SomaScan",
        sublabel="· SURMOUNT-5 and STEP 2 comparison · adj. age, sex, baseline protein",
    )
    _add_panel_label(fig, axs[0], "a")
    _add_panel_label(fig, axs[2], "b")

    return fig


def _add_panel_label(fig, ax_left, label: str):
    """Place a bold panel label (a, b) above a row's banner."""
    bb = ax_left.get_position()
    banner_h = BANNER_HEIGHT_IN / fig.get_figheight()
    label_y = bb.y1 + bb.height * 0.02 + banner_h + 0.006
    fig.text(
        bb.x0 - 0.025,
        label_y,
        label,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _add_banner(fig, ax_left, ax_right, label: str, sublabel: str = ""):
    """Full-width lightgray banner above a row of panels (matches volcano style).

    Combined label + sublabel is measured and centered as a single block; the
    junction-based centering used by the volcano leans toward the longer side
    when label and sublabel have very different widths.
    """
    bb_left = ax_left.get_position()
    bb_right = ax_right.get_position()
    banner_h = BANNER_HEIGHT_IN / fig.get_figheight()
    banner_y = bb_left.y1 + bb_left.height * 0.02
    banner = mpatches.FancyBboxPatch(
        (bb_left.x0, banner_y),
        bb_right.x1 - bb_left.x0,
        banner_h,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = (bb_left.x0 + bb_right.x1) / 2
    mid_y = banner_y + banner_h / 2

    t1 = fig.text(
        0,
        mid_y,
        label,
        fontsize=FS_BANNER_BOLD,
        fontweight="bold",
        ha="left",
        va="center",
    )
    t2 = (
        fig.text(
            0,
            mid_y,
            sublabel,
            fontsize=FS_BANNER_TAIL,
            color="#666666",
            ha="left",
            va="center",
        )
        if sublabel
        else None
    )

    # Measure rendered widths in figure coordinates and reposition for true centering
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    w1 = t1.get_window_extent().transformed(inv).width
    w2 = t2.get_window_extent().transformed(inv).width if t2 else 0
    gap = 0.004 if t2 else 0
    start_x = mid_x - (w1 + gap + w2) / 2
    t1.set_position((start_x, mid_y))
    if t2:
        t2.set_position((start_x + w1 + gap, mid_y))


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "supp_fig9_cross_study"), formats=("pdf",))
    print("Wrote supp_fig9_cross_study.pdf")
