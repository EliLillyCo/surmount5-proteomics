"""Cross-study proteome comparison: SURMOUNT-5 TZP−SEMA differential vs STEP 1 and STEP 2.

Two-row supplementary figure (nat2, 2 rows × 2 cols):
  Row a: acTrt (TZP−SEMA) Wk24, Wk72 t-statistic vs STEP 1 semaglutide t-statistic
  Row b: acTrt (TZP−SEMA) Wk24, Wk72 t-statistic vs STEP 2 semaglutide t-statistic

x-axis: the TZP − SEMA contrast estimate over its own MMRM StdErr (log2 scale), from the
change-model rawOutput contrasts (``chg_diff_csv`` + ``with_contrast_stderr``), i.e. the
model's t-statistic.

Each panel annotates the Spearman ρ of the two statistics across matched assays.

External data — STEP 1 and STEP 2 SomaScan results:
  Source paper: Maretty et al., Nat Med 31, 1–11 (2025)
                https://www.nature.com/articles/s41591-024-03355-2
  Supplementary Table 2 (xlsx, sheets S2_tx_STEP1 and S3_tx_STEP2):
                https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-024-03355-2/MediaObjects/41591_2024_3355_MOESM3_ESM.xlsx

  Prepared by analysis/pipelines/maretty_step_data.py and read from
  analysis/outputs/maretty_step_data.parquet.

  Columns used:
    - ANALYTEID: `seq.XXXX.XX` — converted to `XXXX-XX` SomaScan SeqId for joining
    - test_statistic: t-statistic from linear mixed model (sema 2.4 mg vs placebo)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.ticker import NullLocator
from scipy.stats import spearmanr, theilslopes

from figures._internal._common import (
    BANNER_HEIGHT_IN,
    FS_BANNER_BOLD,
    FS_BANNER_TAIL,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FDR_THRESHOLD,
    LEADER_GRAY,
    deterministic_adjust_text,
    init_figure_theme,
    panel_figure,
    panel_grid,
    marker_labels,
    save_figure,
)
from analysis.mmrm_outputs import (
    across_treatment_csv,
    chg_diff_csv,
    with_contrast_stderr,
)
from analysis.design import CONTROL_VALUE, COVAR_PHRASE, POST_BASELINE_WEEKS, TREAT_VALUE
from figures._internal._maretty_step import load_step_primary as load_step

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_W_IN = 7.05
FIG_H_IN = 6.55
N_LABEL_MAIN = 10
N_LABEL_DISCORDANT = 5
N_LABEL_CONCORDANT = 6
QUADRANT_Y_MIN = 2.0

# ---------------------------------------------------------------------------
# Gene symbol lookup
# ---------------------------------------------------------------------------


def build_soma_gene_lookup() -> dict[str, str]:
    """SeqId → protein label (``analysis.protein_labels``)."""
    return marker_labels("soma")


def actrt_t_statistics(final: pl.DataFrame, raw: pl.DataFrame, week: int) -> pd.DataFrame:
    """TZP − SEMA t-statistic at ``week``: the contrast estimate over its own MMRM SE.

    ``final`` is the ``finalRes/acTrt`` table (``FC = 2^β``, ``fdr``); ``raw`` is the
    change-model ``rawOutput`` contrasts (``chg_diff_csv``), which carry the contrast's
    ``StdErr`` on the model (log2) scale. ``z`` = log₂(FC) / StdErr is the model's
    t-statistic (``tValue``), the same construction as the STEP statistic on the y-axis.
    The column keeps the name ``z`` that the plotting code uses.
    """
    df = (
        with_contrast_stderr(
            final, raw, trt=TREAT_VALUE, _trt=CONTROL_VALUE, time=week, _time=week
        )
        .with_columns(
            (pl.col("FC").log(base=2) / pl.col("StdErr")).alias("z"),
            (pl.col("fdr") < FDR_THRESHOLD).alias("sig"),
            (pl.col("FC") > 1).alias("dir_up"),  # True = TZP > SEMA
        )
        .select(pl.col("marker").alias("SeqId"), "z", "sig", "dir_up", "fdr")
        .to_pandas()
    )
    return df.dropna()


def load_actrt(week: int) -> pd.DataFrame:
    """Between-treatment (TZP vs SEMA) t-statistics and FDR on SomaScan at ``week``."""
    return actrt_t_statistics(
        pl.read_csv(across_treatment_csv("soma", week)),
        pl.read_csv(chg_diff_csv("soma")),
        week,
    )


def build_panel_data(
    week: int,
    step: pd.DataFrame,
    gene_lookup: dict[str, str],
) -> pd.DataFrame:
    """SURMOUNT-5 TZP-SEMA t-statistics at ``week`` joined by SeqId to one STEP trial's.

    Args:
        week: SURMOUNT-5 post-baseline week (``POST_BASELINE_WEEKS``).
        step: The STEP trial's ``SeqId`` / ``z_step`` frame (``load_step``).
        gene_lookup: SeqId to gene symbol; a SeqId with no gene keeps its SeqId as ``symbol``.

    Returns:
        One row per shared SeqId with ``z``, ``sig``, ``dir_up``, ``fdr``, ``z_step`` and
        ``symbol``, rows with a missing statistic dropped, sorted by SeqId.
    """
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
    """Draw one SURMOUNT-5 vs STEP t-statistic scatter into ``ax``.

    Points are colored by SURMOUNT-5 FDR and direction; the panel carries a Theil-Sen line,
    gene labels for the top hits and each quadrant, Spearman rho with its P value and a week
    pill.

    Args:
        ax: Target axes.
        df: Panel data from :func:`build_panel_data`.
        week: SURMOUNT-5 week, shown in the pill.
        trial_label: STEP trial name for the y label.
        show_ylabel: Draw the y label and tick labels.
        show_xlabel: Draw the x label and tick labels.
        show_x_direction_cues: Draw the "SEMA > TZP" / "TZP > SEMA" cues.
    """
    # x = SMT-5 acTrt t-statistic (TZP − SEMA); y = STEP trial t-statistic (sema vs placebo)
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

    # Theil-Sen line (median of pairwise slopes).
    slope, intercept, _, _ = theilslopes(y, x)
    ax.axline((0, intercept), slope=slope, color="#333333", lw=0.6, alpha=0.9, zorder=3)

    # Gene labels: one label per gene, arrows to all aptamers — matches volcano pattern.
    # Five pools, ensuring all four quadrants get representation:
    #   - main: top N_LABEL_MAIN FDR<0.05 points by |z|
    #   - top-left discordant     (x<0, y≥Y_MIN): SEMA > TZP but sema-up in STEP 1
    #   - bottom-right discordant (x>0, y≤-Y_MIN): TZP > SEMA but sema-down in STEP 1
    #   - bottom-left concordant  (x<0, y≤-Y_MIN): SEMA > TZP and sema-down in STEP 1
    #   - top-right concordant    (x>0, y≥Y_MIN): TZP > SEMA and sema-up in STEP 1
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
                zorder=5,
            )
        )

    if texts:
        deterministic_adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color=LEADER_GRAY, lw=0.6, alpha=0.7),
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
                arrowprops=dict(arrowstyle="-", color=LEADER_GRAY, lw=0.6, alpha=0.7),
                zorder=4,
            )

    # Spearman ρ — upper-right, single row, no frame
    r, p = spearmanr(x, y)
    if p < 1e-300 or not np.isfinite(p):
        p_str = "P<10⁻³⁰⁰"
    else:
        mantissa, exp = f"{p:.1e}".split("e")
        sup_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        exp_str = str(int(exp)).translate(sup_map)
        p_str = f"p = {mantissa} × 10{exp_str}"
    ax.text(
        0.97,
        0.97,
        f"ρ = {r:.3f} · {p_str}".replace("-", "−"),
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
        )
    ax.set_xlabel("SURMOUNT-5 t-statistic (TZP vs SEMA)" if show_xlabel else "")
    ax.set_ylabel(f"{trial_label} t-statistic (SEMA vs Placebo)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
        ax.yaxis.set_minor_locator(NullLocator())
    if not show_xlabel:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.xaxis.set_minor_locator(NullLocator())


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure():
    """Build Extended Data Fig 5: SURMOUNT-5 TZP-SEMA vs STEP 1 (row a) and STEP 2 (row b), weeks 24 and 72.

    Reads the SomaScan between-treatment and change-model contrast outputs and
    ``analysis/outputs/maretty_step_data.parquet``.

    Returns:
        The figure.
    """
    init_figure_theme()

    gene_lookup = build_soma_gene_lookup()
    step1 = load_step("step1")
    step2 = load_step("step2")
    early, late = POST_BASELINE_WEEKS
    panel_data = {
        (trial, week): build_panel_data(week, step, gene_lookup)
        for trial, step in (("step1", step1), ("step2", step2))
        for week in (early, late)
    }

    # 179 x 166 mm page, inside the 180 x 170 mm Extended Data box; the panels are wider
    # than tall rather than square to fit the height.
    fig = panel_figure(FIG_W_IN, FIG_H_IN)
    axs = panel_grid(
        fig,
        [[1, 2], [3, 4]],
        left=0.3994444,
        right=0.0451389,
        top=0.42,  # row 1 banner and panel label
        bottom=0.3131944,
        wspace=0.1354167,  # 1.5 em of the 6.5 pt body text
        hspace=0.62,  # room for the row 2 banner above the bottom row
    )
    # axs flat order: [top-left, top-right, bottom-left, bottom-right]
    plot_scatter_into(
        axs[0],
        panel_data[("step1", early)],
        week=early,
        trial_label="STEP 1",
        show_ylabel=True,
        show_xlabel=False,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[1],
        panel_data[("step1", late)],
        week=late,
        trial_label="STEP 1",
        show_ylabel=False,
        show_xlabel=False,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[2],
        panel_data[("step2", early)],
        week=early,
        trial_label="STEP 2",
        show_ylabel=True,
        show_xlabel=True,
        show_x_direction_cues=True,
    )
    plot_scatter_into(
        axs[3],
        panel_data[("step2", late)],
        week=late,
        trial_label="STEP 2",
        show_ylabel=False,
        show_xlabel=True,
        show_x_direction_cues=True,
    )

    # Shared x across all four panels (same SURMOUNT-5 t-statistics anchor every column)
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
        y1 += (y1 - y0) * 0.04  # headroom for the top label
        for ax in row_axes:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

    # One banner per row + a/b panel labels above each banner
    fig.canvas.draw()
    _add_banner(
        fig,
        axs[0],
        axs[1],
        label="SomaScan",
        sublabel=f"· SURMOUNT-5 vs STEP 1 · SURMOUNT-5 adj. {COVAR_PHRASE}, baseline protein · STEP 1 adj. baseline protein",
    )
    _add_banner(
        fig,
        axs[2],
        axs[3],
        label="SomaScan",
        sublabel=f"· SURMOUNT-5 vs STEP 2 · SURMOUNT-5 adj. {COVAR_PHRASE}, baseline protein · STEP 2 adj. baseline protein",
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
    save_figure(fig, str(out_dir / "supp_fig_cross_study"), formats=("pdf",))
    print("Wrote supp_fig_cross_study.pdf")
