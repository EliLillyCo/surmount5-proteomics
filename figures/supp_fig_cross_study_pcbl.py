"""Cross-study replication: SURMOUNT-5 SEMA change from baseline (Wk72) vs STEP 1 / STEP 2.

Two-panel supplementary figure (nat2, 1 row × 2 cols):
  Panel a: SURMOUNT-5 SEMA Δ from baseline (Wk72) vs STEP 1 semaglutide t-statistic
  Panel b: SURMOUNT-5 SEMA Δ from baseline (Wk72) vs STEP 2 semaglutide t-statistic

Compares the SURMOUNT-5 SEMA week-72 within-arm change from baseline with the STEP 1 / STEP 2
semaglutide-vs-placebo statistic; each panel annotates the Spearman ρ.

x-axis: t = LSmean / StdErr of the SEMA arm's week-72 change from baseline in the
change-from-baseline MMRM, on the model (log2) scale (rawOutput ``*_chg_lsmeans.csv``,
``chg_lsmeans_csv`` + ``with_lsmean_stderr``); FDR from ``pcblRes``.
y-axis: STEP 1 / STEP 2 SomaScan t-statistic (Maretty et al. 2025).
See supp_fig_cross_study.py for the external data provenance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.ticker import NullLocator
from scipy.stats import norm, spearmanr, theilslopes

from figures._internal._common import (
    BANNER_HEIGHT_IN,
    FS_BANNER_BOLD,
    FS_BANNER_TAIL,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FDR_THRESHOLD,
    LEADER_GRAY,
    NAT_W2,
    deterministic_adjust_text,
    init_figure_theme,
    panel_figure,
    panel_grid,
    marker_labels,
    save_figure,
)
from analysis.mmrm_outputs import chg_lsmeans_csv, pcbl_csv, with_lsmean_stderr
from analysis.design import CONTROL_VALUE, COVAR_PHRASE, LATE_WEEK
from figures._internal._maretty_step import load_step_primary as load_step

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_LABEL_PER_SIDE = 12  # top genes by |z| on each side of x=0 (Δ from baseline)

SEMA_ARM_FULL = CONTROL_VALUE
SEMA_WEEK = LATE_WEEK


def _spearman_with_exact_p(x, y) -> tuple[float, str]:
    """Spearman ρ + two-sided p-value formatted with unicode superscript.

    The P value is calculated in log space with the Fisher z-transform (a large-n
    approximation).
    """
    r, _ = spearmanr(x, y)
    n = len(x)
    if not np.isfinite(r) or abs(r) >= 1:
        return r, "P<10⁻³⁰⁰"
    z_stat = np.sqrt(n - 3) * np.arctanh(r)
    log10_p = (np.log(2) + norm.logsf(abs(z_stat))) / np.log(10)
    exp = int(np.floor(log10_p))
    if exp < -300:
        return r, "P<10⁻³⁰⁰"
    mantissa = 10 ** (log10_p - exp)
    sup_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_str = str(exp).translate(sup_map)
    return r, f"p = {mantissa:.1f} × 10{exp_str}"


def sema_t_statistics(pcbl: pl.DataFrame, lsmeans: pl.DataFrame, week: int) -> pd.DataFrame:
    """SEMA within-arm change from baseline at ``week`` as the model's t-statistic.

    ``pcbl`` is the arm x visit ``finalRes/pcblRes`` table (``fdr``); ``lsmeans`` is the
    change-model ``rawOutput`` LSmeans (``chg_lsmeans_csv``). ``z`` = LSmean / StdErr on
    the model (log2) scale, the same construction as the STEP statistic (β / SE).
    ``dir_up`` flags proteins that increased from baseline (LSmean > 0). The
    column keeps the name ``z`` that the plotting code uses.
    """
    df = (
        with_lsmean_stderr(pcbl, lsmeans, trt=SEMA_ARM_FULL, time=week)
        .with_columns(
            (pl.col("LSmean") / pl.col("StdErr")).alias("z"),
            (pl.col("fdr") < FDR_THRESHOLD).alias("sig"),
            (pl.col("LSmean") > 0).alias("dir_up"),
        )
        .select(pl.col("marker").alias("SeqId"), "z", "sig", "dir_up", "fdr")
        .to_pandas()
    )
    return df.dropna()


def load_sema_pcbl() -> pd.DataFrame:
    """SURMOUNT-5 SEMA Wk72 within-arm change-from-baseline t-statistics and FDR."""
    return sema_t_statistics(
        pl.read_csv(pcbl_csv("soma", SEMA_ARM_FULL, SEMA_WEEK)),
        pl.read_csv(chg_lsmeans_csv("soma")),
        SEMA_WEEK,
    )


def build_panel_data(step: pd.DataFrame, gene_lookup: dict[str, str]) -> pd.DataFrame:
    """SURMOUNT-5 SEMA week-72 change-from-baseline t-statistics joined by SeqId to one STEP trial's.

    Args:
        step: The STEP trial's ``SeqId`` / ``z_step`` frame (``load_step``).
        gene_lookup: SeqId to gene symbol; a SeqId with no gene keeps its SeqId as ``symbol``.

    Returns:
        One row per shared SeqId with ``z``, ``sig``, ``dir_up``, ``fdr``, ``z_step`` and
        ``symbol``, rows with a missing statistic dropped, sorted by SeqId.
    """
    pcbl = load_sema_pcbl()
    merged = (
        pcbl.merge(step, on="SeqId", how="inner")
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
    trial_label: str,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
):
    """Draw one SURMOUNT-5 SEMA vs STEP t-statistic scatter into ``ax``.

    Points are colored by SURMOUNT-5 FDR and direction; the panel carries a Theil-Sen line,
    gene labels and Spearman rho with a P value computed in log space.

    Args:
        ax: Target axes.
        df: Panel data from :func:`build_panel_data`.
        trial_label: STEP trial name for the y label.
        show_ylabel: Draw the y label and tick labels.
        show_xlabel: Draw the x label and tick labels.
    """
    # x = SMT-5 SEMA Δ-from-baseline t-statistic; y = STEP trial t-statistic (sema vs placebo)
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

    # Theil-Sen regression line.
    slope, intercept, _, _ = theilslopes(y, x)
    ax.axline((0, intercept), slope=slope, color="#333333", lw=0.6, alpha=0.9, zorder=3)

    # Gene labels: top N by |z| on each side of x=0.
    # SURMOUNT-5 x-axis is Δ from baseline, so "each side" = proteins that
    # went down (z<0) vs up (z>0) under SEMA. One label per gene, secondary
    # arrows fan out to other aptamers of the same gene.
    sig_df = df[df["sig"]].copy().assign(abs_z=lambda d: d["z"].abs())

    pos_pool = (
        sig_df[sig_df["z"] > 0]
        .sort_values(
            ["abs_z", "symbol", "SeqId", "z", "z_step"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .head(N_LABEL_PER_SIDE * 2)
    )
    neg_pool = (
        sig_df[sig_df["z"] < 0]
        .sort_values(
            ["abs_z", "symbol", "SeqId", "z", "z_step"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .head(N_LABEL_PER_SIDE * 2)
    )
    candidates = pd.concat([pos_pool, neg_pool], ignore_index=True).drop_duplicates(
        "SeqId"
    )

    gene_points: dict[str, list[tuple[float, float]]] = {}
    for _, row in candidates.iterrows():
        gene_points.setdefault(row["symbol"], []).append(
            (float(row["z"]), float(row["z_step"]))
        )

    gene_best: dict[str, tuple[float, float]] = {
        gene: max(pts, key=lambda p: (abs(p[0]), abs(p[1]), p[0], p[1]))
        for gene, pts in gene_points.items()
    }

    def _gene_pick(pool: pd.DataFrame, n: int) -> list[str]:
        return (
            pool.sort_values(
                ["abs_z", "symbol", "SeqId", "z", "z_step"],
                ascending=[False, True, True, True, True],
                kind="mergesort",
            )
            .drop_duplicates("symbol")
            .head(n)["symbol"]
            .tolist()
        )

    pos_genes = _gene_pick(pos_pool, N_LABEL_PER_SIDE) if len(pos_pool) else []
    neg_genes = _gene_pick(neg_pool, N_LABEL_PER_SIDE) if len(neg_pool) else []
    top_genes = sorted(
        dict.fromkeys(pos_genes + neg_genes),
        key=lambda gene: (gene_best[gene][0], gene_best[gene][1], gene),
    )

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

    # Spearman ρ + exact p — upper-left, two lines (top-right gets gene labels)
    r, p_str = _spearman_with_exact_p(x, y)
    ax.text(
        0.03,
        0.97,
        f"ρ = {r:.3f}\n{p_str}",
        transform=ax.transAxes,
        fontsize=5.5,
        ha="left",
        va="top",
        color="#444444",
        zorder=6,
        linespacing=1.3,
    )

    ax.set_xlabel(
        f"SURMOUNT-5 t-statistic (SEMA Δ from baseline, Wk{SEMA_WEEK})" if show_xlabel else ""
    )
    ax.set_ylabel(f"{trial_label} t-statistic (SEMA vs Placebo, Wk68)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
        ax.yaxis.set_minor_locator(NullLocator())
    if not show_xlabel:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.xaxis.set_minor_locator(NullLocator())


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


TRIAL_ORDER = [("step1", "STEP 1"), ("step2", "STEP 2")]
PANEL_LABELS = ["a", "b"]


def make_figure():
    """Build Supplementary Fig 7: SURMOUNT-5 SEMA week-72 change vs STEP 1 and STEP 2, side by side.

    Reads the SomaScan SEMA PCBL and change-model LSmean outputs and
    ``analysis/outputs/maretty_step_data.parquet``; both panels share the x range.

    Returns:
        The figure.
    """
    init_figure_theme()

    gene_lookup = marker_labels("soma")
    panel_data = {
        trial_key: build_panel_data(load_step(trial_key), gene_lookup)
        for trial_key, _ in TRIAL_ORDER
    }

    # Double column (183 mm), two square panels.
    fig = panel_figure(NAT_W2, 3.5407)
    axs = panel_grid(
        fig, [[1, 2]], left=0.3832, right=0.0451, top=0.0650, bottom=0.3132, wspace=0.4514
    )

    for col_idx, (trial_key, trial_label) in enumerate(TRIAL_ORDER):
        plot_scatter_into(
            axs[col_idx],
            panel_data[trial_key],
            trial_label=trial_label,
            show_ylabel=True,  # each panel has its own STEP y-axis
            show_xlabel=True,
        )

    # Shared x across both panels — same SURMOUNT-5 SEMA Wk72 data feeds both.
    xlims = [ax.get_xlim() for ax in axs]
    x0 = min(xl[0] for xl in xlims)
    x1 = max(xl[1] for xl in xlims)
    x_pad = (x1 - x0) * 0.05
    x0 -= x_pad
    x1 += x_pad
    for ax in axs:
        ax.set_xlim(x0, x1)

    # One banner + panel label per panel (each panel is its own STEP comparison).
    fig.canvas.draw()
    for col_idx, (_, trial_label) in enumerate(TRIAL_ORDER):
        _add_banner(
            fig,
            axs[col_idx],
            label="SomaScan",
            sublabel=f"· SURMOUNT-5 SEMA vs {trial_label} · adj. {COVAR_PHRASE}, baseline protein",
        )
        _add_panel_label(fig, axs[col_idx], PANEL_LABELS[col_idx])

    return fig


def _add_panel_label(fig, ax, label: str):
    """Place a bold panel label (a, b) above a panel's banner."""
    bb = ax.get_position()
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


def _add_banner(fig, ax, label: str, sublabel: str = ""):
    """Lightgray banner above a single panel (matches volcano style)."""
    bb = ax.get_position()
    banner_h = BANNER_HEIGHT_IN / fig.get_figheight()
    banner_y = bb.y1 + bb.height * 0.02
    banner = mpatches.FancyBboxPatch(
        (bb.x0, banner_y),
        bb.width,
        banner_h,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(banner)
    mid_x = bb.x0 + bb.width / 2
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
    save_figure(fig, str(out_dir / "supp_fig_cross_study_pcbl"), formats=("pdf",))
    print("Wrote supp_fig_cross_study_pcbl.pdf")
