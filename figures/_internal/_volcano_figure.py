"""Figure 2 — between-treatment volcano plots (Olink / SomaScan x Wk24 / Wk72)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke
from matplotlib.ticker import NullLocator

from analysis.design import COVAR_PHRASE, PLATFORMS, POST_BASELINE_WEEKS
from analysis.mmrm_outputs import across_treatment_csv

from ._common import (
    COLOR_DOWN,
    COLOR_UP,
    COLOR_NS,
    FDR_THRESHOLD,
    NAT_W2,
    LEADER_GRAY,
    marker_labels,
    deterministic_adjust_text,
    init_figure_theme,
    panel_figure,
    panel_grid,
)

# ===========================================================================
# VOLCANO SECTION
# ===========================================================================

N_LABEL = 20
DIRECTION_FONTSIZE = 6  # axes-internal "← SEMA > TZP / TZP > SEMA →" cue size

DATA_FILES = {
    (platform, week): across_treatment_csv(platform, week)
    for platform in PLATFORMS
    for week in POST_BASELINE_WEEKS
}


# ---------------------------------------------------------------------------
# detectCap
# ---------------------------------------------------------------------------


def detect_cap(vals: np.ndarray) -> tuple[float | None, float | None]:
    """Cap extreme outliers: only if top > 3x second-highest.

    Returns (cap_hi, cap_lo). None means no capping needed on that end.
    """
    if len(vals) < 3:
        return (None, None)
    sv = np.sort(vals)
    cap_hi = None
    cap_lo = None
    top, second = sv[-1], sv[-2]
    bot, second_bot = sv[0], sv[1]
    if second > 0 and top > 3 * second:
        cap_hi = second * 1.2
    if second_bot < 0 and bot < 3 * second_bot:
        cap_lo = second_bot * 1.2
    return (cap_hi, cap_lo)


# ---------------------------------------------------------------------------
# Volcano data loading
# ---------------------------------------------------------------------------


def load_actrt(path: Path, gene_lookup: dict[str, str]) -> pd.DataFrame:
    """Read one between-treatment CSV and add the volcano columns.

    Args:
        path: The ``finalRes`` between-treatment CSV (``FC``, ``fdr``, ``marker``, ``Assay``).
        gene_lookup: Marker to gene symbol; a marker with no gene falls back to ``Assay``.

    Returns:
        The table with ``log2FC``, ``neglog`` (-log10 FDR; infinite values capped at 1.05 x
        the largest finite one), ``significant`` (FDR < ``FDR_THRESHOLD``) and ``Symbol``.
    """
    df = pd.read_csv(path)
    df["log2FC"] = np.log2(df["FC"])
    raw_neglog = -np.log10(df["fdr"].clip(lower=1e-300))
    # Cap infinite values at 1.05 x the largest finite one
    finite_mask = np.isfinite(raw_neglog)
    max_fin = raw_neglog[finite_mask].max() if finite_mask.any() else 300
    cap = max_fin * 1.05
    df["neglog"] = np.where(np.isfinite(raw_neglog), np.minimum(raw_neglog, cap), cap)
    df["significant"] = df["fdr"] < FDR_THRESHOLD
    # Map marker -> protein label (analysis.protein_labels); fall back to Assay if it has none
    df["Symbol"] = df["marker"].map(gene_lookup).fillna(df["Assay"])
    return df


# ---------------------------------------------------------------------------
# Single volcano panel renderer
# ---------------------------------------------------------------------------


def plot_volcano_into(
    ax,
    df: pd.DataFrame,
    week: int,
    n_label: int = N_LABEL,
    show_ylabel: bool = True,
):
    """Render a volcano plot into an existing axes."""
    df = df.copy()
    df = df.sort_values(
        ["marker", "Symbol", "log2FC", "neglog"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Outlier capping (detectCap logic)
    x_cap = detect_cap(df["log2FC"].to_numpy())
    y_cap = detect_cap(df["neglog"].to_numpy())

    df["origL2"] = df["log2FC"]
    df["origNL"] = df["neglog"]
    df["capped"] = False

    if y_cap[0] is not None:
        mask = df["neglog"] > y_cap[0]
        df.loc[mask, "capped"] = True
    if x_cap[0] is not None:
        mask = df["log2FC"] > x_cap[0]
        df.loc[mask, "capped"] = True
    if x_cap[1] is not None:
        mask = df["log2FC"] < x_cap[1]
        df.loc[mask, "capped"] = True

    # Classify points
    not_capped = ~df["capped"]
    sig_up = df["significant"] & (df["FC"] > 1) & not_capped
    sig_dn = df["significant"] & (df["FC"] <= 1) & not_capped
    ns = ~df["significant"] & not_capped

    # Count analytes/assays per category (for legend)
    n_up = df.loc[df["significant"] & (df["FC"] > 1), "marker"].nunique()
    n_dn = df.loc[df["significant"] & (df["FC"] <= 1), "marker"].nunique()
    n_ns = df.loc[~df["significant"], "marker"].nunique()

    ax.scatter(
        df.loc[ns, "log2FC"].to_numpy(),
        df.loc[ns, "neglog"].to_numpy(),
        s=5,
        c=COLOR_NS,
        alpha=0.6,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        df.loc[sig_up, "log2FC"].to_numpy(),
        df.loc[sig_up, "neglog"].to_numpy(),
        s=9,
        c=COLOR_UP,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
        label=f"Up ({n_up})",
    )
    ax.scatter(
        df.loc[sig_dn, "log2FC"].to_numpy(),
        df.loc[sig_dn, "neglog"].to_numpy(),
        s=9,
        c=COLOR_DOWN,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
        label=f"Down ({n_dn})",
    )
    ax.scatter(
        [], [], s=5, c=COLOR_NS, alpha=0.6, edgecolors="none", label=f"ns ({n_ns})"
    )

    ax.axhline(
        -np.log10(FDR_THRESHOLD),
        color="#888888",
        lw=0.6,
        zorder=0,
    )
    ax.axvline(0, color="#AAAAAA", lw=0.5, ls="-", zorder=0)

    # Axis range: computed from non-capped points only, symmetric x, y starts at 0
    non_capped_x = df.loc[~df["capped"], "log2FC"].to_numpy()
    non_capped_y = df.loc[~df["capped"], "neglog"].to_numpy()
    x_abs_max = np.abs(non_capped_x).max()
    x_pad = x_abs_max * 0.12
    y_max = non_capped_y.max()
    # A top margin of 12% of the largest -log10(FDR) leaves room for the count legend
    # in whichever top corner the points leave empty (chosen below).
    y_pad = y_max * 0.12
    xlim = (-(x_abs_max + x_pad), x_abs_max + x_pad)
    ylim = (0, y_max + y_pad)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Reposition capped points to sit near the axis boundary, then draw as
    # directional triangles pointing toward the true value
    capped_mask = df["capped"]
    if capped_mask.any():
        for idx in df.index[capped_mask]:
            x_trunc = False
            y_trunc = False
            if df.loc[idx, "log2FC"] < xlim[0]:
                df.loc[idx, "log2FC"] = xlim[0] + (xlim[1] - xlim[0]) * 0.03
                x_trunc = True
            elif df.loc[idx, "log2FC"] > xlim[1]:
                df.loc[idx, "log2FC"] = xlim[1] - (xlim[1] - xlim[0]) * 0.03
                x_trunc = True
            if df.loc[idx, "neglog"] > ylim[1]:
                df.loc[idx, "neglog"] = ylim[1] - (ylim[1] - ylim[0]) * 0.03
                y_trunc = True
            if x_trunc and y_trunc:
                df.loc[idx, "_marker"] = "D"
            elif x_trunc and df.loc[idx, "origL2"] < 0:
                df.loc[idx, "_marker"] = "<"
            elif x_trunc:
                df.loc[idx, "_marker"] = ">"
            elif y_trunc:
                df.loc[idx, "_marker"] = "^"
            else:
                df.loc[idx, "_marker"] = "D"

        capped_up = capped_mask & df["significant"] & (df["origL2"] > 0)
        capped_dn = capped_mask & df["significant"] & (df["origL2"] <= 0)
        capped_ns_mask = capped_mask & ~df["significant"]
        for mask, color in [
            (capped_up, COLOR_UP),
            (capped_dn, COLOR_DOWN),
            (capped_ns_mask, COLOR_NS),
        ]:
            if mask.any():
                for m_shape in ["<", ">", "^", "D"]:
                    sub = mask & (df["_marker"] == m_shape)
                    if sub.any():
                        ax.scatter(
                            df.loc[sub, "log2FC"].to_numpy(),
                            df.loc[sub, "neglog"].to_numpy(),
                            s=18,
                            marker=m_shape,
                            c=color,
                            edgecolors="black",
                            linewidths=0.5,
                            alpha=0.9,
                            zorder=3,
                        )

    # ---- Count legend (created BEFORE gene labels so adjust_text avoids it) ----
    # Place the legend away from capped markers: it takes the top corner without a
    # capped point. If both or neither corner has one, use the emptier corner
    # (ties → left). `leg` is passed to adjust_text so gene labels are repelled from
    # its bounds.
    _cap = df[df["capped"]]
    _cap_left = bool((_cap["log2FC"] < 0).any())
    _cap_right = bool((_cap["log2FC"] > 0).any())
    if _cap_left and not _cap_right:
        _legend_loc = "upper right"
    elif _cap_right and not _cap_left:
        _legend_loc = "upper left"
    else:
        _xf = (df["log2FC"] - xlim[0]) / (xlim[1] - xlim[0])
        _yf = (df["neglog"] - ylim[0]) / (ylim[1] - ylim[0])
        _top = _yf >= 0.74
        _n_tl = int((_top & (_xf <= 0.34)).sum())
        _n_tr = int((_top & (_xf >= 0.66)).sum())
        _legend_loc = "upper left" if _n_tl <= _n_tr else "upper right"
    leg = ax.legend(
        loc=_legend_loc,
        ncols=1,
        fontsize=6,
        handletextpad=0.3,
        handlelength=0.8,
        markerscale=1.4,
        borderaxespad=0.4,
        frameon=False,
    )
    leg.set_zorder(5)
    # Transparent legend (no fill) so a point in the legend corner and its leader
    # line stay visible through the legend. White halos keep the legend text legible
    # over any points beneath it.
    for _ltxt in leg.get_texts():
        _ltxt.set_path_effects([withStroke(linewidth=2.5, foreground="white")])
    ax.figure.canvas.draw()  # realize the legend bbox so adjust_text can avoid it

    # --- Gene-level label deduplication ---
    sig_up_all = (
        df[df["significant"] & (df["FC"] > 1)]
        .sort_values(
            ["fdr", "Symbol", "marker", "neglog", "log2FC"],
            ascending=[True, True, True, False, False],
            kind="mergesort",
        )
        .head(n_label)
    )
    sig_dn_all = (
        df[df["significant"] & (df["FC"] <= 1)]
        .sort_values(
            ["fdr", "Symbol", "marker", "neglog", "log2FC"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .head(n_label)
    )
    label_candidates = pd.concat([sig_up_all, sig_dn_all], ignore_index=True)

    # One label per assay (the protein label rule names a multi-gene assay once)
    label_candidates = label_candidates.sort_values(
        ["Symbol", "marker", "neglog", "log2FC"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    # Collect all points per gene, pick best (highest neglog) for label position
    gene_points: dict[str, list[tuple[float, float]]] = {}
    for _, row in label_candidates.iterrows():
        gene = row["Symbol"]
        pt = (row["log2FC"], row["neglog"])
        if gene not in gene_points:
            gene_points[gene] = []
        gene_points[gene].append(pt)

    gene_best: dict[str, tuple[float, float]] = {}
    for gene, pts in gene_points.items():
        gene_best[gene] = max(pts, key=lambda p: (p[1], abs(p[0]), p[0]))

    # Identify dual-truncated genes (diamond marker = truncated on both axes)
    dual_genes: dict[str, str] = {}
    if "_marker" in df.columns:
        dual_rows = df[df["capped"] & (df["_marker"] == "D")]
        for _, row in dual_rows.iterrows():
            dual_genes[row["Symbol"]] = f"({row['origL2']:.1f}, {row['origNL']:.0f})"

    # Place one label per gene at the best point (skip dual-truncated)
    halo = [withStroke(linewidth=3.5, foreground="white")]
    texts = []
    label_positions = []
    for gene, (bx, by) in sorted(
        gene_best.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    ):
        if gene in dual_genes:
            continue
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
        label_positions.append((gene, bx, by))

    if texts:
        deterministic_adjust_text(
            texts,
            ax=ax,
            objects=[leg],  # repel gene labels away from the count legend
            arrowprops=dict(arrowstyle="-", color=LEADER_GRAY, lw=0.7, alpha=0.7),
            expand=(1.4, 1.6),
        )
        # Deterministic guard for the legend corner. Two cases leave a gene
        # visually disconnected from its marker:
        #   (1) the LABEL still overlaps the legend after repulsion, or
        #   (2) the gene's ANCHOR POINT sits under the legend, so adjust_text's
        #       short arrow and the point itself are masked by the legend.
        # In either case drop the label to a clear spot just below the legend,
        # draw a thin leader up to the true point, and re-draw that point on top
        # of the (transparent) legend so it stays visible.
        ax.figure.canvas.draw()
        leg_bb = leg.get_window_extent()
        inv = ax.transData.inverted()
        for t, (_gene_i, ax_pt, ay_pt) in zip(texts, label_positions):
            tb = t.get_window_extent()
            pt_disp = ax.transData.transform((ax_pt, ay_pt))
            anchor_under_leg = bool(leg_bb.contains(pt_disp[0], pt_disp[1]))
            if not (tb.overlaps(leg_bb) or anchor_under_leg):
                continue
            x_d, _ = t.get_position()
            x_disp, _ = ax.transData.transform((x_d, 0))
            _, y_new = inv.transform((x_disp, leg_bb.y0 - 4))  # just below legend
            t.set_position((x_d, y_new))
            t.set_va("top")
            t.set_zorder(7)
            # Thin leader from the label up to the true marker.
            ax.annotate(
                "",
                xy=(ax_pt, ay_pt),
                xytext=(x_d, y_new),
                arrowprops=dict(arrowstyle="-", color=LEADER_GRAY, lw=0.7, alpha=0.7),
                zorder=4,
            )
            # Re-draw the marker above the transparent legend so it reads.
            _pc = COLOR_UP if ax_pt > 0 else COLOR_DOWN
            ax.scatter(
                [ax_pt],
                [ay_pt],
                s=9,
                c=_pc,
                edgecolors="white",
                linewidths=0.2,
                zorder=6,
            )

    # Label the points capped on both axes by hand, BELOW and to the right of the marker.
    for gene, true_label in dual_genes.items():
        if gene not in gene_best:
            continue
        bx, by = gene_best[gene]
        ax.annotate(
            gene,
            xy=(bx, by),
            xytext=(4, -1),
            textcoords="offset points",
            fontsize=5,
            ha="left",
            va="top",
            path_effects=halo,
            zorder=5,
        )
        ax.annotate(
            true_label,
            xy=(bx, by),
            xytext=(4, -7),
            textcoords="offset points",
            fontsize=5,
            color="#888888",
            ha="left",
            va="top",
            path_effects=halo,
            zorder=5,
        )

    # Draw secondary arrows from label to other points of the same gene
    for i, (gene, bx, by) in enumerate(label_positions):
        pts = gene_points[gene]
        if len(pts) <= 1:
            continue
        txt = texts[i]
        lx, ly = txt.get_position()
        for px, py in pts:
            if px == bx and py == by:
                continue
            ax.annotate(
                "",
                xy=(px, py),
                xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-", color=LEADER_GRAY, lw=0.7, alpha=0.7),
                zorder=4,
            )

    ax.set_xlabel("log₂(Fold Change)")
    ax.set_ylabel("−log₁₀(FDR)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(left=False, labelleft=False)
        ax.yaxis.set_minor_locator(NullLocator())

    # Week label as pill/island at top center of axes
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

    # Direction annotations at bottom of plot area
    ax.text(
        0.03,
        0.02,
        "← SEMA > TZP",
        transform=ax.transAxes,
        fontsize=DIRECTION_FONTSIZE,
        color=COLOR_DOWN,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.97,
        0.02,
        "TZP > SEMA →",
        transform=ax.transAxes,
        fontsize=DIRECTION_FONTSIZE,
        color=COLOR_UP,
        ha="right",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Volcano banner helper
# ---------------------------------------------------------------------------


def _add_banner(fig, ax_left, ax_right, label: str):
    """Add a full-width lightgray banner above a row of volcano panels."""
    bb_left = ax_left.get_position()
    bb_right = ax_right.get_position()
    banner_height = 0.022
    banner_y = bb_left.y1 + 0.005
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
        mid_x - 0.002,
        mid_y,
        label,
        fontsize=6.5,
        fontweight="bold",
        ha="right",
        va="center",
    )
    fig.text(
        mid_x + 0.002,
        mid_y,
        f"· TZP vs SEMA · Adj. {COVAR_PHRASE}, baseline protein",
        fontsize=5.5,
        color="#666666",
        ha="left",
        va="center",
    )


def make_volcano_figure():
    """Figure 2 — between-treatment volcano plots (Olink/SomaScan x Wk24/Wk72).

    2x2 grid: Olink row (a) over SomaScan row (b); Wk24/Wk72 as columns.
    Pathway enrichment is Figure 3 (make_pathway_figure).
    """
    init_figure_theme()

    array = [
        [1, 2],
        [3, 4],
    ]
    fig = panel_figure(NAT_W2, 7.3)
    axs = panel_grid(
        fig, array, left=0.55, right=0.25, top=0.45, bottom=0.55, wspace=0.45, hspace=0.8
    )

    gene_lookup = {**marker_labels("olink"), **marker_labels("soma")}
    vol_data = {key: load_actrt(path, gene_lookup) for key, path in DATA_FILES.items()}
    early, late = POST_BASELINE_WEEKS
    plot_volcano_into(axs[0], vol_data[("olink", early)], week=early, show_ylabel=True)
    plot_volcano_into(axs[1], vol_data[("olink", late)], week=late, show_ylabel=False)
    plot_volcano_into(axs[2], vol_data[("soma", early)], week=early, show_ylabel=True)
    plot_volcano_into(axs[3], vol_data[("soma", late)], week=late, show_ylabel=False)

    for left, right in [(axs[0], axs[1]), (axs[2], axs[3])]:
        y0 = min(left.get_ylim()[0], right.get_ylim()[0])
        y1 = max(left.get_ylim()[1], right.get_ylim()[1])
        left.set_ylim(y0, y1)
        right.set_ylim(y0, y1)

    fig.canvas.draw()
    _add_banner(fig, axs[0], axs[1], "Olink")
    _add_banner(fig, axs[2], axs[3], "SomaScan")

    PANEL_FONTSIZE = 8
    axs[0].text(
        -0.08,
        1.02,
        "a",
        transform=axs[0].transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
    )
    axs[2].text(
        -0.08,
        1.02,
        "b",
        transform=axs[2].transAxes,
        fontsize=PANEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
    )
    return fig
