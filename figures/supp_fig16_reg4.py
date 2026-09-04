"""Supplementary figure for REG4 — the non-GIP counterpart of ``fig5_ppy.py``.

Same three analysis panels as the main PPY figure *minus* the GIP-receiving
cell-types panel (that mechanism panel is specific to the PPY story):

  Panel a (top-left):  SURMOUNT-5 %CFB trajectories — Olink + SomaScan,
                       TZP vs SEMA across Wk0/24/72.
  Panel b (top-right): STEP 1 + STEP 2 SomaScan forest — SEMA vs PBO.
  Panel c (bottom):    GIPR rs1800437 (E354Q) × genotype boxplots, 1×4.

All drawing primitives, the bottom-row genotype panel, the shared banner, and
the vertical layout are reused from ``fig5_ppy``; only the top row (two panels
instead of three) is laid out here. The top-row block is *centered* in the page
width rather than stretched, and the legends are spread out so the full FDR key
(REG4 has a ``*`` star) does not collide with the arm legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _internal._common import (  # noqa: E402
    ARM_COLORS,
    ARM_LABELS,
    COLOR_NS,
    FS_AUX,
    FS_PANEL,
    c,
    init_figure_theme,
    save_figure,
)
import _internal._ppy_figure as F  # noqa: E402  (reuse layout constants + drawing primitives)

GENE = "REG4"
FDR_LEGEND_TEXT = "FDR: * <0.05   ** <0.01   *** <0.001"

FIG_W, FIG_H = F.FIG_W, F.FIG_H

# --- Top-row horizontal layout (trajectory pair + forest, no GIP) -----------
# Plots keep their main-figure size (trajectory aspect-locked); the inter-
# trajectory gap is nudged up and the forest is slightly wider than the main
# figure's. The whole block is then centered in the page width.
_TRAJ_W_IN = F._TRAJ_W_IN  # aspect-locked trajectory panel width
_WSPACE_IN = 0.30  # between the two trajectory panels
_FOREST_W_IN = 1.20  # slightly wider than the main figure (1.09)
_TRAJ_FOREST_GAP_IN = 0.55  # holds the forest STEP y-labels
_YLABEL_GUTTER_IN = 0.55  # holds the trajectory y-axis label + ticks

_block_in = _TRAJ_W_IN + _WSPACE_IN + _TRAJ_W_IN + _TRAJ_FOREST_GAP_IN + _FOREST_W_IN
_visual_w = _YLABEL_GUTTER_IN + _block_in
_left_visual = (FIG_W - _visual_w) / 2.0
_traj_left = _left_visual + _YLABEL_GUTTER_IN
_soma_right = _traj_left + _TRAJ_W_IN + _WSPACE_IN + _TRAJ_W_IN
_forest_left = _soma_right + _TRAJ_FOREST_GAP_IN
_forest_right = _forest_left + _FOREST_W_IN

PANEL_A_LEFT = _traj_left / FIG_W
PANEL_A_RIGHT = _soma_right / FIG_W
PANEL_A_WSPACE = _WSPACE_IN / _TRAJ_W_IN
FOREST_L = _forest_left / FIG_W
FOREST_R = _forest_right / FIG_W

# Drop the row-1 legend lower than the main figure's so there's clearer vertical
# space between it and the x-axis labels (axis labels stay at F.AXIS_LABEL_Y,
# 0.30 in below the plot; the legend sits 0.46 in below).
LEGEND_Y = F._yt(F._R1_TOP_DIST + F._R1_PLOT_H_IN + 0.46)


def render(save: bool = True) -> None:
    init_figure_theme()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Panel a: trajectory (Olink + SomaScan, 1×2 grid)
    gs_a = fig.add_gridspec(
        1,
        2,
        left=PANEL_A_LEFT,
        right=PANEL_A_RIGHT,
        top=F.PLOT_TOP,
        bottom=F.PLOT_BOT,
        wspace=PANEL_A_WSPACE,
    )
    ax_o = fig.add_subplot(gs_a[0, 0])
    ax_s = fig.add_subplot(gs_a[0, 1])
    F._render_trajectory_panel(ax_o, "olink", GENE, show_x=True, show_ylabel=True)
    F._render_trajectory_panel(ax_s, "soma", GENE, show_x=True, show_ylabel=False)

    # Panel b: STEP forest (single axis)
    gs_b = fig.add_gridspec(
        1,
        1,
        left=FOREST_L,
        right=FOREST_R,
        top=F.PLOT_TOP,
        bottom=F.PLOT_BOT,
    )
    ax_b = fig.add_subplot(gs_b[0, 0])
    F._draw_step_forest(ax_b, GENE)

    fig.canvas.draw()

    # Panel a banner — gene · contrast · covariates, with platform edge labels.
    F._draw_shared_banner(
        fig,
        ax_o,
        ax_s,
        bold_text=GENE,
        tail_text=" · TZP vs SEMA · adj. age, sex, baseline protein",
        left_edge_text="Olink",
        right_edge_text="SomaScan",
    )
    # Panel b banner (forest) — spans only the forest plot box.
    F._draw_shared_banner(
        fig,
        ax_b,
        ax_b,
        bold_text=GENE,
        tail_text=" · SEMA vs PBO",
    )

    # NS-criterion legend in the gap between the forest banner and its plot top.
    bb_b = ax_b.get_position()
    ns_y = bb_b.y1 + F.BANNER_GAP / 2
    ns_handle = mlines.Line2D(
        [0],
        [0],
        marker="o",
        linewidth=0,
        markerfacecolor=COLOR_NS,
        markeredgecolor=c["black"],
        markeredgewidth=0.3,
        markersize=4,
        label=f"NS: adj. P ≥ {F.SIG_THRESHOLD}",
    )
    fig.legend(
        handles=[ns_handle],
        loc="center",
        bbox_to_anchor=((bb_b.x0 + bb_b.x1) / 2, ns_y),
        fontsize=FS_AUX,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.3,
        borderpad=0,
        borderaxespad=0,
    )

    # Panel letters (a = trajectory, b = forest; c = genotype below).
    bb_o = ax_o.get_position()
    bb_s = ax_s.get_position()
    banner_top = bb_o.y1 + F.BANNER_GAP + F.BANNER_HEIGHT
    for ax, label in ((ax_o, "a"), (ax_b, "b")):
        bb = ax.get_position()
        fig.text(
            bb.x0 - 0.05,
            banner_top,
            label,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # ---- Row-1 axis labels + legends ---------------------------------------
    fig.text(
        (bb_o.x0 + bb_s.x1) / 2,
        F.AXIS_LABEL_Y,
        "Week",
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )
    fig.text(
        (bb_b.x0 + bb_b.x1) / 2,
        F.AXIS_LABEL_Y,
        "log₂ FC (Week 68)",
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )

    # FDR star key (full — REG4 has a * star) left-aligned under the trajectory.
    fig.text(
        bb_o.x0,
        LEGEND_Y,
        FDR_LEGEND_TEXT,
        fontsize=F.BOTTOM_LABEL_FONTSIZE - 0.5,
        color=c["black"],
        ha="left",
        va="center",
        style="italic",
    )
    # Arm legend right-aligned at the trajectory's right edge — clear of both
    # the (full) FDR key on its left and the forest key on its right.
    legend_handles_a = [
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS["TZP"],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS["TZP"],
        ),
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS["SEMA"],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS["SEMA"],
        ),
    ]
    fig.legend(
        handles=legend_handles_a,
        loc="center right",
        bbox_to_anchor=(bb_s.x1, LEGEND_Y),
        ncol=2,
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    # Forest key right-aligned under the forest.
    legend_handles_b = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linewidth=0,
            markerfacecolor=COLOR_NS,
            markeredgecolor=c["black"],
            markeredgewidth=0.3,
            markersize=4,
            label="Primary",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linewidth=0,
            markerfacecolor="white",
            markeredgecolor=COLOR_NS,
            markeredgewidth=0.9,
            markersize=4,
            label="+∆Weight, HbA1c",
        ),
    ]
    fig.legend(
        handles=legend_handles_b,
        loc="center right",
        bbox_to_anchor=(bb_b.x1, LEGEND_Y),
        ncol=2,
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.3,
        columnspacing=0.6,
    )

    # ---- Panel c: GIPR E354Q genotype boxplots (full-width bottom row) ------
    axes_c, (olink_l, olink_r, soma_l, soma_r) = F._draw_genotype_panel(fig, GENE)
    fig.canvas.draw()

    ax_c_first, ax_c_last = axes_c[0], axes_c[-1]
    F._draw_shared_banner(
        fig,
        ax_c_first,
        ax_c_last,
        bold_text=GENE,
        tail_text=" · Δ from baseline · adj. age, sex, genetic PC 1-3, baseline protein",
        gap=F.PANEL_C_BANNER_GAP,
    )
    bb_c1 = axes_c[0].get_position()
    bb_c4 = axes_c[-1].get_position()
    banner_c_mid_y = bb_c1.y1 + F.PANEL_C_BANNER_GAP + F.BANNER_HEIGHT / 2
    fig.text(
        olink_l + 0.006,
        banner_c_mid_y,
        "Olink",
        fontsize=FS_AUX,
        fontweight="bold",
        color="#444",
        ha="left",
        va="center",
    )
    fig.text(
        soma_r - 0.006,
        banner_c_mid_y,
        "SomaScan",
        fontsize=FS_AUX,
        fontweight="bold",
        color="#444",
        ha="right",
        va="center",
    )

    # Panel c label.
    banner_c_top = bb_c1.y1 + F.PANEL_C_BANNER_GAP + F.BANNER_HEIGHT
    fig.text(
        bb_c1.x0 - 0.075,
        banner_c_top,
        "c",
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Vertical divider through the platform gap (Olink pair | SomaScan pair).
    fig.canvas.draw()
    soma_w24_ax = axes_c[2]
    inv = fig.transFigure.inverted()
    tick_left_edges = [
        inv.transform((lbl.get_window_extent().x0, 0))[0]
        for lbl in soma_w24_ax.get_yticklabels()
        if lbl.get_text()
    ]
    soma_tick_left = min(tick_left_edges) if tick_left_edges else soma_l
    div_x = (olink_r + soma_tick_left) / 2
    fig.add_artist(
        mlines.Line2D(
            [div_x, div_x],
            [F.PLOT_C_BOT - 0.045, F.PLOT_C_TOP],
            transform=fig.transFigure,
            color=c["bold_grey"],
            linewidth=0.8,
            zorder=20,
            clip_on=False,
        )
    )

    # X-axis label below panel c.
    fig.text(
        (bb_c1.x0 + bb_c4.x1) / 2,
        F.PANEL_C_AXIS_LABEL_Y,
        "rs1800437 (GIPR E354Q) genotype",
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )

    if save:
        out_path = str(THIS_DIR / "supp_fig16_reg4")
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved → {out_path}.pdf")
        plt.close()


if __name__ == "__main__":
    render()
