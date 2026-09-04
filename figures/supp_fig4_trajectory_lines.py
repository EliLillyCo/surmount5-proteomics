"""Supplementary trajectory-line exemplars (companion to Fig 1 panel c).

2×4 grid: row 0 = Olink, row 1 = SomaScan. Each column shows a different
trajectory class or platform-specific protein. Panels for proteins unique to
one platform carry a "Not covered by <other>" note at baseline. Same line-plot
style as panel c, with across-treatment FDR stars as in the other trajectory
supplements. Reuses the panel primitives from ``_trajectory_lines`` so the
styling stays identical.

Olink row (L→R):  HSD11B1 (Progressive ↑),  LPL (Reversal ↓↑),
                  DMP1* (Sustained ↑),        GUCA2A* (Reversal ↓↑)
SomaScan row:     CRP* (Progressive ↓),       FABP3 (Reversal ↑↓),
                  HAMP* (Sustained ↑),         APOC3* (Sustained ↓)
(* = platform-specific)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _internal._common import (  # noqa: E402
    ARM_COLORS,
    ARM_LABELS,
    BANNER_HEIGHT_IN,
    FS_EMPH,
    FS_PANEL,
    NAT_W2,
    c,
    init_figure_theme,
    save_figure,
)
import _internal._trajectory_lines as tl  # noqa: E402

# Olink-specific (row 0) and SomaScan-specific (row 1) exemplars.
# Columns 2-3 of each row are truly platform-specific (noted in the panel).
PLATFORM_GENES: dict[str, list[str]] = {
    "olink": ["HSD11B1", "LPL", "DMP1", "GUCA2A"],
    "soma": ["CRP", "FABP3", "HAMP", "APOC3"],
}

_OTHER_LABEL = {"olink": "SomaScan", "soma": "Olink"}


def render() -> None:
    init_figure_theme()

    # 2 rows (Olink, SomaScan) x 4 cols. Geometry matched to supp_fig5 (same
    # column layout, same left/right/top/bottom margins, same wspace/hspace) so
    # panel aspect ratios are consistent across trajectory supplements. Figure
    # height derived so panels come out at ~1.21 width/height, matching the
    # standard PCBL aspect used throughout the supplement.
    fig = plt.figure(figsize=(NAT_W2, 3.81))
    gs = fig.add_gridspec(
        2,
        4,
        left=0.075,
        right=0.985,
        top=0.78,
        bottom=0.075,
        wspace=0.35,
        hspace=0.50,
    )

    coverage = tl._genes_per_platform()
    plat_axes: dict[str, list] = {"olink": [], "soma": []}
    for r, platform in enumerate(("olink", "soma")):
        other_genes = coverage[{"olink": "soma", "soma": "olink"}[platform]]
        other_label = _OTHER_LABEL[platform]
        for ci, gene in enumerate(PLATFORM_GENES[platform]):
            ax = fig.add_subplot(gs[r, ci])
            not_covered = other_label if gene not in other_genes else None
            traj = tl._draw_panel(
                ax,
                platform,
                gene,
                show_x=True,
                not_covered_by=not_covered,
                show_fdr_stars=False,
            )
            tl._draw_subplot_header(ax, gene, traj, show_name=True)
            plat_axes[platform].append(ax)

    # --- Platform banners + panel letters -----------------------------------
    def _platform_banner(axes: list, label: str, panel_label: str) -> None:
        x0 = min(a.get_position().x0 for a in axes)
        x1 = max(a.get_position().x1 for a in axes)
        gene_top_y = max(
            a.get_position().y1 + 0.24 * a.get_position().height for a in axes
        )
        banner_height = BANNER_HEIGHT_IN / fig.get_figheight()
        banner_y = gene_top_y + 0.022
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (x0, banner_y),
                x1 - x0,
                banner_height,
                boxstyle="square,pad=0",
                facecolor="#E8E8E8",
                edgecolor="none",
                transform=fig.transFigure,
                clip_on=False,
            )
        )
        fig.text(
            (x0 + x1) / 2,
            banner_y + banner_height / 2,
            label,
            fontsize=FS_EMPH,
            fontweight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            x0 - 0.030,
            banner_y + banner_height + 0.004,
            panel_label,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    _platform_banner(plat_axes["olink"], "Olink", "a")
    _platform_banner(plat_axes["soma"], "SomaScan", "b")

    # --- Y-axis label (single, centred on both platform blocks) ---------------
    all_axes = plat_axes["olink"] + plat_axes["soma"]
    fig_cy = (
        min(a.get_position().y0 for a in all_axes)
        + max(a.get_position().y1 for a in all_axes)
    ) / 2
    fig.text(
        0.018,
        fig_cy,
        "% Change from baseline (95% CI)",
        fontsize=FS_EMPH,
        ha="center",
        va="center",
        rotation=90,
    )

    # --- Bottom strip: Week (center) + arm legend (right), single line --------
    LEFT, RIGHT = 0.075, 0.985
    STRIP_Y = 0.030
    plot_center_x = (LEFT + RIGHT) / 2

    fig.text(plot_center_x, STRIP_Y, "Week", fontsize=FS_EMPH, ha="center", va="center")

    legend_handles = [
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
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(RIGHT, STRIP_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=FS_EMPH,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    out_path = str(THIS_DIR / "supp_fig4_trajectory_lines")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
