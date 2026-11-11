"""Supplemental figure: TZP-vs-SEMA opposing within-arm responses.

Analytes with a significant between-treatment difference at Wk72 (FDR < 0.05)
whose within-arm %CFB moves in opposite directions in the two arms at both
Wk24 and Wk72. Per platform the row shows the four with the lowest Wk72
between-treatment FDR (fewer if fewer qualify). The rule, the criterion and the
excluded assays live in ``analysis.opposing``; nothing about the markers is hardcoded here.

Per-marker %CFB line plots (Wk0/24/72) for TZP and SEMA. FDR stars give the
between-treatment FDR at each visit and sit above the higher arm.

Layout: 2 platform rows (Olink top, SomaScan bottom), up to 4 cols each; a
platform with fewer than 4 opposing analytes gets fewer panels of the same size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figures._internal._common import init_figure_theme, nature_figsize, save_figure
import figures._internal._trajectory_lines as tl
from figures._internal._trajectory_lines import MarkerSpec
from analysis.design import PLATFORMS
from analysis.opposing import (
    EXCLUDED_MARKERS,
    TOP_K,
    load_opposing_frame,
    select_opposing,
)
from analysis.protein_labels import marker_labels

THIS_DIR = Path(__file__).resolve().parent


def opposing_layout(platform: str) -> list[MarkerSpec]:
    """The platform's top-``TOP_K`` opposing analytes (see ``analysis.opposing``)."""
    chosen = select_opposing(load_opposing_frame(platform), excluded=EXCLUDED_MARKERS[platform])
    labels = marker_labels(platform)
    return [MarkerSpec(m, labels.get(m, m)) for m in chosen["marker"]]


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def render() -> None:
    """Build Supplementary Fig 6: the opposing-response proteins, and write ``figures/supp_fig_tzp_opposing.pdf``."""
    init_figure_theme()

    # Two platform rows (Olink top, SomaScan bottom), 4 cols each. Each row is its own
    # single-row gridspec and the rows are separated by an explicit absolute-inch GAP_IN
    # budgeted to hold the SomaScan banner + its (trajectory pill + marker + gene) headers +
    # the Olink x-axis labels + clearance, so the SomaScan banner never crowds the bottom of
    # the Olink plots. Panel height is set for the trajectory panel aspect.
    COLS = TOP_K
    LEFT, RIGHT = tl.STACK_LEFT, tl.STACK_RIGHT
    WSPACE = tl.STACK_WSPACE

    fig_w = nature_figsize("2col", 1.0)[0]
    PANEL_H_IN = tl.stacked_panel_height_in(fig_w, COLS)
    TOP_IN = 0.78  # Olink banner + trajectory pill + marker + gene headers
    GAP_IN = 1.30  # SomaScan banner + headers + Olink x-labels + clearance
    BOTTOM_IN = 0.66  # x-labels + FDR/Week/arm legend strip

    height = TOP_IN + PANEL_H_IN + GAP_IN + PANEL_H_IN + BOTTOM_IN
    soma_bottom = BOTTOM_IN / height
    soma_top = (BOTTOM_IN + PANEL_H_IN) / height
    olink_bottom = (BOTTOM_IN + PANEL_H_IN + GAP_IN) / height
    olink_top = (BOTTOM_IN + PANEL_H_IN + GAP_IN + PANEL_H_IN) / height

    fig = plt.figure(figsize=(fig_w, height))
    gs_by_platform = {
        "olink": fig.add_gridspec(
            1, COLS, left=LEFT, right=RIGHT, top=olink_top, bottom=olink_bottom,
            wspace=WSPACE,
        ),
        "soma": fig.add_gridspec(
            1, COLS, left=LEFT, right=RIGHT, top=soma_top, bottom=soma_bottom,
            wspace=WSPACE,
        ),
    }

    plat_axes: dict[str, list] = {platform: [] for platform in PLATFORMS}
    for platform in PLATFORMS:
        for col, spec in enumerate(opposing_layout(platform)):
            ax = fig.add_subplot(gs_by_platform[platform][0, col])
            tl.draw_marker_panel(ax, platform, spec)
            plat_axes[platform].append(ax)

    # --- Platform banners + panel letters (the full grid width, whatever number of panels) ---
    for platform, label, letter in (("olink", "Olink", "a"), ("soma", "SomaScan", "b")):
        tl.draw_platform_banner(
            fig, plat_axes[platform], label, letter, x0=LEFT, x1=RIGHT, gap=0.020, letter_dx=0.030
        )

    # --- Y-axis label (centered across both platform blocks) ------------------
    all_axes = plat_axes["olink"] + plat_axes["soma"]
    fig_cy = (
        min(a.get_position().y0 for a in all_axes)
        + max(a.get_position().y1 for a in all_axes)
    ) / 2
    tl.draw_ylabel(fig, 0.020, fig_cy)

    # --- Bottom strip, ~0.30 in above the figure bottom, clear of the x-labels ---
    tl.draw_bottom_strip(fig, LEFT, RIGHT, 0.30 / height)

    out_path = str(THIS_DIR / "supp_fig_tzp_opposing")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
