"""Supplemental figure: proteins selected by the TZP-induced rule (a week-72 increase from
baseline in both arms, larger with TZP).

Per-marker %CFB line plots (Wk0/24/72) for TZP and SEMA. The week-72 rule (larger-change arm
significant), the exclusions and the ranking live in ``analysis.trajectory_panels``; the
platform's panels are filled row-major in rank order, so a gene's position is its rank on
that platform, not a shared slot.

FDR stars are drawn above the higher arm's percentage label (the suppressed figure
places them below the lower arm).

Layout: 4 rows × 4 cols total, Olink left 2 cols, SomaScan right 2 cols.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figures._internal._common import init_figure_theme, save_figure
import figures._internal._trajectory_lines as tl
from figures._internal._trajectory_lines import MarkerSpec
from analysis.trajectory_panels import (
    EXCLUDED_MARKERS,
    TZP_INDUCED,
    load_panel_frame,
    select_panels,
)

THIS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Marker layout: the platform's top-``TOP_K`` panels under the week-72 rule
# (``analysis.trajectory_panels``), in rule order, filled row-major. Nothing about
# the markers is hardcoded here.
# ---------------------------------------------------------------------------
def panel_layout(platform: str) -> list[MarkerSpec]:
    """The platform's panels for this figure, ranked by late between-treatment FDR."""
    chosen = select_panels(
        load_panel_frame(platform), TZP_INDUCED, excluded_markers=EXCLUDED_MARKERS[platform]
    )
    return [MarkerSpec(m, g) for m, g in chosen.select("marker", "label").iter_rows()]


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def render() -> None:
    """Build Supplementary Fig 2: the TZP-induced proteins, Olink and SomaScan side by side, and write ``figures/supp_fig_tzp_induced.pdf``."""
    init_figure_theme()

    NROWS = 4
    PANEL_COLS = 2
    NCOLS = PANEL_COLS * 2  # Olink + Soma side by side

    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(
        NROWS,
        NCOLS,
        left=0.075,
        right=0.985,
        top=0.78,
        bottom=0.075,
        wspace=0.35,
        hspace=0.50,
    )

    def _populate(
        panel: list[MarkerSpec | None], platform: str, col_offset: int
    ) -> list:
        axes_list: list = []
        for slot, spec in enumerate(panel):
            r, sub_c = divmod(slot, PANEL_COLS)
            ax = fig.add_subplot(gs[r, col_offset + sub_c])
            if spec is None:
                ax.set_axis_off()
                continue
            tl.draw_marker_panel(ax, platform, spec)
            axes_list.append(ax)
        return axes_list

    olink_axes = _populate(panel_layout("olink"), "olink", col_offset=0)
    soma_axes = _populate(panel_layout("soma"), "soma", col_offset=PANEL_COLS)

    # --- Platform banners + panel labels ---
    def _platform_banner(axes: list, label: str, panel_label: str) -> None:
        if not axes:
            return
        x0 = min(a.get_position().x0 for a in axes)
        x1 = max(a.get_position().x1 for a in axes)
        tl.draw_platform_banner(fig, axes, label, panel_label, x0=x0, x1=x1, letter_dy=0.003)

    _platform_banner(olink_axes, "Olink", "a")
    _platform_banner(soma_axes, "SomaScan", "b")

    # --- Y-axis label, hugging the plot column ---
    tl.draw_ylabel(fig, 0.020, 0.43)

    # --- Bottom strip: FDR legend (left), Week label (center), arm legend (right) ---
    tl.draw_bottom_strip(fig, 0.075, 0.985, 0.028)

    out_path = str(THIS_DIR / "supp_fig_tzp_induced")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
