"""Supplemental figure: proteins selected by the TZP-suppressed rule (a week-72 decrease from
baseline in both arms, larger with TZP).

Mirror of supp_fig_tzp_induced.py for the down-regulated direction. The week-72 rule
(larger-change arm significant), the exclusions and the ranking live in
``analysis.trajectory_panels``; nothing about the markers is hardcoded here.

Layout: stacked, Olink block on top (up to ``TOP_K`` panels, 4 cols), SomaScan block below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figures._internal._common import init_figure_theme, save_figure
import figures._internal._trajectory_lines as tl
from figures._internal._trajectory_lines import MarkerSpec
from analysis.trajectory_panels import (
    EXCLUDED_MARKERS,
    TZP_SUPPRESSED,
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
        load_panel_frame(platform), TZP_SUPPRESSED, excluded_markers=EXCLUDED_MARKERS[platform]
    )
    return [MarkerSpec(m, g) for m, g in chosen.select("marker", "label").iter_rows()]


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def render() -> None:
    """Build Supplementary Fig 3: the TZP-suppressed proteins, Olink above SomaScan, and write ``figures/supp_fig_tzp_suppressed.pdf``."""
    init_figure_theme()

    # Stars below the lower arm; only the bottom of each panel is extended to fit them.
    fig = tl.stacked_platform_figure(
        panel_layout("olink"), panel_layout("soma"), stars_above=False, fit_labels=False
    )

    out_path = str(THIS_DIR / "supp_fig_tzp_suppressed")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
