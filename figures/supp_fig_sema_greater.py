"""Supplemental figure: proteins with a larger week-72 change under SEMA than under TZP.

The rule (same-sign change in both arms, significant between-treatment difference,
larger-change arm significant), the exclusions and the ranking live in
``analysis.trajectory_panels`` and are evaluated on every platform; nothing about the
markers is hardcoded here. A platform where no analyte meets the rule is left out. Panels are
filled row-major in the trajectory-class order of ``display_order`` (shared class, then
split, then no TZP change), rank order within each.

Per-marker %CFB line plots (Wk0/24/72) for TZP and SEMA, drawn with ``_trajectory_lines.draw_marker_panel``
(FDR stars above the higher arm). Layout: one block per drawn
platform, Olink above SomaScan, each up to ``TOP_K`` panels in 4 columns; the blocks carry
panel letters when there is more than one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figures._internal._common import init_figure_theme, nature_figsize, save_figure
from figures._internal._theme import PLATFORM_SHORT
import figures._internal._trajectory_data as td
import figures._internal._trajectory_lines as tl
from figures._internal._trajectory_lines import MarkerSpec
from analysis.qc_contract import PLATFORMS
from analysis.trajectory_panels import (
    EXCLUDED_MARKERS,
    SEMA_GREATER,
    display_order,
    load_panel_frame,
    select_panels,
)

THIS_DIR = Path(__file__).resolve().parent


def panel_layout(platform: str) -> list[tuple[str, str]]:
    """The platform's (marker, gene) panels in drawing order (``display_order``)."""
    chosen = select_panels(
        load_panel_frame(platform), SEMA_GREATER, excluded_markers=EXCLUDED_MARKERS[platform]
    )
    ranked = [
        (marker, gene, *td.trajectory_classes(platform, marker))
        for marker, gene in chosen.select("marker", "label").iter_rows()
    ]
    return [(marker, gene) for marker, gene, _, _ in display_order(ranked)]


def drawn_layouts() -> dict[str, list[tuple[str, str]]]:
    """Each platform with at least one qualifying panel -> its panels, in ``PLATFORMS`` order.

    Raises:
        ValueError: When no platform has a qualifying analyte.
    """
    layouts = {platform: panel_layout(platform) for platform in PLATFORMS}
    drawn = {platform: layout for platform, layout in layouts.items() if layout}
    if not drawn:
        raise ValueError("no platform has an analyte meeting the SEMA-greater rule")
    return drawn


def render() -> None:
    """Build Supplementary Fig 5: the SEMA-greater proteins, one block per platform, and write ``figures/supp_fig_sema_greater.pdf``."""
    init_figure_theme()

    COLS = tl.STACK_COLS
    LEFT, RIGHT = tl.STACK_LEFT, tl.STACK_RIGHT
    layouts = drawn_layouts()
    rows = {platform: -(-len(layout) // COLS) for platform, layout in layouts.items()}

    # Same panel size and spacing as the stacked platform blocks of the TZP-suppressed figure.
    fig_w = nature_figsize("2col", 1.0)[0]
    PANEL_H_IN = tl.stacked_panel_height_in(fig_w)
    HSPACE = tl.STACK_HSPACE
    GAP_IN = tl.STACK_GAP_IN
    BOTTOM_IN = tl.STACK_BOTTOM_IN
    block_in = {p: PANEL_H_IN * (n + (n - 1) * HSPACE) for p, n in rows.items()}
    height = tl.STACK_TOP_IN + sum(block_in.values()) + GAP_IN * (len(layouts) - 1) + BOTTOM_IN

    # Block extents (figure fractions), stacked up from the bottom strip.
    extents: dict[str, tuple[float, float]] = {}
    cursor = BOTTOM_IN
    for platform in reversed(layouts):
        extents[platform] = (cursor / height, (cursor + block_in[platform]) / height)
        cursor += block_in[platform] + GAP_IN
    bottom = min(lo for lo, _ in extents.values())
    top = max(hi for _, hi in extents.values())

    fig = plt.figure(figsize=nature_figsize("2col", height))

    for index, (platform, layout) in enumerate(layouts.items()):
        block_bottom, block_top = extents[platform]
        gs = fig.add_gridspec(
            rows[platform],
            COLS,
            left=LEFT,
            right=RIGHT,
            top=block_top,
            bottom=block_bottom,
            wspace=tl.STACK_WSPACE,
            hspace=HSPACE,
        )
        axes: list = []
        for slot, (marker, gene) in enumerate(layout):
            r, col = divmod(slot, COLS)
            ax = fig.add_subplot(gs[r, col])
            tl.draw_marker_panel(ax, platform, MarkerSpec(marker, gene))
            axes.append(ax)

        # --- Platform banner; the blocks carry letters when there is more than one ---
        letter = chr(ord("a") + index) if len(layouts) > 1 else None
        tl.draw_platform_banner(fig, axes, PLATFORM_SHORT[platform], letter, x0=LEFT, x1=RIGHT)

    tl.draw_ylabel(fig, 0.018, (top + bottom) / 2)
    tl.draw_bottom_strip(fig, LEFT, RIGHT, bottom * 0.32)

    out_path = str(THIS_DIR / "supp_fig_sema_greater")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
