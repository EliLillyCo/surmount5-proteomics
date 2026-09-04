"""Shared annotation and layout helpers for manuscript figures."""

from __future__ import annotations

import matplotlib.patches as mpatches

from ._theme import FS_BANNER_BOLD, FS_BANNER_TAIL, FS_PANEL

# ---------------------------------------------------------------------------
# Shared text-placement helpers
# ---------------------------------------------------------------------------


def draw_segments_right_aligned(fig, x_right, y, segments, **kwargs):
    """Place text segments right-aligned ending at x_right (figure coords).

    segments: list of (text, color) tuples.
    """
    renderer = fig.canvas.get_renderer()
    artists = []
    for text, color in segments:
        t = fig.text(
            0,
            y,
            text,
            ha="left",
            va="center",
            color=color,
            **kwargs,
        )
        artists.append(t)
    widths_frac = [
        a.get_window_extent(renderer=renderer).width / (fig.dpi * fig.get_figwidth())
        for a in artists
    ]
    cur_x = x_right - sum(widths_frac)
    for a, w in zip(artists, widths_frac):
        a.set_x(cur_x)
        cur_x += w


def draw_segments_left_aligned(fig, x_left, y, segments, **kwargs):
    """Place text segments left-aligned starting at x_left (figure coords)."""
    renderer = fig.canvas.get_renderer()
    cur_x = x_left
    for text, color in segments:
        t = fig.text(
            cur_x,
            y,
            text,
            ha="left",
            va="center",
            color=color,
            **kwargs,
        )
        bbw = t.get_window_extent(renderer=renderer).width
        cur_x += bbw / (fig.dpi * fig.get_figwidth())


# ---------------------------------------------------------------------------
# Shared panel-letter and banner helpers (gaps in INCHES = width-invariant)
# ---------------------------------------------------------------------------
# Offsets are specified in inches and converted to figure fraction using the
# live figwidth/figheight, so the same call gives the same visual gap at any
# Nature width. Centralised here so every figure places letters and banners
# identically.

PANEL_DX_IN = 0.10  # letter inset to the LEFT of the axis left edge
PANEL_DY_IN = 0.04  # letter rise ABOVE the axis top edge
# Canonical banner strip height (inches). 0.20in ≈ 5.2mm, matching the
# supp_fig5_tzp_induced banner that sets the house style. Use the SAME physical
# height on every figure (convert to fraction via fig.get_figheight()).
BANNER_HEIGHT_IN = 0.20
BANNER_GAP_IN = 0.04  # gap between panel top and banner bottom
BANNER_EDGE_INSET_IN = 0.05  # inset for left/right edge labels

# Canonical aspect ratio (width / height) of a "% change from baseline"
# trajectory subplot. Taken from supp_fig5_tzp_induced (the reference figure).
# Every trajectory figure is sized so its panels render at this ratio.
TRAJ_PANEL_ASPECT = 1.21


def place_panel_letter(
    fig,
    ax,
    label: str,
    *,
    dx_in: float = PANEL_DX_IN,
    dy_in: float = PANEL_DY_IN,
    extra_dy_in: float = 0.0,
    fontsize: float = FS_PANEL,
    **kwargs,
):
    """Place a bold panel letter just outside the top-left corner of ``ax``.

    ``dx_in`` / ``dy_in`` are inch offsets (width-invariant). ``extra_dy_in``
    adds clearance for figures that put a banner above the panel (the letter
    then sits above the banner). Returns the Text artist.
    """
    fig.canvas.draw()
    bb = ax.get_position()
    dx = dx_in / fig.get_figwidth()
    dy = (dy_in + extra_dy_in) / fig.get_figheight()
    return fig.text(
        bb.x0 - dx,
        bb.y1 + dy,
        label,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
        **kwargs,
    )


def draw_banner(
    fig,
    ax_left,
    ax_right,
    *,
    bold_text: str,
    tail_text: str = "",
    left_edge_text: str = "",
    right_edge_text: str = "",
    height_in: float = BANNER_HEIGHT_IN,
    gap_in: float = BANNER_GAP_IN,
    fontsize_bold: float = FS_BANNER_BOLD,
    fontsize_tail: float = FS_BANNER_TAIL,
    facecolor: str = "#E8E8E8",
) -> None:
    """Light-grey banner strip spanning a (left, right) axes pair.

    Bold-prefix + grey-tail render as a single centred block. Optional
    ``left_edge_text`` / ``right_edge_text`` sit at the inner banner edges
    (e.g. platform labels). ``height_in`` / ``gap_in`` are inches.
    """
    height = height_in / fig.get_figheight()
    gap = gap_in / fig.get_figheight()
    edge_inset = BANNER_EDGE_INSET_IN / fig.get_figwidth()

    bb_left = ax_left.get_position()
    bb_right = ax_right.get_position()
    banner_y = bb_left.y1 + gap
    patch = mpatches.FancyBboxPatch(
        (bb_left.x0, banner_y),
        bb_right.x1 - bb_left.x0,
        height,
        boxstyle="square,pad=0",
        facecolor=facecolor,
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(patch)
    mid_x = (bb_left.x0 + bb_right.x1) / 2
    mid_y = banner_y + height / 2

    t_bold = fig.text(
        mid_x,
        mid_y,
        bold_text,
        fontsize=fontsize_bold,
        fontweight="bold",
        ha="left",
        va="center",
    )
    t_tail = None
    if tail_text:
        t_tail = fig.text(
            mid_x,
            mid_y,
            tail_text,
            fontsize=fontsize_tail,
            color="#666666",
            ha="left",
            va="center",
        )

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bb_b = t_bold.get_window_extent().transformed(inv)
    w_bold = bb_b.x1 - bb_b.x0
    w_tail = 0.0
    if t_tail is not None:
        bb_t = t_tail.get_window_extent().transformed(inv)
        w_tail = bb_t.x1 - bb_t.x0

    new_bold_x = mid_x - (w_bold + w_tail) / 2
    t_bold.set_x(new_bold_x)
    if t_tail is not None:
        t_tail.set_x(new_bold_x + w_bold)

    if left_edge_text:
        fig.text(
            bb_left.x0 + edge_inset,
            mid_y,
            left_edge_text,
            fontsize=fontsize_tail,
            fontweight="bold",
            color="#444",
            ha="left",
            va="center",
        )
    if right_edge_text:
        fig.text(
            bb_right.x1 - edge_inset,
            mid_y,
            right_edge_text,
            fontsize=fontsize_tail,
            fontweight="bold",
            color="#444",
            ha="right",
            va="center",
        )
