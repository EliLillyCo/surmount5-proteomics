"""Overlap check run on every figure at save time: no two texts and no two banners may intersect.

A card and its drop shadow (same size, offset by a few points) and a banner nested inside
another are not overlaps.
"""

from __future__ import annotations

import itertools

# Two boxes count as overlapping when they share more than this much in both directions, in
# points; boxes that only touch, such as a banner's bold and gray halves, pass.
OVERLAP_TOL_PT = 0.5


def _texts(fig) -> list:
    """Every visible, non-empty text of ``fig``: figure texts, axes texts, titles, axis labels, drawn tick labels, legends."""
    found = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        found += ax.texts
        found += [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if axis.get_visible():
                    found += [axis.label, *_drawn_tick_labels(axis)]
        legend = ax.get_legend()
        if legend is not None:
            found += legend.get_texts()
    for legend in fig.legends:
        found += legend.get_texts()
    return [t for t in found if t.get_visible() and t.get_text().strip()]


def _drawn_tick_labels(axis) -> list:
    """The tick labels ``axis`` draws: those of the ticks inside its view limits."""
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-9
    return [
        label
        for tick in axis.get_major_ticks()
        if lo - eps <= tick.get_loc() <= hi + eps
        for label in (tick.label1, tick.label2)
        if label.get_visible()
    ]


def _banners(fig) -> list:
    """Patches drawn outside the plot area: the figure's patches and the axes' unclipped patches."""
    found = list(fig.patches)
    for ax in fig.axes:
        if ax.get_visible():
            found += [p for p in ax.patches if not p.get_clip_on()]
    return [p for p in found if p.get_visible()]


def _overlap(a, b, tol: float) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol


def find_overlaps(fig) -> list[str]:
    """Describe each pair of texts, and each pair of banners, of ``fig`` that intersect.

    Args:
        fig: A figure at its final layout.

    Returns:
        One line per overlapping pair, empty when there is none.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tol = OVERLAP_TOL_PT * fig.dpi / 72.0
    found = []
    texts = [(t.get_text().replace("\n", " "), t.get_window_extent(renderer)) for t in _texts(fig)]
    for (name_a, box_a), (name_b, box_b) in itertools.combinations(texts, 2):
        if _overlap(box_a, box_b, tol):
            found.append(f"text {name_a!r} overlaps text {name_b!r}")
    banners = [p.get_window_extent(renderer) for p in _banners(fig)]
    for (i, box_a), (j, box_b) in itertools.combinations(enumerate(banners), 2):
        if (
            _overlap(box_a, box_b, tol)
            and not (_inside(box_a, box_b) or _inside(box_b, box_a))
            and not _shadow(box_a, box_b, tol)
        ):
            found.append(f"banner {i} overlaps banner {j} at y {box_a.y0:.0f}-{box_a.y1:.0f} / {box_b.y0:.0f}-{box_b.y1:.0f} px")
    return found


def _shadow(a, b, tol: float) -> bool:
    """Same size and offset by a few points: a card and its drop shadow."""
    return abs(a.width - b.width) <= tol and abs(a.height - b.height) <= tol and abs(a.x0 - b.x0) < 8 * tol and abs(a.y0 - b.y0) < 8 * tol


def _inside(inner, outer) -> bool:
    return inner.x0 >= outer.x0 and inner.x1 <= outer.x1 and inner.y0 >= outer.y0 and inner.y1 <= outer.y1


def check_overlaps(fig) -> None:
    """Raise when any two texts or any two banners of ``fig`` intersect.

    Raises:
        RuntimeError: Listing every overlapping pair.
    """
    found = find_overlaps(fig)
    if found:
        raise RuntimeError(f"{len(found)} overlap(s) in the figure:\n  " + "\n  ".join(found))
