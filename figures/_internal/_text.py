"""Deterministic text-placement helpers for the figures."""

from __future__ import annotations

import itertools

import numpy as np
from adjustText import adjust_text as _adjust_text
from matplotlib.patches import FancyArrowPatch


def deterministic_adjust_text(*args, seed: int = 0, iter_lim: int = 500, **kwargs):
    """Run adjustText with fixed RNG and iteration limits so rerenders stay stable."""
    if "iter_lim" not in kwargs and "time_lim" not in kwargs:
        kwargs["iter_lim"] = iter_lim
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return _adjust_text(*args, **kwargs)
    finally:
        np.random.set_state(state)


def separate_overlapping_labels(
    ax,
    movable: list,
    targets: list,
    arrows: list,
    *,
    arrowprops: dict,
    min_arrow_len: float,
    pad_pt: float = 1.0,
) -> list[str]:
    """Move as few labels as possible, as little as possible, until no two texts of ``ax`` overlap.

    Meant as the last step after ``deterministic_adjust_text`` and at the figure's final
    geometry. Nothing is random, so a rerender gives the same labels. For the first pair of
    texts whose bounding boxes intersect, the candidates are each movable member shifted
    left, right, up or down just far enough to clear one neighbor, or all the neighbors
    it touches, by ``pad_pt``. A candidate that would touch another text or leave the axes
    is dropped; of the rest, the one that leaves the fewest overlapping pairs in the axes,
    and then the shortest shift, is applied, and the search repeats. The label keeps its
    leader line: the arrow adjustText drew is re-ended at the label's new position, or a
    new one is added when the label now sits at least ``min_arrow_len`` pixels from its point.

    Args:
        ax: The axes holding the labels; every visible text in ``ax.texts`` is an obstacle.
        movable: Labels that may move, centered on their position (as adjustText leaves them).
        targets: Data-space ``(x, y)`` of the point each movable label names, or None for a
            label that must not move (it has leader lines drawn at its current position).
        arrows: The arrow patches ``deterministic_adjust_text`` returned.
        arrowprops: Arrow style for a leader line that has to be added.
        min_arrow_len: Distance in pixels from the label box to its point beyond which a
            leader line is drawn.
        pad_pt: Clearance left between a moved label and its neighbors, in points.

    Returns:
        The moved labels' strings, in the order they moved.

    Raises:
        RuntimeError: When an overlapping pair has no movable member that can clear it.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72.0
    inv = ax.transData.inverted()
    axes_box = ax.get_window_extent(renderer)
    can_move = {id(t): target for t, target in zip(movable, targets) if target is not None}
    moved: list[str] = []

    def boxes():
        return {
            id(t): (t, t.get_window_extent(renderer))
            for t in ax.texts
            if t.get_visible() and t.get_text().strip()
        }

    def intersect(a, b, margin=0.0):
        return min(a.x1, b.x1) - max(a.x0, b.x0) > -margin and min(a.y1, b.y1) - max(a.y0, b.y0) > -margin

    for _ in range(len(ax.texts) ** 2):
        current = boxes()
        pairs = sorted(
            (t1.get_text(), t2.get_text(), k1, k2)
            for (k1, (t1, b1)), (k2, (t2, b2)) in itertools.combinations(current.items(), 2)
            if intersect(b1, b2)
        )
        if not pairs:
            return moved
        _, _, k1, k2 = pairs[0]
        best = None
        # Overlapping pairs that do not involve a given label: they stay whichever way it moves.
        apart = {k: sum(1 for p in pairs if k not in p[2:]) for k in (k1, k2)}
        for mover in (k1, k2):
            if mover not in can_move:
                continue
            text, box = current[mover]
            touching = [b for k, (_, b) in current.items() if k != mover and intersect(box, b)]
            # Each way to clear one neighbor, and each way to clear all of them at once.
            shifts = []
            for group in [[b] for b in touching] + [touching]:
                shifts += [
                    (min(b.x0 for b in group) - box.x1 - pad, 0.0),
                    (max(b.x1 for b in group) - box.x0 + pad, 0.0),
                    (0.0, min(b.y0 for b in group) - box.y1 - pad),
                    (0.0, max(b.y1 for b in group) - box.y0 + pad),
                ]
            for dx, dy in shifts:
                new = box.translated(dx, dy)
                inside = (
                    new.x0 >= axes_box.x0 and new.x1 <= axes_box.x1 and new.y0 >= axes_box.y0 and new.y1 <= axes_box.y1
                )
                others = [b for k, (_, b) in current.items() if k != mover]
                left = sum(intersect(new, b) for b in others) + apart[mover]
                clear = all(not intersect(new, b, pad / 2) for b in others if not intersect(box, b))
                score = (left, abs(dx) + abs(dy))
                if inside and clear and (best is None or score < best[0]):
                    best = (score, text, dx, dy)
        if best is None:
            names = (current[k1][0].get_text(), current[k2][0].get_text())
            raise RuntimeError(f"no clear position for either of the overlapping labels {names}")
        _, text, dx, dy = best
        cx, cy = ax.transData.transform(text.get_position())
        new_x, new_y = inv.transform((cx + dx, cy + dy))
        text.set_position((new_x, new_y))
        moved.append(text.get_text())
        target = can_move[id(text)]
        arrow = next((a for a in arrows if a.patchA is text), None)
        if arrow is not None:
            arrow.set_positions((new_x, new_y), target)
        else:
            box = text.get_window_extent(renderer)
            tx, ty = ax.transData.transform(target)
            gap = max(box.x0 - tx, tx - box.x1, box.y0 - ty, ty - box.y1, 0.0)
            if gap >= min_arrow_len:
                arrow = FancyArrowPatch(
                    posA=(new_x, new_y), posB=target, patchA=text, transform=ax.transData, **arrowprops
                )
                ax.add_patch(arrow)
                arrows.append(arrow)
    raise RuntimeError("label separation did not converge")
