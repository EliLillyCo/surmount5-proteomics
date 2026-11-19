"""Figure sizing constants shared across the figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

_MM = 1.0 / 25.4  # millimetre -> inch

NAT_W1 = 89 * _MM  # 3.504 in — single column
NAT_W15A = 120 * _MM  # 4.724 in — 1.5 column (narrow)
NAT_W15B = 136 * _MM  # 5.354 in — 1.5 column (wide)
NAT_W2 = 183 * _MM  # 7.205 in — double column
NAT_HMAX = 247 * _MM  # 9.724 in — full page depth (hard ceiling)

_NAT_WIDTHS: dict[str, float] = {
    "1col": NAT_W1,
    "1.5col-narrow": NAT_W15A,
    "1.5col": NAT_W15B,
    "2col": NAT_W2,
}


def nature_figsize(width, height_in: float) -> tuple[float, float]:
    """Return (width_in, height_in) for a figure of the given width, in inches."""
    w = _NAT_WIDTHS[width] if isinstance(width, str) else float(width)
    assert height_in <= NAT_HMAX + 1e-6, (
        f"figure height {height_in:.2f}in exceeds the page depth "
        f"{NAT_HMAX:.2f}in (247 mm)"
    )
    return (w, height_in)

class PanelFigure(Figure):
    """A figure whose ``draw_without_rendering`` leaves its dpi as it was.

    matplotlib draws without rendering on a renderer of the default save format; with
    ``savefig.format = "pdf"`` that sets the figure dpi to 72 and leaves it there, so text
    measured afterwards (label repulsion, banner centering) would be measured at 72 dpi.
    """

    def draw_without_rendering(self) -> None:
        """Draw the figure without output, then restore its dpi."""
        dpi = self.dpi
        super().draw_without_rendering()
        self.set_dpi(dpi)


def panel_figure(width: float, height: float) -> PanelFigure:
    """A ``PanelFigure`` of ``width`` x ``height`` inches, to lay out with ``panel_grid``."""
    return plt.figure(figsize=(width, height), FigureClass=PanelFigure)


# Tick widths (pt) of the axes panel_grid makes; FIGURE_RC's 0.5 / 0.4 pt apply elsewhere.
GRID_TICK_WIDTH = 0.6
# Decimal places kept of each panel box in figure fractions (see panel_grid).
GRID_FRACTION_DIGITS = 9
GRID_MINOR_TICK_WIDTH = 0.48


def grid_boxes(
    fig,
    layout,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    wspace=0.0,
    hspace=0.0,
    wratios=None,
    hratios=None,
) -> list[tuple[float, float, float, float]]:
    """Panel boxes, in figure fractions, of ``layout`` on a grid with margins and gaps in inches.

    The column widths share the figure width left after the margins and gaps in the
    proportions ``wratios``, and the row heights likewise with ``hratios``; a panel spans
    every cell that carries its number.

    Each box is rounded to ``GRID_FRACTION_DIGITS`` decimals of the figure (under 1e-5 px on
    a 10-inch side at 600 dpi), so adjustText placements are reproducible: adjustText turns
    a last-bit difference in a box into different label placements.

    Args:
        fig: The figure; its size fixes the grid.
        layout: Rows of positive panel numbers, one per cell; a number repeated over a
            rectangle of cells makes a spanning panel. Use ``[[1, 2], [3, 4]]`` for a plain 2 x 2.
        left, right, top, bottom: Margins between the figure edge and the grid, in inches.
        wspace, hspace: Gap between adjacent columns or rows, in inches: one value for every
            gap, or one per gap.
        wratios, hratios: Relative column widths and row heights (default all equal).

    Returns:
        ``(x0, y0, width, height)`` per panel, in panel-number order.
    """
    grid = np.asarray(layout)
    nrows, ncols = grid.shape
    fig_w, fig_h = fig.get_size_inches()
    wgaps = np.broadcast_to(np.asarray(wspace, dtype=float), (ncols - 1,))
    hgaps = np.broadcast_to(np.asarray(hspace, dtype=float), (nrows - 1,))
    wr = np.ones(ncols) if wratios is None else np.asarray(wratios, dtype=float)
    hr = np.ones(nrows) if hratios is None else np.asarray(hratios, dtype=float)
    widths = (fig_w - left - right - wgaps.sum()) * wr / wr.sum()
    heights = (fig_h - top - bottom - hgaps.sum()) * hr / hr.sum()
    x0 = left + np.concatenate([[0.0], np.cumsum(widths[:-1] + wgaps)])
    y_top = fig_h - top - np.concatenate([[0.0], np.cumsum(heights[:-1] + hgaps)])
    boxes = []
    for num in sorted(set(grid.ravel())):
        rows, cols = np.nonzero(grid == num)
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        x = x0[c0]
        w = x0[c1] + widths[c1] - x
        y = y_top[r1] - heights[r1]
        h = y_top[r0] - y
        box = (x / fig_w, y / fig_h, w / fig_w, h / fig_h)
        boxes.append(tuple(round(float(v), GRID_FRACTION_DIGITS) for v in box))
    return boxes


def panel_grid(fig, layout, **grid) -> list:
    """Add one axes per panel of ``layout`` at its ``grid_boxes`` box.

    Major ticks are ``GRID_TICK_WIDTH`` wide and minor ticks ``GRID_MINOR_TICK_WIDTH``.

    Args:
        fig: The figure.
        layout: Panel numbers per cell, as for ``grid_boxes``.
        **grid: Margins, gaps and ratios, as for ``grid_boxes``.

    Returns:
        The axes, in panel-number order.
    """
    axes = []
    for box in grid_boxes(fig, layout, **grid):
        ax = fig.add_axes(box)
        ax.tick_params(which="major", width=GRID_TICK_WIDTH)
        ax.tick_params(which="minor", width=GRID_MINOR_TICK_WIDTH)
        axes.append(ax)
    return axes
