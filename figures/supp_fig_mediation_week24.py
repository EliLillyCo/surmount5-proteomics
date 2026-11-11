"""Supplementary mediation overview at Week 24."""

from __future__ import annotations

from pathlib import Path

from figures._internal._common import (
    draw_banner,
    init_figure_theme,
    nature_figsize,
    panel_figure,
    panel_grid,
    place_panel_letter,
    save_figure,
)
from analysis.design import EARLY_WEEK
from figures._internal._mediation_figure import (
    GRID_LEFT_IN,
    GRID_RIGHT_IN,
    GRID_TOP_IN,
    PAIR_LETTER_DY_IN,
    PAIR_WSPACE,
    SCATTER_BANNER_TAIL,
    load_mediation,
    plot_global_scatter,
)

_HERE = Path(__file__).resolve().parent


# The earliest post-baseline proteomics week.
WEEK = EARLY_WEEK


def make_figure():
    """Build Extended Data Fig 6: week-24 total-vs-direct mediation scatters, Olink (a) and SomaScan (b).

    Returns:
        The figure.
    """
    init_figure_theme()

    med_ol = load_mediation("olink", WEEK)
    med_so = load_mediation("soma", WEEK)

    fig = panel_figure(*nature_figsize("2col", 4.05))
    ax_ol, ax_so = panel_grid(
        fig,
        [[1, 2]],
        left=GRID_LEFT_IN,
        right=GRID_RIGHT_IN,
        top=GRID_TOP_IN,
        bottom=0.46,
        wspace=PAIR_WSPACE,
    )

    plot_global_scatter(ax_ol, med_ol, WEEK, "olink", show_ylabel=True)
    plot_global_scatter(ax_so, med_so, WEEK, "soma", show_ylabel=False)

    fig.canvas.draw()
    draw_banner(
        fig,
        ax_ol,
        ax_ol,
        bold_text="Olink",
        tail_text=SCATTER_BANNER_TAIL,
    )
    draw_banner(
        fig,
        ax_so,
        ax_so,
        bold_text="SomaScan",
        tail_text=SCATTER_BANNER_TAIL,
    )

    fig.canvas.draw()
    place_panel_letter(fig, ax_ol, "a", extra_dy_in=PAIR_LETTER_DY_IN)
    place_panel_letter(fig, ax_so, "b", extra_dy_in=PAIR_LETTER_DY_IN)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = _HERE
    save_figure(fig, str(out_dir / "supp_fig_mediation_week24"), formats=("pdf",))
    print("Wrote supp_fig_mediation_week24.pdf")
