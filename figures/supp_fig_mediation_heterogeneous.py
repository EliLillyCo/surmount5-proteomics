"""Supplementary Fig 10: week-72 mediation forest pairs for the ``SUPP2_THEME_ORDER`` themes.

The Neural theme spans the tall left column, and the other themes stack beside it.
"""

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
from analysis.mediation.themes import THEME_CANDIDATES
from figures._internal._mediation_figure import (
    LATE_WEEK,
    FOREST_GRID_BOTTOM_IN,
    GRID_LEFT_IN,
    GRID_RIGHT_IN,
    GRID_TOP_IN,
    PAIR_LETTER_DY_IN,
    add_forest_legend,
    figure_themes,
    load_mediation,
    plot_theme_pair,
)
from figures.supp_fig_mediation_weight_mediated import (
    N_MAX_DEFAULT,
    THEME_N_MAX,
    theme_label,
)

_HERE = Path(__file__).resolve().parent


SUPP2_THEME_ORDER = (
    "Vascular",
    "ECM & TGFβ\nremodeling",
    "Neural",
    "Iron\nhomeostasis",
    "Mast cell\nsignaling",
    "Pancreatic exocrine",
)

THEME_FOREST_KWARGS = {
    "ECM & TGFβ\nremodeling": {
        "label_line_offset_pt": 2.4,
    },
    "Neural": {
        "label_line_offset_pt": 2.6,
    },
}


def make_figure():
    """Build Supplementary Fig 10: week-72 mediation forests for the ``SUPP2_THEME_ORDER`` themes.

    Each theme is an Olink and SomaScan forest pair, its genes cut to those with a
    significant total effect (``figure_themes``); Neural spans the tall left column.

    Returns:
        The figure.
    """
    init_figure_theme()

    themes = figure_themes(THEME_CANDIDATES, SUPP2_THEME_ORDER, LATE_WEEK)
    med_ol = load_mediation("olink", LATE_WEEK)
    med_so = load_mediation("soma", LATE_WEEK)

    row1 = [1, 2, 3, 4]
    row2 = [5, 6, 7, 8]
    row3 = [5, 6, 9, 10]
    row4 = [5, 6, 11, 12]
    theme_panel_ids: dict[str, tuple[int, int]] = {
        "Vascular": (1, 2),
        "ECM & TGFβ\nremodeling": (3, 4),
        "Neural": (5, 6),
        "Iron\nhomeostasis": (7, 8),
        "Mast cell\nsignaling": (9, 10),
        "Pancreatic exocrine": (11, 12),
    }

    fig = panel_figure(*nature_figsize("2col", 9.72))
    axs = panel_grid(
        fig,
        [row1, row2, row3, row4],
        left=GRID_LEFT_IN,
        right=GRID_RIGHT_IN,
        top=GRID_TOP_IN,
        bottom=FOREST_GRID_BOTTOM_IN,
        wspace=0.70,
        hspace=(0.68, 0.70, 0.70),
        hratios=(1.42, 0.72, 0.72, 1.06),
    )
    pid_to_ax = {pid: ax for pid, ax in enumerate(axs, start=1)}

    for theme_name in SUPP2_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        panel_kwargs = THEME_FOREST_KWARGS.get(theme_name, {})
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            genes=themes[theme_name],
            med_ol=med_ol,
            med_so=med_so,
            n_max=THEME_N_MAX.get(theme_name, N_MAX_DEFAULT),
            **panel_kwargs,
        )

    add_forest_legend(fig)
    fig.canvas.draw()
    for theme_name in SUPP2_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        draw_banner(
            fig,
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            bold_text=theme_label(theme_name),
            left_edge_text="Olink",
            right_edge_text="SomaScan",
        )

    fig.canvas.draw()
    for letter, theme_name in zip("abcdef", SUPP2_THEME_ORDER, strict=False):
        panel_ol, _ = theme_panel_ids[theme_name]
        place_panel_letter(fig, pid_to_ax[panel_ol], letter, extra_dy_in=PAIR_LETTER_DY_IN)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = _HERE
    save_figure(fig, str(out_dir / "supp_fig_mediation_heterogeneous"), formats=("pdf",))
    print("Wrote supp_fig_mediation_heterogeneous.pdf")
