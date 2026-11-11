"""Supplementary Fig 9: week-72 mediation forest pairs for the ``SUPP1_THEME_ORDER`` themes.

In the lower section, the coagulation and complement panel sits beside two stacked panels.
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
from analysis.mediation.themes import NEURAL_THEME_NAME, THEME_CANDIDATES
from figures._internal._mediation_figure import (
    LATE_WEEK,
    FOREST_GRID_BOTTOM_IN,
    GRID_LEFT_IN,
    GRID_RIGHT_IN,
    GRID_TOP_IN,
    MAIN_THEME_LABELS,
    PAIR_LETTER_DY_IN,
    PAIR_WSPACE,
    add_forest_legend,
    figure_themes,
    load_mediation,
    plot_theme_pair,
)

_HERE = Path(__file__).resolve().parent


# Per-theme row caps.
N_MAX_DEFAULT = 7
THEME_N_MAX: dict[str, int] = {
    NEURAL_THEME_NAME: 18,
}

SUPP1_THEME_ORDER = (
    "Coagulation\n& complement",
    "Hepatic\nacute-phase",
    "Steroid hormone\nmetabolism",
    "GH–IGF axis",
    "Apolipoproteins & lipid handling",
)


def theme_label(theme_name: str) -> str:
    """The theme's banner text: its Fig. 4 label when it has one, line breaks replaced by spaces."""
    if theme_name in MAIN_THEME_LABELS:
        return MAIN_THEME_LABELS[theme_name].replace("\n", " ")
    return theme_name.replace("\n", " ")


def make_figure():
    """Build Supplementary Fig 9: week-72 mediation forests for the ``SUPP1_THEME_ORDER`` themes.

    Each theme is an Olink and SomaScan forest pair, its genes cut to those with a
    significant total effect (``figure_themes``).

    Returns:
        The figure.
    """
    init_figure_theme()

    themes = figure_themes(THEME_CANDIDATES, SUPP1_THEME_ORDER, LATE_WEEK)
    med_ol = load_mediation("olink", LATE_WEEK)
    med_so = load_mediation("soma", LATE_WEEK)

    row1 = [1, 2, 3, 4]
    row2 = [1, 2, 5, 6]
    row3 = [7, 8, 9, 10]
    theme_panel_ids: dict[str, tuple[int, int]] = {
        "Coagulation\n& complement": (1, 2),
        "Hepatic\nacute-phase": (3, 4),
        "Steroid hormone\nmetabolism": (5, 6),
        "GH–IGF axis": (7, 8),
        "Apolipoproteins & lipid handling": (9, 10),
    }

    fig = panel_figure(*nature_figsize("2col", 7.3))
    axs = panel_grid(
        fig,
        [row1, row2, row3],
        left=GRID_LEFT_IN,
        right=GRID_RIGHT_IN,
        top=GRID_TOP_IN,
        bottom=FOREST_GRID_BOTTOM_IN,
        wspace=PAIR_WSPACE,
        hspace=(0.56, 0.72),
        hratios=(0.95, 0.95, 1.35),
    )
    pid_to_ax = {pid: ax for pid, ax in enumerate(axs, start=1)}

    for theme_name in SUPP1_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            genes=themes[theme_name],
            med_ol=med_ol,
            med_so=med_so,
            n_max=THEME_N_MAX.get(theme_name, N_MAX_DEFAULT),
        )

    add_forest_legend(fig)
    fig.canvas.draw()
    for theme_name in SUPP1_THEME_ORDER:
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
    for letter, theme_name in zip("abcde", SUPP1_THEME_ORDER, strict=False):
        panel_ol, _ = theme_panel_ids[theme_name]
        place_panel_letter(fig, pid_to_ax[panel_ol], letter, extra_dy_in=PAIR_LETTER_DY_IN)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = _HERE
    save_figure(fig, str(out_dir / "supp_fig_mediation_weight_mediated"), formats=("pdf",))
    print("Wrote supp_fig_mediation_weight_mediated.pdf")
