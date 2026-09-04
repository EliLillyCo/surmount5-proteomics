"""Supplementary mediation figure 2.

Heterogeneous and weight-independent themes. Neural occupies a tall spanning
panel so the long protein list remains readable, while the smaller themes stack
beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import ultraplot as uplt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _internal._common import (  # noqa: E402
    BANNER_GAP_IN,
    draw_banner,
    init_figure_theme,
    nature_figsize,
    place_panel_letter,
    save_figure,
)
from _internal._mediation_figure import (  # noqa: E402
    PAIR_LETTER_DY_IN,
    add_forest_legend,
    classify_mediation,
    load_across_treatment,
    load_mediation,
    plot_theme_pair,
)
from supp_fig13_mediation_weight_mediated import (  # noqa: E402
    N_MAX_DEFAULT,
    THEME_N_MAX,
    theme_genes,
    theme_label,
)

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
        "pm_header_y": 1.0,
        "pm_header_va": "bottom",
        "pm_header_offset_in": BANNER_GAP_IN,
        "label_line_offset_pt": 2.6,
    },
}


def make_figure():
    init_figure_theme()

    med_ol = load_mediation("olink")
    med_so = load_mediation("soma")
    ac_ol = pl.concat(
        [
            load_across_treatment("olink", 24),
            load_across_treatment("olink", 72),
        ]
    )
    ac_so = pl.concat(
        [
            load_across_treatment("soma", 24),
            load_across_treatment("soma", 72),
        ]
    )
    full_ol_72 = classify_mediation(
        med_ol.filter(pl.col("visit") == 20),
        ac_df=ac_ol,
        week=72,
    )
    full_so_72 = classify_mediation(
        med_so.filter(pl.col("visit") == 20),
        ac_df=ac_so,
        week=72,
    )

    row1 = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    row2 = [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3
    row3 = [5] * 3 + [6] * 3 + [9] * 3 + [10] * 3
    row4 = [5] * 3 + [6] * 3 + [11] * 3 + [12] * 3
    theme_panel_ids: dict[str, tuple[int, int]] = {
        "Vascular": (1, 2),
        "ECM & TGFβ\nremodeling": (3, 4),
        "Neural": (5, 6),
        "Iron\nhomeostasis": (7, 8),
        "Mast cell\nsignaling": (9, 10),
        "Pancreatic exocrine": (11, 12),
    }

    figwidth, figheight = nature_figsize("2col", 9.72)
    fig, axs = uplt.subplots(
        array=[row1, row2, row3, row4],
        figwidth=figwidth,
        figheight=figheight,
        share=False,
        hratios=(1.42, 0.72, 0.72, 1.06),
        hspace=("0.68in", "0.70in", "0.70in"),
        wspace=("0.70in",),
        left="0.58in",
        right="0.32in",
        top="0.44in",
        bottom="0.82in",
    )
    axs.format(abc=False)
    flat_axs = list(np.ravel(np.atleast_1d(axs)))
    pid_to_ax = {pid: ax for pid, ax in enumerate(flat_axs, start=1)}

    for theme_name in SUPP2_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        panel_kwargs = THEME_FOREST_KWARGS.get(theme_name, {})
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            theme_name=theme_name,
            genes=theme_genes(theme_name),
            med_ol=med_ol,
            med_so=med_so,
            ac_olink_all=ac_ol,
            ac_soma_all=ac_so,
            full_ol_72=full_ol_72,
            full_so_72=full_so_72,
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
    save_figure(fig, str(out_dir / "supp_fig14_mediation_heterogeneous"), formats=("pdf",))
    print("Wrote supp_fig14_mediation_heterogeneous.pdf")
