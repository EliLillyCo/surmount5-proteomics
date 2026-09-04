"""Supplementary mediation overview at Week 24."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import ultraplot as uplt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _internal._common import (  # noqa: E402
    draw_banner,
    init_figure_theme,
    nature_figsize,
    place_panel_letter,
    save_figure,
)
from _internal._mediation_figure import (  # noqa: E402
    PAIR_LETTER_DY_IN,
    PAIR_WSPACE,
    SCATTER_BANNER_TAIL,
    load_across_treatment,
    load_mediation,
    plot_global_scatter,
)


def make_figure():
    init_figure_theme()

    med_ol = load_mediation("olink")
    med_so = load_mediation("soma")
    ac_ol = load_across_treatment("olink", 24)
    ac_so = load_across_treatment("soma", 24)

    figwidth, figheight = nature_figsize("2col", 4.05)
    fig, axs = uplt.subplots(
        ncols=2,
        figwidth=figwidth,
        figheight=figheight,
        share=False,
        wspace=(PAIR_WSPACE,),
        left="0.58in",
        right="0.32in",
        top="0.44in",
        bottom="0.46in",
    )
    axs.format(abc=False)
    flat_axs = list(np.ravel(np.atleast_1d(axs)))
    ax_ol, ax_so = flat_axs

    plot_global_scatter(ax_ol, med_ol, ac_ol, 24, "olink", show_ylabel=True)
    plot_global_scatter(ax_so, med_so, ac_so, 24, "soma", show_ylabel=False)

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
    save_figure(fig, str(out_dir / "supp_fig12_mediation_week24"), formats=("pdf",))
    print("Wrote supp_fig12_mediation_week24.pdf")
