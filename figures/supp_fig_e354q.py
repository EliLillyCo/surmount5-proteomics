"""Supplementary figure: GIPR rs1800437 (E354Q) genotype results for PPY.

One 1×4 genotype boxplot row: residualized change from baseline by genotype × arm at
weeks 24 and 72, on Olink (left pair) and SomaScan (right pair), with the per-genotype
TZP vs. SEMA MMRM P values. The row, its banner and its labels are drawn by
``_internal._ppy_figure.draw_genotype_row``; a single panel carries no letter.

Genotype + MMRM data: produced by ``analysis/pharmacogenomics/run.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from analysis.design import ARMS, COVAR_PHRASE
from figures._internal._common import ARM_COLORS, ARM_LABELS, init_figure_theme, save_figure
import figures._internal._ppy_figure as F  # genotype row drawing

THIS_DIR = Path(__file__).resolve().parent


GENE = "PPY"
TAIL_TEXT = f" · Δ from baseline · adj. {COVAR_PHRASE}, genetic PC 1-3, baseline protein"

TOP_IN = 0.12
BOTTOM_IN = 0.12
FIG_H_IN = TOP_IN + F.GENO_BLOCK_IN + BOTTOM_IN


def make_figure():
    """Assemble the one-row genotype figure."""
    init_figure_theme()
    fig = plt.figure(figsize=(F.FIG_W, FIG_H_IN))
    F.draw_genotype_row(fig, GENE, fig_h_in=FIG_H_IN, top_in=TOP_IN, letter=None, tail_text=TAIL_TEXT)
    # Arm key, right-aligned on the row's x-axis-label line.
    plot_bot_in = TOP_IN + F.BANNER_HEIGHT_IN + F.GENO_BANNER_GAP_IN + F.GENO_PLOT_H_IN
    key_y = 1.0 - (plot_bot_in + F.GENO_AXIS_LABEL_DROP_IN) / FIG_H_IN
    handles = [
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS[arm],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS[arm],
        )
        for arm in tuple(ARMS)
    ]
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(F.GENO_RIGHT, key_y),
        ncol=2,
        fontsize=F.BOTTOM_LABEL_FONTSIZE,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    return fig


def render(save: bool = True) -> None:
    """Build Supplementary Fig 11 and, when ``save``, write ``figures/supp_fig_e354q.pdf``.

    Args:
        save: Write the PDF and close the figure; ``False`` only builds it.
    """
    fig = make_figure()
    if save:
        out_path = str(THIS_DIR / "supp_fig_e354q")
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved → {out_path}.pdf")
        plt.close()


if __name__ == "__main__":
    render()
