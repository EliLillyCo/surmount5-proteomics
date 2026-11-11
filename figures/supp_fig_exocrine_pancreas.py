"""Supplemental figure: exocrine pancreas enzyme trajectories.

Per-marker %CFB line plots (Wk0/24/72) for TZP and SEMA, selecting exocrine
pancreas markers by between-treatment FDR at Wk24 or Wk72 and the configured
additional genes.
Every selected marker is shown, split across two figures by rank (decreasing max
|log2FC|, per platform): ``--part lead`` is the Extended Data figure (four columns,
``ED_OLINK_ROWS`` Olink rows and one SomaScan row, which fits the 180 x 170 mm box) and
``--part rest`` the remainder, a Supplementary figure with the same panel size.

Font sizes, marker sizes, line widths, and trajectory banner styling are
matched to the Fig 1 trajectory-line panel (`_trajectory_lines.py`).
Platform banner height matches the Fig 2 volcano banner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from figures._internal._common import (
    init_figure_theme,
    load_marker_to_gene,
    marker_labels,
    save_figure,
)
import figures._internal._trajectory_data as td
import figures._internal._trajectory_lines as tl
from figures._internal._trajectory_lines import MarkerSpec
from analysis.design import EARLY_WEEK, LATE_WEEK
from analysis.statistics import FDR_THRESHOLD
from analysis.trajectory_panels import EXOCRINE_GENES

THIS_DIR = Path(__file__).resolve().parent


# Genes drawn in addition to those selected by the between-treatment FDR.
ADDITIONAL_GENES: dict[str, frozenset[str]] = {
    "olink": frozenset({"CTRL"}),
    "soma": frozenset({"CTRB1", "CPA1"}),
}


# ---------------------------------------------------------------------------
# Marker selection
# ---------------------------------------------------------------------------
def _select_markers(platform: str) -> pl.DataFrame:
    """Select exocrine genes by between-treatment FDR at either post-baseline week
    or membership in ``ADDITIONAL_GENES``, sorted by max |log2FC| descending.
    For Soma, keep one SeqId per gene."""
    mtg = load_marker_to_gene(platform)
    exo_map = {
        m: g for m, g in mtg.items() if g in EXOCRINE_GENES
    }
    include_genes = ADDITIONAL_GENES.get(platform, frozenset())

    act_early = td.load_across_treatment(platform, EARLY_WEEK).rename({"FC": "FC_early", "fdr": "fdr_early"})
    act_late = td.load_across_treatment(platform, LATE_WEEK).rename({"FC": "FC_late", "fdr": "fdr_late"})

    merged = (
        act_early.join(act_late, on="marker", how="inner")
        .filter(pl.col("marker").is_in(list(exo_map)))
        .with_columns(
            pl.col("marker").replace_strict(exo_map).alias("gene"),
            pl.col("marker").replace_strict(marker_labels(platform), default=pl.col("marker")).alias("label"),
            pl.max_horizontal(
                pl.col("FC_early").log(2).abs(),
                pl.col("FC_late").log(2).abs(),
            ).alias("max_abs_log2fc"),
        )
        .filter(
            (pl.col("fdr_early") < FDR_THRESHOLD)
            | (pl.col("fdr_late") < FDR_THRESHOLD)
            | pl.col("gene").is_in(list(include_genes))
        )
    )

    if platform == "soma":
        merged = merged.sort("max_abs_log2fc", descending=True).unique(
            subset=["gene"], keep="first"
        )

    return merged.sort("max_abs_log2fc", descending=True)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
# Extended Data grid: four columns, two Olink rows plus one SomaScan row (three rows of
# panels), the most that fits the 170 mm box with the banners and the legend strip.
COLS = tl.STACK_COLS
ED_OLINK_ROWS = 2
ED_SOMA_ROWS = 1
# Per-platform caps on the ranked markers.
MAX_OLINK, MAX_SOMA = 11, 5
PARTS = ("lead", "rest")


def split_markers(
    olink_sel: pl.DataFrame, soma_sel: pl.DataFrame, part: str
) -> tuple[list[MarkerSpec], list[MarkerSpec]]:
    """Rank-split each platform's markers into the Extended Data and supplementary figures.

    Args:
        olink_sel: Selected Olink markers, ranked (``_select_markers``).
        soma_sel: Selected SomaScan markers, ranked.
        part: ``"lead"`` for the Extended Data figure, ``"rest"`` for the supplementary one.

    Returns:
        ``(olink_panels, soma_panels)`` for the chosen part.
    """
    n_olink, n_soma = ED_OLINK_ROWS * COLS, ED_SOMA_ROWS * COLS
    for platform, selected, cap in (("olink", olink_sel, MAX_OLINK), ("soma", soma_sel, MAX_SOMA)):
        if selected.height > cap:
            raise ValueError(
                f"{platform}: {selected.height} selected markers exceed the cap of {cap}; the figures "
                "show every selected marker, so raise the cap and the panel grid"
            )
    olink = [MarkerSpec(m, g) for m, g in olink_sel.select("marker", "label").iter_rows()]
    soma = [MarkerSpec(m, g) for m, g in soma_sel.select("marker", "label").iter_rows()]
    if part == "lead":
        return olink[:n_olink], soma[:n_soma]
    return olink[n_olink:], soma[n_soma:]


def render(part: str = "lead") -> None:
    """Build one part of the exocrine pancreas trajectories and write ``figures/supp_fig_exocrine_pancreas.pdf``.

    Both parts write the same stem; the figure bundle runs each with its own ``--part``.

    Args:
        part: ``"lead"`` for Extended Data Fig 4 (the top-ranked markers) or ``"rest"`` for
            Supplementary Fig 4 (the remainder).
    """
    init_figure_theme()

    olink_panels, soma_panels = split_markers(_select_markers("olink"), _select_markers("soma"), part)
    fig = tl.stacked_platform_figure(olink_panels, soma_panels)

    out_path = str(THIS_DIR / "supp_fig_exocrine_pancreas")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=PARTS, default=PARTS[0])
    render(parser.parse_args().part)
