"""GIP-receiving cell-type panel — helper for ``fig5_ppy.py`` (panel a).

Loads and draws the ranked "hormone receiving strength by cell type" bar chart
from the Hormone Cell Atlas (Fei et al., Science 2026), supplementary Table S6A
("Hormone receiving strengths at the fine-grained cell-type level across all 47
tissues"), replotted as a clean horizontal bar chart with bold tissue labels and
value annotations.

Provides the reusable pieces used by the main figure:
``load_hormone_strengths``, ``draw_gip_panel`` and ``ATLAS_DATA``.

Data source
-----------
data/external/science_aeb2672_tableS6A.parquet — the columns
(Hormone_short, Tissue, fine-grained cell type, Strength) extracted from sheet
"Table S6A" of the Hormone Cell Atlas supplement. Rows are already
one-per-(tissue, fine-grained cell type) with assay deduplication (max across
scRNA-/snRNA-seq) applied upstream, so no assay collapsing is needed.
"""

from __future__ import annotations

import re
import matplotlib as mpl
import numpy as np
import polars as pl

from ._common import (
    FS_BODY,
    ROOT,
    build_cmap,
    c,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATLAS_DATA = ROOT / "data" / "external" / "science_aeb2672_tableS6A.parquet"
ATLAS_CITATION = "Hormone Cell Atlas (Fei et al., Science 2026), Table S6A"

# Subtitle/caption grey: brand bold_grey darkened for legibility on white.
SUBTITLE_GREY = mpl.colors.to_hex([x * 0.62 for x in mpl.colors.to_rgb(c["bold_grey"])])

CELLTYPE_COL = "Tissue level fine-grained cell type (celltype_level2)"

# Display label for the title; falls back to upper-case for unmapped hormones.
HORMONE_LABELS: dict[str, str] = {
    "gip": "GIP",
    "glp1": "GLP-1",
    "amylin_iapp": "Amylin (IAPP)",
    "pomc_beta_endorphin": "β-endorphin (POMC)",
}

# Proper nouns to restore casing for after lower-cased atlas tokens.
_PROPER_NOUNS = {"henle": "Henle"}

# Irregular atlas names with no "cell" head-token that the generic
# parenthesization rule cannot resolve; mapped explicitly. Extend as needed
# when reusing for other hormones.
_DISPLAY_OVERRIDES = {
    "loop_of_henle_ascending_thin_limb": "loop of Henle (ascending thin limb)",
}


def _hormone_label(hormone: str) -> str:
    return HORMONE_LABELS.get(hormone, hormone.upper())


def _prettify_celltype(name: str) -> str:
    """Turn an atlas celltype_level2 string into a readable display label.

    underscores -> spaces; restore proper-noun casing; parenthesize the
    qualifier trailing the head noun "cell"
    (e.g. ``ciliated_columnar_cell_tracheobronchial`` ->
    "ciliated columnar cell (tracheobronchial)").
    """
    if name in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[name]
    tokens = [_PROPER_NOUNS.get(t.lower(), t) for t in name.split("_")]
    text = " ".join(tokens)
    m = re.match(r"^(.*\bcell\b)\s+(.+)$", text)
    if m:
        text = f"{m.group(1)} ({m.group(2)})"
    return text


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_hormone_strengths(
    hormone: str = "gip",
    top_n: int = 20,
    path: Path = ATLAS_DATA,
) -> pl.DataFrame:
    """Top-N (tissue, fine-grained cell type) entries by receiving strength.

    Cleaning, in order:
      1. strip trailing ``_<number>`` subcluster index from the cell-type name
      2. fix the source typo "Nasal Mocusa" -> "Nasal Mucosa"
      3. collapse subclusters: max Strength within each (tissue, cleaned name)
      4. sort by Strength descending, keep top_n

    Returns columns [Tissue, celltype, celltype_display, Strength].
    """
    df = pl.read_parquet(path)

    gip = (
        df.filter(pl.col("Hormone_short") == hormone)
        .with_columns(
            pl.col("Strength").cast(pl.Float64),
            pl.col("Tissue").str.replace("Nasal Mocusa", "Nasal Mucosa"),
            pl.col(CELLTYPE_COL).str.replace(r"_\d+$", "").alias("celltype"),
        )
        .group_by(["Tissue", "celltype"])
        .agg(pl.col("Strength").max())
        .sort("Strength", descending=True)
        .head(top_n)
    )
    return gip.with_columns(
        pl.col("celltype")
        .map_elements(_prettify_celltype, return_dtype=pl.String)
        .alias("celltype_display")
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw_gip_panel(
    ax, data, *, hormone: str = "gip", subtitle: bool = True, show_colorbar: bool = True
):
    """Draw the ranked hormone-receiving cell-type bars into an existing axis.

    `data` is the output of load_hormone_strengths (already top-N, sorted). Uses
    only matplotlib axis methods (no ultraplot-specific calls, except the
    optional colorbar) so it works on plain matplotlib axes. Returns `data`.
    """
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Inter"
    mpl.rcParams["mathtext.bf"] = "Inter:bold"
    mpl.rcParams["mathtext.it"] = "Inter:italic"

    plot_df = data.reverse()
    vals = plot_df["Strength"].to_numpy()
    tissues = plot_df["Tissue"].to_list()
    celltypes = plot_df["celltype_display"].to_list()
    n = len(vals)
    y = np.arange(n)

    cmap = build_cmap("Lilly_Blues")
    xmax = float(vals.max())
    norm = mpl.colors.Normalize(vmin=0.0, vmax=xmax)
    colors = cmap(norm(vals))

    ax.grid(True, axis="x", color=c["bold_grey"], alpha=0.22, lw=0.4, zorder=0)
    # matplotlib barh: y positions, bar lengths via `width`, thickness via `height`.
    ax.barh(
        y,
        vals,
        height=0.74,
        color=colors,
        edgecolor=c["black"],
        linewidth=0.3,
        zorder=2,
    )
    for yi, v in zip(y, vals):
        ax.text(
            v + xmax * 0.012,
            yi,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=FS_BODY,
            color=c["black"],
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)
    # Two-line y labels rendered as SEPARATE bold-tissue / cell-type texts at
    # fixed offsets — not one mathtext label. A mathtext bold line's descent
    # varies with its glyphs (e.g. the 'p' in "Adipose", 'g' in "Lung"), which
    # made the gap to the cell-type line inconsistent across rows; fixed offsets
    # give uniform spacing.
    from matplotlib.transforms import offset_copy

    _fs = mpl.rcParams["ytick.labelsize"]
    _base = ax.get_yaxis_transform()  # x: axes fraction, y: data
    _up = offset_copy(_base, fig=ax.figure, x=-2, y=0.6, units="points")
    _dn = offset_copy(_base, fig=ax.figure, x=-2, y=-0.6, units="points")
    for _yi, _tis, _ct in zip(y, tissues, celltypes):
        ax.text(
            0,
            _yi,
            _tis,
            transform=_up,
            ha="right",
            va="bottom",
            fontsize=_fs,
            fontweight="bold",
            color=c["black"],
            clip_on=False,
        )
        ax.text(
            0,
            _yi,
            _ct,
            transform=_dn,
            ha="right",
            va="top",
            fontsize=_fs,
            color=c["black"],
            clip_on=False,
        )
    ax.set_xlabel("Receiving strength")
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    # The shared figure theme enables minor ticks globally; this panel wants
    # none. The only y "ticks" are the per-bar label anchors (length 0), and
    # the x-axis keeps just its major ticks — so no tick mark extends past the
    # bars (above the top bar, below the bottom bar, or right of the last bar).
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="both", which="minor", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    if show_colorbar:
        # ultraplot colorbar; the main figure passes show_colorbar=False since a
        # colorbar would clutter the dense layout and the bars are value-labelled.
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.colorbar(
            sm,
            loc="r",
            width=0.1,
            length=0.55,
            label="Strength",
            ticks=[0, round(xmax, 1)],
        )

    if subtitle:
        ax.set_title(
            f"Predicted {_hormone_label(hormone)}-receiving cell types"
            f"  ({ATLAS_CITATION})",
            fontsize=FS_BODY,
            color=SUBTITLE_GREY,
            pad=6,
        )
    return data
