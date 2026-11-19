"""Mediation forests: three glyph rows per protein, Olink and SomaScan panels row-aligned.

Drawing: per
protein a total (navy), indirect / ACME (teal) and direct / ADE (near-black) glyph, each a
point estimate with its 95% bootstrap CI, drawn as an open gray glyph when that effect is
not significant (its BH FDR >= 0.05; the total is gated on ``total_fdr``). Classes and
FDRs are the ones ``analysis.mediation.results.load_platform`` returns; nothing here
recomputes them.

The figure pairs an Olink and a SomaScan forest per theme, so
rows are built across platforms (``aligned_theme_rows``): a theme's proteins whose total
effect clears FDR on a platform are selected there, clustered across platforms by shared
HGNC gene (a SomaScan heterodimer aptamer shares a row with the Olink assay of either
subunit), and a cluster selected on one platform shows the other platform's measurement
as a shadow row even when it does not clear the gate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib import transforms

from analysis.protein_labels import protein_label
from analysis.qc_contract import PLATFORMS
from analysis.statistics import FDR_THRESHOLD

from ._theme import (
    FS_AUX,
    FS_FOREST_YLABEL,
    GRAY_ZERO_LINE,
    LW_FOREST_CI,
    LW_FOREST_MARKER_EDGE,
    LW_GUIDE,
    LW_NONE,
)


COLOR_TOTAL = "#1f3d99"  # dark navy (total effect)
COLOR_INDIRECT = "#00a0a0"  # teal (indirect, weight-mediated ACME)
COLOR_DIRECT = "#1a1a1a"  # near-black (direct, ADE)
COLOR_NS_FOREST = "#c0c0c0"  # gray for non-significant
BAND_FILL = "#f5f5f5"  # alternating row band
LABEL_INK = "#222222"
LABEL_MUTED = "#999999"
LABEL_MISSING = "#BBBBBB"

# Row-height fraction between a protein's three stacked glyphs.
GLYPH_STEP = 0.28
# Space, in protein rows, between the outer glyph rows and the forest's edges.
FOREST_ROW_PAD = 0.6
# Size (points) of the arrowhead that ends a confidence interval cut at the axis edge.
EDGE_MARKER_PT = 2.6
# Gene label x (axes fraction) and the point offset between gene and marker lines.
LABEL_X = -0.016
LABEL_LINE_OFFSET_PT = 3.5

_EFFECTS = (
    # (estimate prefix, FDR column, color, marker area, glyph y offset in GLYPH_STEP units)
    ("total", "total_fdr", COLOR_TOTAL, 22, 1),
    ("ACME", "acme_fdr", COLOR_INDIRECT, 18, 0),
    ("ADE", "ade_fdr", COLOR_DIRECT, 18, -1),
)


def _significant(r: dict, fdr_col: str) -> bool:
    q = r.get(fdr_col)
    return q is not None and bool(np.isfinite(q)) and q < FDR_THRESHOLD


def _theme_subset(df: pl.DataFrame, genes: list[str]) -> pl.DataFrame:
    return df.filter(pl.col("genes").list.set_intersection(genes).list.len() > 0)


def select_platform(df: pl.DataFrame, genes: list[str], n_max: int) -> pl.DataFrame:
    """A theme's proteins on one platform whose total effect clears FDR.

    Args:
        df: One platform and visit from ``load_platform``, with ``genes`` and ``gene_label``.
        genes: The theme's HGNC symbols.
        n_max: Most proteins kept, the largest |total| first, one per gene label.

    Returns:
        The selected result rows.
    """
    sub = _theme_subset(df, genes).filter(pl.col("total_fdr") < FDR_THRESHOLD)
    return (
        sub.sort([pl.col("total_estimate").abs(), pl.col("marker")], descending=[True, False])
        .unique(subset=["gene_label"], keep="first", maintain_order=True)
        .head(n_max)
    )


def aligned_theme_rows(
    genes: list[str], by_platform: dict[str, pl.DataFrame], *, n_max: int
) -> list[dict]:
    """One row per gene cluster, shared by the Olink and SomaScan forests of a theme.

    Args:
        genes: The theme's HGNC symbols.
        by_platform: Platform -> that platform's classified results at one visit, with
            ``genes`` (list) and ``gene_label`` columns.
        n_max: Most proteins selected per platform.

    Returns:
        Rows from top to bottom (signed total of the stronger platform, descending), each
        ``{"display": the cluster's protein label (``analysis.protein_labels``), "olink": row | None,
        "soma": row | None}``; a
        platform's row is its selected protein, else its strongest measured protein of the
        cluster (a shadow row), else None.
    """
    clusters: list[dict] = []
    for platform in PLATFORMS:
        for r in select_platform(by_platform[platform], genes, n_max).iter_rows(named=True):
            g = set(r["genes"])
            match = next((c for c in clusters if c["genes"] & g), None)
            if match is None:
                clusters.append({"genes": g, "olink": None, "soma": None, "sort_val": r["total_estimate"]})
                match = clusters[-1]
            match["genes"] |= g
            current = match[platform]
            if current is None or abs(r["total_estimate"]) > abs(current["total_estimate"]):
                match[platform] = r
            if abs(r["total_estimate"]) > abs(match["sort_val"]):
                match["sort_val"] = r["total_estimate"]
    for platform in PLATFORMS:
        theme = _theme_subset(by_platform[platform], genes)
        for cluster in clusters:
            if cluster[platform] is not None:
                continue
            hits = [r for r in theme.iter_rows(named=True) if set(r["genes"]) & cluster["genes"]]
            if hits:
                cluster[platform] = max(hits, key=lambda r: (abs(r["total_estimate"]), r["marker"]))
    clusters.sort(key=lambda c: (-c["sort_val"], "|".join(sorted(c["genes"]))))
    return [
        {"display": protein_label(c["genes"]), "olink": c["olink"], "soma": c["soma"]} for c in clusters
    ]


def forest_xlim(rows: list[dict]) -> tuple[float, float]:
    """Shared x-range for a theme's two platform forests (``forest_xlim``).

    The range spans the estimates, extended to the confidence limits but no further than a
    cap set by the estimate span and the median whisker, padded 6% and always straddling
    zero; (-1, 1) when there is nothing to span.
    """
    ests, ci_lo, ci_hi, whiskers = [], [], [], []
    for entry in rows:
        for platform in PLATFORMS:
            r = entry.get(platform)
            if r is None:
                continue
            for prefix, *_ in _EFFECTS:
                e, lo, hi = (r.get(f"{prefix}_{k}") for k in ("estimate", "ci_lower", "ci_upper"))
                if e is None or not np.isfinite(e):
                    continue
                ests.append(float(e))
                ci_lo.append(float(lo))
                ci_hi.append(float(hi))
                whiskers += [abs(float(e) - float(lo)), abs(float(hi) - float(e))]
    if not ests:
        return (-1.0, 1.0)
    e_lo, e_hi = min(ests), max(ests)
    cap = max(0.25 * max(e_hi - e_lo, 1e-6), 2.5 * float(np.median(whiskers)))
    lo = max(min(ci_lo), e_lo - cap)
    hi = min(max(ci_hi), e_hi + cap)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    if lo > 0:
        lo = -pad
    elif hi < 0:
        hi = pad
    return (lo, hi)


def draw_forest(
    ax,
    rows: list[dict],
    platform: str,
    *,
    xlim: tuple[float, float],
    show_xlabel: bool = True,
    xlabel: str = "Effect (95% CI)",
    glyph_step: float = GLYPH_STEP,
    label_line_offset_pt: float = LABEL_LINE_OFFSET_PT,
) -> None:
    """One platform's forest: total, indirect and direct glyph rows per protein.

    A confidence interval that extends beyond ``xlim`` ends in an arrowhead at the axis edge.

    Args:
        ax: Axes for this platform's forest.
        rows: From ``aligned_theme_rows``; row ``i`` is drawn at ``y = n - 1 - i``.
        platform: ``"olink"`` or ``"soma"``, selecting each row's result.
        xlim: Shared x limits from ``forest_xlim``.
        show_xlabel: Whether to label the x axis.
        xlabel: The x-axis label.
        glyph_step: Row-height fraction between the three glyphs.
        label_line_offset_pt: Points between the gene line and the marker-ID line of a label.
    """
    from matplotlib.ticker import NullLocator

    n = len(rows)
    for i in range(0, n, 2):
        y = n - 1 - i
        ax.axhspan(y - 0.5, y + 0.5, facecolor=BAND_FILL, edgecolor="none", zorder=0)
    ax.axvline(0, color=GRAY_ZERO_LINE, lw=LW_GUIDE, zorder=1)

    for i, entry in enumerate(rows):
        r = entry.get(platform)
        if r is None:
            continue
        y0 = n - 1 - i
        for prefix, fdr_col, color, size, step in _EFFECTS:
            sig = _significant(r, fdr_col)
            y = y0 + step * glyph_step
            line = color if sig else COLOR_NS_FOREST
            ax.plot(
                [r[f"{prefix}_ci_lower"], r[f"{prefix}_ci_upper"]],
                [y, y],
                color=line,
                lw=LW_FOREST_CI,
                zorder=2,
                solid_capstyle="round",
            )
            # An interval that runs past the shared axis ends in an arrowhead at the axis edge.
            for limit, bound, marker in (
                (xlim[0], r[f"{prefix}_ci_lower"], "<"),
                (xlim[1], r[f"{prefix}_ci_upper"], ">"),
            ):
                if (bound < limit) if marker == "<" else (bound > limit):
                    ax.plot(
                        [limit],
                        [y],
                        marker=marker,
                        ms=EDGE_MARKER_PT,
                        mfc=line,
                        mec="none",
                        ls="none",
                        zorder=4,
                        clip_on=False,
                    )
            ax.scatter(
                [r[f"{prefix}_estimate"]],
                [y],
                s=size,
                facecolors=color if sig else "white",
                edgecolors=line,
                linewidths=LW_NONE if sig else LW_FOREST_MARKER_EDGE,
                zorder=3,
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels([""] * n)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(-FOREST_ROW_PAD, n - 1 + FOREST_ROW_PAD)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel if show_xlabel else "", fontsize=FS_AUX)
    _draw_row_labels(ax, rows, platform, label_line_offset_pt)


def _draw_row_labels(ax, rows: list[dict], platform: str, offset_pt: float) -> None:
    """Gene label over the platform's marker ID; a cluster the platform lacks is gray italic."""
    n = len(rows)
    base = ax.get_yaxis_transform()
    up = transforms.offset_copy(base, fig=ax.figure, x=0, y=offset_pt, units="points")
    down = transforms.offset_copy(base, fig=ax.figure, x=0, y=-offset_pt, units="points")
    for i, entry in enumerate(rows):
        y = n - 1 - i
        r = entry.get(platform)
        common = dict(ha="right", va="center", clip_on=False)
        if r is None:
            ax.text(
                LABEL_X, y, entry["display"], transform=base, fontsize=FS_FOREST_YLABEL,
                color=LABEL_MISSING, style="italic", **common,
            )
            continue
        ax.text(LABEL_X, y, r["gene_label"], transform=up, fontsize=FS_FOREST_YLABEL, color=LABEL_INK, **common)
        ax.text(
            LABEL_X, y, r["marker"], transform=down, fontsize=FS_FOREST_YLABEL - 1.0, color=LABEL_MUTED, **common
        )


def forest_legend_handles() -> list:
    """Legend glyphs: total, indirect, direct, non-significant."""
    import matplotlib.lines as mlines

    def _dot(face, edge, ew, ms=4.2):
        return mlines.Line2D([], [], marker="o", ls="", ms=ms, mfc=face, mec=edge, mew=ew)

    return [
        _dot(COLOR_TOTAL, COLOR_TOTAL, 0.0),
        _dot(COLOR_INDIRECT, COLOR_INDIRECT, 0.0),
        _dot(COLOR_DIRECT, COLOR_DIRECT, 0.0),
        _dot("white", COLOR_NS_FOREST, 0.6, ms=3.8),
    ]


FOREST_LEGEND_LABELS = (
    "Total effect",
    "Indirect (weight-mediated)",
    "Direct (weight-independent)",
    "ns",
)
