"""Mediation between-treatment figure: TZP 15 mg vs SEMA 2.4 mg.

Row 1: Olink and SomaScan global total-vs-direct scatters at Wk72.
Rows 2-3: paired Olink + SomaScan forest vignettes for the configured themes
(``MAIN_THEME_ROWS``).

Notes
-----
- Values beyond the computed display caps (``detect_cap``) are drawn as edge markers.
- Markers → HGNC via uniprot_map (build_marker_genes from _volcano_figure.py).
- Mediation results, FDRs and the six-way class (``mclass``) come from
  ``analysis.mediation.results.load_platform``: unconditional BH per platform and
  visit on the analytic Sobel/Wald p-values, the class gated on ``total_fdr``. The
  scatters show proteins whose mediation total effect clears FDR < 0.05; this module computes no FDR or class of its own.
- Forest vignettes use ``_mediation_forest.draw_forest`` (total / indirect / direct
  glyph rows, open gray when not significant), Olink and SomaScan row-aligned.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from analysis.mediation.results import load_platform
from analysis.mediation.themes import MAIN_THEME_CANDIDATES, significant_themes
from analysis.design import LATE_WEEK, VISIT_LABELS

from . import _mediation_forest as mf
from ._common import (
    ARM_COLORS,
    FS_AUX,
    BANNER_GAP_IN,
    BANNER_HEIGHT_IN,
    COLOR_NS,
    FS_AXLABEL,
    FS_BODY,
    FS_DIRECTION,
    FS_GENE_LABEL,
    FS_PILL,
    FS_TICK,
    c,
    build_marker_genes,
    marker_labels,
    deterministic_adjust_text,
    draw_banner,
    grid_boxes,
    init_figure_theme,
    nature_figsize,
    panel_figure,
    panel_grid,
    place_panel_letter,
    separate_overlapping_labels,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISIT_TO_WEEK = dict(VISIT_LABELS)  # VISITNUM -> week (platform_inputs)
WEEK_TO_VISIT = {v: k for k, v in VISIT_TO_WEEK.items()}

FDR_THRESHOLD = mf.FDR_THRESHOLD  # mediation total-effect gate and per-effect significance

# Display labels of the curated themes the figures draw (Fig. 4 and Supp Figs 9 and 10).
# The gene lists are ``analysis.mediation.themes.*_CANDIDATES``.
MAIN_THEME_LABELS = {
    # Row 1 — adipose / endocrine / digestive
    "Leptin & adiponectin": "Leptin &\nadiponectin",
    "GH–IGF axis": "GH–IGF axis",
    "Pancreatic exocrine": "Pancreatic\nexocrine",
    "Islet hormones": "Islet hormones",
    # Row 2 — tissue signatures / lipid transport
    "Activin & myogenesis": "Activin &\nmyogenesis",
    "Intestinal epithelium": "Intestinal\nepithelium",
    "Vascular": "Vascular",
    "Apolipoproteins & lipid handling": "Apolipoproteins\n& lipid handling",
}

MAIN_THEME_ROWS = (
    ("Leptin & adiponectin", "Activin & myogenesis"),
    ("Islet hormones", "Intestinal epithelium"),
)
MAIN_THEME_ORDER = tuple(name for row in MAIN_THEME_ROWS for name in row)

def figure_themes(
    candidates: dict[str, list[str]], order: tuple[str, ...], week: int
) -> dict[str, list[str]]:
    """The figure's themes: each candidate list cut to genes with a significant total effect.

    A gene stays when its mediation total effect has FDR < 0.05 on at least one platform at
    ``week`` (``analysis.mediation.themes.significant_themes``). A theme left with no gene
    is dropped, with a note on stdout.

    Args:
        candidates: Theme -> full curated gene list.
        order: The figure's themes in layout order; the layouts have a fixed slot per
            theme, so a theme of ``order`` that is dropped is an error, not a gap.
        week: Week of the figure's results.

    Returns:
        Theme -> significant genes, for every theme of ``candidates`` that keeps one.
    """
    themes = significant_themes(
        {name: candidates[name] for name in order}, WEEK_TO_VISIT[week], build_marker_genes()
    )
    for name in order:
        if name not in themes:
            print(f"Note: theme {name!r} has no gene with total-effect FDR < {FDR_THRESHOLD} at week {week}; dropped.")
    missing = [name for name in order if name not in themes]
    if missing:
        raise RuntimeError(f"The layout has a fixed slot for {missing}, which have no significant gene.")
    for name in order:
        print(f"  {name!r}: {len(candidates[name])} candidates -> {len(themes[name])} significant genes")
    return themes

N_PROTEINS_PER_THEME = 7  # most forest rows per theme and platform, every theme

# ---- Scatter label budget (per direction; doubled across up/down) ----------
# `N_VOLCANO_PER_DIR` is the dominant knob — headline differential genes ranked
# by |total| × −log10(ac_fdr). The per-class supplements guarantee each
# mediation class keeps representation even when its members miss the volcano
# cutoffs. Lowering these reduces scatter clutter; the union is de-duplicated so
# the realized label count is below the raw sum.
N_VOLCANO_PER_DIR = 12
N_SUPPRESSION_PER_DIR = 4
N_DIRECT_PER_DIR = 3
N_MEDIATED_PER_DIR = 3


# Colors (resolved at theme time).
# Scatter dots: hue encodes direction (navy = TZP higher, gray = SEMA higher),
# saturation encodes mediation class:
#   direct_only  : vivid (full navy / full gray) - solid fill
#   partial      : 35% fade toward white - solid fill
#   fully_mediated: 60% fade toward white - solid fill
#   inconsistent : gold fill; significant indirect and direct effects with
#                  opposite signs
#   ns/unresolved: neutral gray (between-treatment sig but neither
#                  ACME nor ADE individually significant)
def _fade(hex_color: str, frac: float) -> str:
    """Fade a hex color toward white by `frac` (0=no fade, 1=pure white)."""
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(hex_color)
    faded = tuple(c + (1.0 - c) * frac for c in rgb)
    return mcolors.to_hex(faded)


def _palette():
    # Arm colors: TZP = dark navy, SEMA = medium gray.
    TZP = ARM_COLORS["TZP"]
    SEMA = ARM_COLORS["SEMA"]
    # Three intensity tiers per direction via programmatic fade toward white:
    #   direct        = full saturation (TZP navy / SEMA gray)
    #   partial       = 35 % fade toward white
    #   fully_mediated = 60 % fade toward white
    # The fade fractions keep each tier visibly distinct from both the NS
    # background and from each other.
    return {
        "ns": COLOR_NS,
        "direct_pos": TZP,
        "direct_neg": SEMA,
        "partial_pos": _fade(TZP, 0.35),
        "partial_neg": _fade(SEMA, 0.35),
        "fm_pos_fill": _fade(TZP, 0.60),
        "fm_neg_fill": _fade(SEMA, 0.60),
        # Unresolved: between-treatment significant but neither ACME nor
        # ADE individually reaches FDR < 0.05.  Single neutral color
        # (no direction encoding — the decomposition is inconclusive).
        "unresolved": "#B8B8B8",
        # Inconsistent effects use gold fill.
        "inconsistent": c["vibrant_gold"],
        "diagonal": c["black"],
        "axline": c["black"],
        "olink": TZP,
        "soma": SEMA,
        "total": c["black"],
        "direct": TZP,
        "mediated_gap": _fade(TZP, 0.35),
        # Aliases used by forest-plot legends.
        "partial": _fade(TZP, 0.35),
        "fully_mediated": _fade(TZP, 0.60),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _label_keys(row: dict) -> list[str]:
    """The drawn label of a mediation row (its protein label), or none when it has no gene."""
    return [row["gene_label"]] if row.get("gene_label") else []


def load_mediation(platform: str, week: int) -> pl.DataFrame:
    """One platform and week of classified mediation results, annotated with HGNC genes.

    Rows, FDRs and ``mclass`` are ``analysis.mediation.results.load_platform``'s; this adds
    ``genes`` (all HGNC symbols of the marker), ``gene_label`` (the protein label,
    ``analysis.protein_labels``; empty for an assay without one) and ``platform``.
    """
    mg = build_marker_genes()
    labels = marker_labels(platform)
    return (
        load_platform(platform, WEEK_TO_VISIT[week])
        .with_columns(
            pl.col("marker")
            .map_elements(lambda m: mg.get(m, []), return_dtype=pl.List(pl.Utf8))
            .alias("genes"),
        )
        .with_columns(
            pl.col("marker").replace_strict(labels, default="").alias("gene_label"),
            pl.lit(platform).alias("platform"),
        )
    )


def detect_cap(
    vals: np.ndarray, ratio: float = 2.0
) -> tuple[float | None, float | None]:
    """Detect outliers warranting truncation (volcano-figure convention).

    Returns ``(cap_hi, cap_lo)``. Walks the sorted values inward from each
    extreme: on the positive side, while the current extreme exceeds the
    next value by ≥ ``ratio``, extends the cap inward, so several stacked
    outliers are capped against one reference value. The cap sits at 1.2× the
    first value inside the bulk, so truncated markers land outside the main
    cloud but inside the panel.

    The volcano uses ratio=3; here ratio=2.
    """
    if len(vals) < 3:
        return (None, None)
    sv = np.sort(vals)

    # Positive side: walk from sv[-1] inward.
    cap_hi: float | None = None
    for k in range(len(sv) - 1, 0, -1):
        current = sv[k]
        nxt = sv[k - 1]
        if current > 0 and nxt > 0 and current > ratio * nxt:
            cap_hi = nxt * 1.2
        else:
            break

    # Negative side: walk from sv[0] outward.
    cap_lo: float | None = None
    for k in range(len(sv) - 1):
        current = sv[k]
        nxt = sv[k + 1]
        if current < 0 and nxt < 0 and current < ratio * nxt:
            cap_lo = nxt * 1.2
        else:
            break

    return (cap_hi, cap_lo)


def cap_exprs(finite: pl.DataFrame) -> tuple[pl.Expr, pl.Expr]:
    """Expressions marking the points truncated on the total (x) and the direct (y) effect.

    The single copy of the scatter's cap rule: caps come from the significant points
    (``total_fdr`` < ``FDR_THRESHOLD``, at least three) on ``total_estimate`` and ``ADE_estimate``,
    and any point beyond a cap is marked. With fewer than three significant points nothing is capped.

    Args:
        finite: Mediation frame whose ``total_estimate`` and ``ADE_estimate`` are finite.

    Returns:
        ``(capped on x, capped on y)`` boolean expressions to evaluate on ``finite``.
    """
    sig = finite.filter(pl.col("total_fdr") < FDR_THRESHOLD)
    if sig.height < 3:
        return pl.lit(False), pl.lit(False)
    exprs = []
    for col in ("total_estimate", "ADE_estimate"):
        hi, lo = detect_cap(sig[col].to_numpy())
        capped: pl.Expr = pl.lit(False)
        if hi is not None:
            capped = capped | (pl.col(col) > hi)
        if lo is not None:
            capped = capped | (pl.col(col) < lo)
        exprs.append(capped)
    return exprs[0], exprs[1]


def capped_gene_labels(med_df: pl.DataFrame) -> list[str]:
    """Sorted gene labels of the points ``plot_global_scatter`` truncates at the panel edge.

    Uses ``cap_exprs``, the scatter's own rule; the scatter draws only significant points, so
    only those are listed.
    """
    finite = med_df.filter(
        pl.col("total_estimate").is_finite() & pl.col("ADE_estimate").is_finite()
    )
    cap_x, cap_y = cap_exprs(finite)
    sig = finite.filter(pl.col("total_fdr") < FDR_THRESHOLD)
    return sorted(set(sig.filter(cap_x | cap_y)["gene_label"].to_list()))


# ---------------------------------------------------------------------------
# Row 1: global total-vs-direct scatters
# ---------------------------------------------------------------------------


def plot_global_scatter(
    ax,
    med_df: pl.DataFrame,
    week: int,
    platform: str,
    show_ylabel: bool = True,
) -> dict[str, int]:
    """Total-vs-direct effect scatter for one platform and week.

    ``med_df`` is ``load_mediation(platform, week)``. Points are the proteins whose
    mediation total effect clears FDR (``total_fdr``), colored by the shared ``mclass``.

    Returns a counts dict for caption annotation.
    """
    pal = _palette()
    # Drop rows with NaN total or ADE — these would break adjustText's KDTree
    # when passed via the `objects` parameter.
    plot_df = (
        med_df.filter(
            pl.col("total_estimate").is_finite() & pl.col("ADE_estimate").is_finite()
        )
        .with_columns(
            pl.col("mclass").alias("class"),
            pl.col("total_fdr").alias("gate_fdr"),
            (pl.col("total_fdr") < FDR_THRESHOLD).alias("gate_sig"),
        )
        .sort(["marker", "total_estimate", "ADE_estimate"])
    )

    # ----- Outlier capping (volcano-figure convention) -----
    # Detect caps from significant points only so a single NS extreme can't
    # trigger capping. Then mark every point exceeding either cap, store
    # original values, and compute truncated display positions.
    cap_x_expr, cap_y_expr = cap_exprs(plot_df)

    plot_df = plot_df.with_columns(
        [
            pl.col("total_estimate").alias("_orig_total"),
            pl.col("ADE_estimate").alias("_orig_ADE"),
            cap_x_expr.alias("_cap_x"),
            cap_y_expr.alias("_cap_y"),
        ]
    ).with_columns((pl.col("_cap_x") | pl.col("_cap_y")).alias("_capped"))

    # Axis span — 8 % padding around the most extreme significant non-capped
    # point. Labels anchored in the top 15 % of the panel flip to `va="top"`
    # (label-placement loop below), keeping them inside the frame.
    sig_non_capped = plot_df.filter(pl.col("gate_sig") & ~pl.col("_capped"))
    if sig_non_capped.height > 0:
        nc_x = float(sig_non_capped["total_estimate"].abs().max() or 0.0)
        nc_y = float(sig_non_capped["ADE_estimate"].abs().max() or 0.0)
        span = max(nc_x, nc_y, 0.05) * 1.08
    else:
        # No significant non-capped point: use all non-capped points.
        non_capped = plot_df.filter(~pl.col("_capped"))
        if non_capped.height > 0:
            nc_x = float(non_capped["total_estimate"].abs().max() or 0.0)
            nc_y = float(non_capped["ADE_estimate"].abs().max() or 0.0)
            span = max(nc_x, nc_y, 0.05) * 1.08
        else:
            span = (
                max(
                    abs(plot_df["total_estimate"].max() or 0.0),
                    abs(plot_df["total_estimate"].min() or 0.0),
                    0.1,
                )
                * 1.08
            )

    # Truncate capped points to 85 % of the axis range, preserving sign.
    # 85 % (vs the volcano's tighter 97 %) keeps the marker visibly displaced
    # from the corner annotations (↗ TZP > SEMA, ↙ SEMA > TZP, counts) but still close
    # to its true position — i.e. just enough to clear the corner labels.
    EDGE = span * 0.85
    disp_total = (
        pl.when(pl.col("_cap_x") & (pl.col("_orig_total") > 0))
        .then(pl.lit(EDGE))
        .when(pl.col("_cap_x") & (pl.col("_orig_total") < 0))
        .then(pl.lit(-EDGE))
        .otherwise(pl.col("total_estimate"))
    )
    disp_ADE = (
        pl.when(pl.col("_cap_y") & (pl.col("_orig_ADE") > 0))
        .then(pl.lit(EDGE))
        .when(pl.col("_cap_y") & (pl.col("_orig_ADE") < 0))
        .then(pl.lit(-EDGE))
        .otherwise(pl.col("ADE_estimate"))
    )
    plot_df = plot_df.with_columns(
        [
            disp_total.alias("_disp_total"),
            disp_ADE.alias("_disp_ADE"),
        ]
    )

    # Only points with total-effect FDR < 0.05 are drawn.

    # Significant points — colored by (class × direction).
    # Filled circles for direct & partial, open circles for fully-mediated.
    sig_nc = plot_df.filter(pl.col("gate_sig") & ~pl.col("_capped"))

    DOT_SIZE = 18
    DOT_EDGE = 0.4

    # Collect all scatter PathCollections so adjust_text can avoid them.
    scatter_objects: list = []

    def _draw_group(sub_df, *, facecolor, edgecolor, edgewidth, zorder=3):
        if sub_df.height == 0:
            return
        sc = ax.scatter(
            sub_df["_disp_total"].to_numpy(),
            sub_df["_disp_ADE"].to_numpy(),
            s=DOT_SIZE,
            facecolors=facecolor,
            edgecolors=edgecolor,
            lw=edgewidth,
            zorder=zorder,
        )
        scatter_objects.append(sc)

    # Plot z-order: direct > partial > mediated > unresolved > inconsistent.
    # Draw lowest-priority first so highest-priority lands on top in overlaps.

    # unresolved — neutral gray (weakest tier, zorder=2)
    _draw_group(
        sig_nc.filter(pl.col("class") == "unresolved"),
        facecolor=pal["unresolved"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=2,
    )
    # mediated — pale tinted fill, white edge (de-emphasized vs direct/partial)
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "fully_mediated") & (pl.col("total_estimate") > 0)
        ),
        facecolor=pal["fm_pos_fill"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "fully_mediated") & (pl.col("total_estimate") <= 0)
        ),
        facecolor=pal["fm_neg_fill"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    # inconsistent — gold filled (ACME and ADE opposite signs)
    _draw_group(
        sig_nc.filter(pl.col("class") == "inconsistent"),
        facecolor=pal["inconsistent"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=3,
    )
    # partial — light hue, white edge
    _draw_group(
        sig_nc.filter((pl.col("class") == "partial") & (pl.col("total_estimate") > 0)),
        facecolor=pal["partial_pos"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=4,
    )
    _draw_group(
        sig_nc.filter((pl.col("class") == "partial") & (pl.col("total_estimate") <= 0)),
        facecolor=pal["partial_neg"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=4,
    )
    # direct — vivid hue, white edge (drawn LAST so it lands on top of overlaps)
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "direct_only") & (pl.col("total_estimate") > 0)
        ),
        facecolor=pal["direct_pos"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=5,
    )
    _draw_group(
        sig_nc.filter(
            (pl.col("class") == "direct_only") & (pl.col("total_estimate") <= 0)
        ),
        facecolor=pal["direct_neg"],
        edgecolor="white",
        edgewidth=DOT_EDGE,
        zorder=5,
    )

    # Count summary — uses all sig (including capped).
    all_sig = plot_df.filter(pl.col("gate_sig"))
    counts: dict[str, int] = {
        cls: all_sig.filter(pl.col("class") == cls).height
        for cls in ("fully_mediated", "partial", "direct_only", "inconsistent", "unresolved")
    }

    # Capped points — directional triangle / diamond markers sized larger
    # so they read as "truncated, beyond axis" vs. regular dots. Class
    # encoding mirrors the non-capped scheme above. Only significant capped
    # points are drawn.
    capped = plot_df.filter(pl.col("_capped") & pl.col("gate_sig"))
    for r in capped.iter_rows(named=True):
        if r["_cap_x"] and r["_cap_y"]:
            mk = "D"
        elif r["_cap_x"]:
            mk = "<" if r["_orig_total"] < 0 else ">"
        else:
            mk = "v" if r["_orig_ADE"] < 0 else "^"
        pos = r["total_estimate"] > 0
        if r["class"] == "inconsistent":
            face, edge, ew = pal["inconsistent"], "white", DOT_EDGE
        elif r["class"] == "unresolved":
            face, edge, ew = pal["unresolved"], "white", DOT_EDGE
        elif r["class"] == "fully_mediated":
            face = pal["fm_pos_fill"] if pos else pal["fm_neg_fill"]
            edge, ew = "white", DOT_EDGE
        elif r["class"] == "partial":
            face = pal["partial_pos"] if pos else pal["partial_neg"]
            edge, ew = "white", DOT_EDGE
        else:  # direct_only
            face = pal["direct_pos"] if pos else pal["direct_neg"]
            edge, ew = "white", DOT_EDGE
        ax.scatter(
            [r["_disp_total"]],
            [r["_disp_ADE"]],
            marker=mk,
            s=DOT_SIZE * 2.2,
            facecolors=face,
            edgecolors=edge,
            lw=ew,
            zorder=5,
        )

    # Reference lines: solid diagonal (y=x), solid x and y axes.
    # Use the panel-level `span` (40 % padded) so the diagonal stretches
    # corner-to-corner regardless of where extreme points land.
    ax.plot(
        [-span, span], [-span, span], color=pal["diagonal"], lw=0.6, alpha=0.7, zorder=2
    )
    ax.axhline(0, color=pal["axline"], lw=0.4, alpha=0.5, zorder=2)
    ax.axvline(0, color=pal["axline"], lw=0.4, alpha=0.5, zorder=2)

    # Create direction cues before label adjustment so they act as obstacles.
    # The limits are set here so transAxes coordinates land at the final positions.
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    EDGE_MARGIN = 0.008  # 0.8 % from top/bottom border
    DIAG_INSET = 0.04  # offset from right/left edge to clear the diagonal
    cue_tzp = ax.text(
        1 - DIAG_INSET,
        1 - EDGE_MARGIN,
        "↗ TZP > SEMA",
        transform=ax.transAxes,
        fontsize=FS_DIRECTION,
        color=pal["direct_pos"],
        ha="right",
        va="top",
        zorder=6,
    )
    cue_sema = ax.text(
        DIAG_INSET,
        EDGE_MARGIN,
        "↙ SEMA > TZP",
        transform=ax.transAxes,
        fontsize=FS_DIRECTION,
        color=pal["direct_neg"],
        ha="left",
        va="bottom",
        zorder=6,
    )

    # ----- Label selection: one label per protein label (analysis.protein_labels) -----
    # ONLY genes significant in between-treatment (FDR<0.05) are considered —
    # the `sig` filter above guarantees this. Primary ranker is the
    # between-treatment FDR (smallest first) split by direction, mirroring the
    # volcano figure. ADE/ACME extremes layered in afterward as supplements.
    # Capped points are ALWAYS labeled so the reader sees what's truncated.
    # A multi-gene marker is one entry under its label (KLK3/SERPINA3); markers sharing a label
    # (several SomaScan aptamers of one gene) pool under it.
    sig_all = plot_df.filter(pl.col("gate_sig"))  # includes capped + non-capped sig
    gene_points: dict[str, list[dict]] = {}
    for r in sig_all.iter_rows(named=True):
        for gene in _label_keys(r):
            gene_points.setdefault(gene, []).append(r)

    def _row_tiebreak(r: dict) -> tuple[str, str, float, float]:
        return (
            str(r.get("marker") or ""),
            "|".join(r.get("genes") or []),
            float(r.get("_orig_total") or 0.0),
            float(r.get("_orig_ADE") or 0.0),
        )

    def _best_by(rows, col, mode="abs"):
        if mode == "min":
            return sorted(rows, key=lambda r: (float(r[col]),) + _row_tiebreak(r))[0]
        return sorted(
            rows,
            key=lambda r: (-abs(float(r[col])),) + _row_tiebreak(r),
        )[0]

    # ----- Class-aware label selection -----
    # Each class gets labeled members per direction (TZP > SEMA and SEMA > TZP)
    # even when they miss the volcano-pool cutoffs.
    def _best_in_class(rows, cls, key):
        cand = [r for r in rows if r["class"] == cls]
        return _best_by(cand, key) if cand else None

    def _direction_up(rows):
        return _best_by(rows, "total_estimate")["total_estimate"] > 0

    def _vol_score(rows):
        best = _best_by(rows, "total_estimate")
        return abs(best["total_estimate"]) * -np.log10(max(best["gate_fdr"], 1e-300))

    def _top_per_dir(pool, score_fn, n=3):
        """Top n per direction from pool=[(g, rows, ...)] ranked by score_fn."""
        scored_up = sorted(
            ((g, rs) for g, rs in pool if _direction_up(rs)),
            key=lambda kv: (-score_fn(kv[1]), kv[0]),
        )[:n]
        scored_dn = sorted(
            ((g, rs) for g, rs in pool if not _direction_up(rs)),
            key=lambda kv: (-score_fn(kv[1]), kv[0]),
        )[:n]
        return scored_up + scored_dn

    # Volcano pool: headline differential genes, ranked by combined
    # |total| × −log10(gate_fdr) so effect size and significance trade off.
    by_volcano = _top_per_dir(
        list(gene_points.items()),
        _vol_score,
        n=N_VOLCANO_PER_DIR,
    )

    # Rank inconsistent effects by absolute total effect within each direction
    # and apply the configured label budget (``N_SUPPRESSION_PER_DIR``).
    sup_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "inconsistent", "total_estimate") is not None
    ]
    suppression_pool = _top_per_dir(
        sup_pool,
        lambda rs: abs(
            _best_in_class(rs, "inconsistent", "total_estimate")["total_estimate"]
        ),
        n=N_SUPPRESSION_PER_DIR,
    )

    # Top direct-only per direction by |ADE|.
    do_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "direct_only", "ADE_estimate") is not None
    ]
    by_direct = _top_per_dir(
        do_pool,
        lambda rs: abs(
            _best_in_class(rs, "direct_only", "ADE_estimate")["ADE_estimate"]
        ),
        n=N_DIRECT_PER_DIR,
    )

    # Top fully-mediated per direction by |ACME|.
    fm_pool = [
        (g, rs)
        for g, rs in gene_points.items()
        if _best_in_class(rs, "fully_mediated", "ACME_estimate") is not None
    ]
    by_mediated = _top_per_dir(
        fm_pool,
        lambda rs: abs(
            _best_in_class(rs, "fully_mediated", "ACME_estimate")["ACME_estimate"]
        ),
        n=N_MEDIATED_PER_DIR,
    )

    # Only capped points with a significant total effect are labeled.
    capped_pool: list[tuple[str, list[dict]]] = []
    capped_genes: set[str] = set()
    for r in capped.filter(pl.col("gate_sig")).iter_rows(named=True):
        for gene in _label_keys(r):
            if gene in capped_genes:
                continue
            capped_genes.add(gene)
            capped_pool.append((gene, gene_points.get(gene, [r])))

    # Anchor each selected gene at the matching analyte. Pool order matters
    # only for which row anchors the label (the gene set is a union): capped
    # first, then inconsistent, then the volcano pool, then the per-class
    # direction-split pools.
    selected: dict[str, dict] = {}
    for source, col, mode in (
        (capped_pool, "gate_fdr", "min"),
        (suppression_pool, "gate_fdr", "min"),
        (by_volcano, "gate_fdr", "min"),
        (by_direct, "ADE_estimate", "abs"),
        (by_mediated, "ACME_estimate", "abs"),
    ):
        for gene, rows in source:
            if gene not in selected:
                selected[gene] = _best_by(rows, col, mode=mode)

    # Split selection: capped genes get manual placement (volcano-style
    # gene name + gray coordinate annotation right next to the truncated
    # marker). Non-capped genes go through adjustText as usual.
    capped_selected = {g: r for g, r in selected.items() if r["_capped"]}
    selected = {g: r for g, r in selected.items() if not r["_capped"]}

    # One Text per gene. Same pattern as `_volcano_figure.py`:
    # start with `va="bottom"` so the text bbox sits naturally *above* the
    # marker (avoiding the zero-gradient degeneracy of `va="center"`).
    #
    # Labels near the top edge flip `va` so they sit below the marker, away from
    # the `↗ TZP > SEMA` cue.
    #
    # A gene with several analytes is anchored at their centroid; lead lines fan
    # out from there to every analyte.
    texts = []
    label_anchors: list[tuple[str, float, float, bool]] = []
    for gene, anchor in sorted(
        selected.items(),
        key=lambda item: (item[1]["_disp_total"], item[1]["_disp_ADE"], item[0]),
    ):
        rows = gene_points[gene]
        multi = len(rows) > 1
        if multi:
            bx = float(np.mean([r["_disp_total"] for r in rows]))
            by = float(np.mean([r["_disp_ADE"] for r in rows]))
        else:
            bx = anchor["_disp_total"]
            by = anchor["_disp_ADE"]
        # Anchors in the top 15 % of the panel use va="top" so the label
        # sits BELOW the marker, away from the `↗ TZP > SEMA` corner cue.
        # Bottom-15 % anchors keep the default va="bottom" (label above
        # marker, away from the `↙ SEMA > TZP` cue).
        va_init = "top" if by > 0.85 * span else "bottom"
        texts.append(
            ax.text(
                bx,
                by,
                gene,
                fontsize=FS_GENE_LABEL,
                color="#222",
                ha="center",
                va=va_init,
                zorder=5,
            )
        )
        label_anchors.append((gene, bx, by, multi))

    label_arrowprops = dict(arrowstyle="-", color="#777", lw=0.4, alpha=0.7)
    label_arrows: list = []
    if texts:
        # `min_arrow_len` drops the lead line when a label stays near its
        # anchor; labels that moved keep their connector.
        #
        # `objects=[cue_tzp, cue_sema]` makes the direction cues static
        # obstacles; `force_static` is raised above the default (0.1, 0.2) so
        # they repel more strongly than text-text avoidance.
        _, label_arrows = deterministic_adjust_text(
            texts,
            ax=ax,
            objects=[cue_tzp, cue_sema],
            arrowprops=label_arrowprops,
            expand=(1.2, 1.4),
            force_static=(0.6, 1.2),
            min_arrow_len=4,
        )

        # Post-pass safety net: adjust_text occasionally leaves a label
        # straddling the cue bbox when the panel is dense. Iterate over
        # labels, check display-space overlap with each cue, and shift any
        # offender down (top cue) or up (bottom cue) just enough to clear.
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        for cue, direction in [(cue_tzp, "down"), (cue_sema, "up")]:
            cue_bb = cue.get_window_extent(renderer=renderer)
            for txt in texts:
                tb = txt.get_window_extent(renderer=renderer)
                if tb.overlaps(cue_bb):
                    px, py = txt.get_position()
                    inv = ax.transData.inverted()
                    if direction == "down":
                        shift_px = (cue_bb.y0 - tb.y1) - 2  # 2 px clearance
                    else:
                        shift_px = (cue_bb.y1 - tb.y0) + 2
                    _, dy = inv.transform((0, shift_px)) - inv.transform((0, 0))
                    txt.set_position((px, py + dy))

    # What ``separate_labels`` needs to settle overlaps at the figure's final geometry; a
    # label with secondary lines (several analytes) is never moved.
    ax.gene_labels = dict(
        texts=texts,
        targets=[None if multi else (bx, by) for _, bx, by, multi in label_anchors],
        arrows=label_arrows,
        arrowprops=label_arrowprops,
    )

    # Secondary lines: for any selected gene measured by >1 analyte, draw a
    # thin connector from the post-adjustText label position to *every*
    # analyte point of that gene. Since multi-analyte labels are anchored
    # at the centroid (not at one specific analyte), every analyte gets a
    # secondary line — there is no "primary" analyte to skip.
    for txt, (gene, bx, by, multi) in zip(texts, label_anchors):
        if not multi:
            continue
        rows = gene_points[gene]
        lx, ly = txt.get_position()
        for r in rows:
            ax.annotate(
                "",
                xy=(r["_disp_total"], r["_disp_ADE"]),
                xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.3, alpha=0.7),
                zorder=4,
            )

    # Capped-point labels: each capped gene gets a 2-line label block
    # (gene name in dark, coordinates in gray below) placed near its
    # marker. Direction depends on the marker geometry:
    #
    #   • Triangle on the bottom/top edge (only y-capped): label goes to
    #     the SIDE (toward the panel interior in x).
    #   • Diamond at a corner (both x AND y capped):
    #       - alone at that y-edge → SIDE (toward the data cloud)
    #       - sharing the y-edge with another capped marker → VERTICAL
    #         (away from edge), so the diamond's label sits ABOVE/BELOW
    #         the marker and the companion's label sits to the SIDE
    SIDE_OFFSET_PT = 4.0  # horizontal gap from marker to side label
    SIDE_HALF_BLOCK_PT = 3.5  # side-placed block centered on marker y
    SIDE_DROP_PT = 2.0  # tiny downward shift of side blocks at
    # the bottom edge (or upward at top) so
    # a companion vertical-above label can
    # sit closer to its own marker without
    # the two labels touching
    VERTICAL_BASE_OFFSET_PT = 9.0  # vertical: above-placed label's coord
    # above marker — set to just clear the
    # companion side-label's top edge
    LINE_HEIGHT_PT = 7.0  # spacing between the two lines of a
    # vertical-above block

    from matplotlib import transforms

    # Group capped markers by y-edge so we can detect "companions".
    y_groups: dict[float, list[tuple[str, dict]]] = {}
    for gene, anchor in capped_selected.items():
        key = round(anchor["_disp_ADE"], 3)
        y_groups.setdefault(key, []).append((gene, anchor))

    for by, members in y_groups.items():
        for gene, anchor in members:
            ox = anchor["_orig_total"]
            oy = anchor["_orig_ADE"]
            bx = anchor["_disp_total"]
            cap_x = anchor["_cap_x"]
            cap_y = anchor["_cap_y"]
            is_corner = bool(cap_x and cap_y)
            on_left = anchor["_orig_total"] < 0
            on_bottom = anchor["_orig_ADE"] < 0
            has_companions = len(members) > 1

            # Direction decision: vertical (above/below marker) when the
            # diamond at a corner has a companion sharing the same y-edge;
            # horizontal (to the side) otherwise. In BOTH cases the two
            # text lines stack as a single grouped block with the gene
            # name on top and the coordinates directly below (closer to
            # the marker) — never split across the marker.
            # Bottom-left and top-right corners are the same corners used by
            # the directional cues, so keep those capped labels vertical even
            # when they have no companion marker at that edge.
            place_above = is_corner and (has_companions or (on_left == on_bottom))
            if place_above:
                x_off = 0
                ha = "center"
            else:
                x_off = (1 if on_left else -1) * SIDE_OFFSET_PT
                ha = "left" if on_left else "right"

            # Block layout:
            #   place_above=True  → block sits VERTICALLY above (or below)
            #     the marker. Both lines on the same side; gene name at
            #     the top, coord directly underneath. The base offset is
            #     large enough to clear any companion side-label that
            #     shares the same y-edge.
            #   place_above=False → block is CENTERED on the marker y,
            #     positioned to the side. Gene sits half-block above the
            #     marker line, coord sits half-block below. The marker
            #     glyph occupies the same y range but is offset to the
            #     left in x, so the gene+coord still read as a single
            #     2-line label "right next to" the marker.
            if place_above:
                block_dir = 1 if on_bottom else -1
                base_off = VERTICAL_BASE_OFFSET_PT
                coord_y_off = block_dir * base_off
                gene_y_off = block_dir * (base_off + LINE_HEIGHT_PT)
                gene_va = "bottom" if block_dir > 0 else "top"
                coord_va = gene_va
            else:
                # Tiny downward (for bottom-edge marker) or upward (for
                # top-edge) shift so a companion vertical-above label can
                # tuck in closer to its own marker.
                drop = -SIDE_DROP_PT if on_bottom else SIDE_DROP_PT
                gene_y_off = SIDE_HALF_BLOCK_PT + drop
                coord_y_off = -SIDE_HALF_BLOCK_PT + drop
                gene_va = "center"
                coord_va = "center"

            # Coordinates (gray).
            ax.text(
                bx,
                by,
                f"({ox:+.2f}, {oy:+.2f})".replace("-", "−"),
                fontsize=FS_GENE_LABEL - 1,
                color="#888",
                ha=ha,
                va=coord_va,
                transform=transforms.offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=x_off,
                    y=coord_y_off,
                    units="points",
                ),
                zorder=6,
            )
            # Gene name (dark).
            ax.text(
                bx,
                by,
                gene,
                fontsize=FS_GENE_LABEL,
                color="#222",
                ha=ha,
                va=gene_va,
                transform=transforms.offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=x_off,
                    y=gene_y_off,
                    units="points",
                ),
                zorder=6,
            )

    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xlabel("Total effect", fontsize=FS_AXLABEL)
    ax.set_ylabel("Direct effect" if show_ylabel else "", fontsize=FS_AXLABEL)
    ax.tick_params(labelsize=FS_TICK)

    # Island-style "Week 24" / "Week 72" pill at top-center (volcano convention).
    ax.text(
        0.5,
        0.97,
        f"Week {week}",
        transform=ax.transAxes,
        fontsize=FS_PILL,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#F0F0F0",
            edgecolor="#CCCCCC",
            linewidth=0.4,
        ),
        zorder=6,
    )

    # Direction cues are created above (before adjust_text) so they can be
    # passed as obstacles to avoid; no need to draw them again here.

    # In-panel legend: single column, ordered by color (red shades → blue
    # shades → inconsistent). All swatches use a white edge so mediated
    # (paler fill) reads as de-emphasized relative to direct/partial.
    def _swatch(face, edge, ew):
        return Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=4.0,
            markeredgewidth=ew,
        )

    def _count_dir(cls_name, sign):
        cls_df = all_sig.filter(pl.col("class") == cls_name)
        if sign == "pos":
            return cls_df.filter(pl.col("total_estimate") > 0).height
        return cls_df.filter(pl.col("total_estimate") <= 0).height

    legend_handles: list = []
    legend_labels: list[str] = []

    # Order: TZP shades (TZP > SEMA) — vivid → light → palest, then SEMA
    # shades (SEMA > TZP) in the same order, then inconsistent last.
    legend_specs = [
        ("Direct, TZP > SEMA", "direct_only", "pos", pal["direct_pos"]),
        ("Partial, TZP > SEMA", "partial", "pos", pal["partial_pos"]),
        ("Mediated, TZP > SEMA", "fully_mediated", "pos", pal["fm_pos_fill"]),
        ("Direct, SEMA > TZP", "direct_only", "neg", pal["direct_neg"]),
        ("Partial, SEMA > TZP", "partial", "neg", pal["partial_neg"]),
        ("Mediated, SEMA > TZP", "fully_mediated", "neg", pal["fm_neg_fill"]),
    ]
    for label, cls_name, sign, color in legend_specs:
        n = _count_dir(cls_name, sign)
        if n:
            legend_handles.append(_swatch(color, "white", 0.4))
            legend_labels.append(f"{label} ({n})")

    if counts["unresolved"]:
        legend_handles.append(_swatch(pal["unresolved"], "white", 0.4))
        legend_labels.append(f"Unresolved ({counts['unresolved']})")

    if counts["inconsistent"]:
        legend_handles.append(_swatch(pal["inconsistent"], "white", 0.4))
        legend_labels.append(f"Inconsistent ({counts['inconsistent']})")

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=FS_BODY,
        frameon=False,
        ncol=1,
        handlelength=0.65,
        handletextpad=0.22,
        labelspacing=0.14,
        borderpad=0.15,
        borderaxespad=0.15,
    )
    return counts


# ---------------------------------------------------------------------------
# Rows 2-3: theme forest pairs (Olink | SomaScan)
# ---------------------------------------------------------------------------

SCATTER_BANNER_TAIL = " · TZP vs SEMA · mediator: % Δweight"
PAIR_LETTER_DY_IN = BANNER_HEIGHT_IN + BANNER_GAP_IN
PAIR_WSPACE = 0.62  # inches between the Olink and SomaScan panels of a pair, and between pairs
# Margins (inches) of the mediation figures' panel grids.
GRID_LEFT_IN = 0.58
GRID_RIGHT_IN = 0.32
GRID_TOP_IN = 0.44
# The forest legend sits in a strip this tall at this height, below the panel grid.
LEGEND_STRIP_BOTTOM_IN = 0.82
LEGEND_STRIP_H_IN = 0.2
FOREST_GRID_BOTTOM_IN = LEGEND_STRIP_BOTTOM_IN + LEGEND_STRIP_H_IN + 1 / 3  # 24 pt above the strip


def separate_labels(ax) -> list[str]:
    """Clear any overlap between the gene labels of a ``plot_global_scatter`` axes.

    Args:
        ax: An axes ``plot_global_scatter`` drew on, at its final position.

    Returns:
        The labels that moved.
    """
    labels = ax.gene_labels
    return separate_overlapping_labels(
        ax,
        labels["texts"],
        labels["targets"],
        labels["arrows"],
        arrowprops=labels["arrowprops"],
        min_arrow_len=4,
    )


def plot_theme_pair(
    ax_ol,
    ax_so,
    *,
    genes: list[str],
    med_ol: pl.DataFrame,
    med_so: pl.DataFrame,
    week: int = LATE_WEEK,
    n_max: int = N_PROTEINS_PER_THEME,
    label_line_offset_pt: float = mf.LABEL_LINE_OFFSET_PT,
) -> list[dict]:
    """Draw a theme's row-aligned Olink and SomaScan forests sharing one x-range.

    Args:
        ax_ol: Olink axes.
        ax_so: SomaScan axes.
        genes: The theme's HGNC symbols.
        med_ol: ``load_mediation("olink", week)``.
        med_so: ``load_mediation("soma", week)``.
        week: Week of the mediation results, for the axis label.
        n_max: Most proteins selected per platform.
        label_line_offset_pt: Points between a label's gene and marker lines.

    Returns:
        The aligned rows drawn.
    """
    rows = mf.aligned_theme_rows(genes, {"olink": med_ol, "soma": med_so}, n_max=n_max)
    xlim = mf.forest_xlim(rows)
    for ax, platform in ((ax_ol, "olink"), (ax_so, "soma")):
        mf.draw_forest(
            ax,
            rows,
            platform,
            xlim=xlim,
            xlabel=f"Effect, week {week} (95% CI)",
            label_line_offset_pt=label_line_offset_pt,
        )
    return rows


def add_forest_legend(fig) -> None:
    """Shared bottom legend of the three-glyph forest encoding, in a strip below the grid."""
    fig_w, fig_h = fig.get_size_inches()
    strip = fig.add_axes(
        (
            GRID_LEFT_IN / fig_w,
            LEGEND_STRIP_BOTTOM_IN / fig_h,
            (fig_w - GRID_LEFT_IN - GRID_RIGHT_IN) / fig_w,
            LEGEND_STRIP_H_IN / fig_h,
        )
    )
    strip.set_axis_off()
    strip.legend(
        handles=mf.forest_legend_handles(),
        labels=list(mf.FOREST_LEGEND_LABELS),
        loc="upper center",
        ncols=4,
        frameon=False,
        fontsize=FS_AUX,
        borderpad=0.0,  # the entries' top meets the strip's top
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.5,
    )


def align_to_aspect_box(span_ax, axes) -> None:
    """Rescale the columns of ``axes`` into the box ``span_ax`` keeps after its fixed aspect.

    ``span_ax`` spans the same grid columns as ``axes``; its equal aspect narrows its box,
    and the panels below are narrowed and shifted the same way so their edges line up.
    """
    span_ax.apply_aspect()
    grid = span_ax.get_position(original=True)
    box = span_ax.get_position()
    scale = box.width / grid.width
    for ax in axes:
        pos = ax.get_position()
        x0 = box.x0 + (pos.x0 - grid.x0) * scale
        ax.set_position((x0, pos.y0, pos.width * scale, pos.height))


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure():
    """Build Figure 4: week-72 total-vs-direct scatters (a, b) and four theme forest pairs (c-f).

    Reads each platform's classified mediation results (``load_mediation``); theme genes
    are cut to those with a significant total effect (``figure_themes``).

    Returns:
        The figure.
    """
    init_figure_theme()

    med_ol = load_mediation("olink", LATE_WEEK)
    med_so = load_mediation("soma", LATE_WEEK)

    def _theme_row(
        theme_names: tuple[str, ...], start_pid: int
    ) -> tuple[list[int], dict[str, tuple[int, int]], int]:
        row: list[int] = []
        panel_ids: dict[str, tuple[int, int]] = {}
        for theme_name in theme_names:
            panel_ids[theme_name] = (start_pid, start_pid + 1)
            row += [start_pid] * 3 + [start_pid + 1] * 3
            start_pid += 2
        return row, panel_ids, start_pid

    scatter_row = [1] * 6 + [2] * 6
    forest_row_1, forest_ids_1, next_pid = _theme_row(MAIN_THEME_ROWS[0], 3)
    forest_row_2, forest_ids_2, _ = _theme_row(MAIN_THEME_ROWS[1], next_pid)
    theme_panel_ids = forest_ids_1 | forest_ids_2
    main_themes = figure_themes(MAIN_THEME_CANDIDATES, MAIN_THEME_ORDER, LATE_WEEK)

    fig = panel_figure(*nature_figsize("2col", 9.45))
    layout = [scatter_row, forest_row_1, forest_row_2]
    grid = dict(
        left=GRID_LEFT_IN,
        right=GRID_RIGHT_IN,
        top=GRID_TOP_IN,
        wspace=(0.0, 0.0, PAIR_WSPACE, 0.0, 0.0, PAIR_WSPACE, 0.0, 0.0, PAIR_WSPACE, 0.0, 0.0),
        hspace=(0.84, 0.74),
        hratios=(1.65, 1.0, 1.0),
    )
    # Gene labels are placed with the grid reaching the legend strip's bottom; the panels then
    # move to their final boxes and the labels keep their data positions.
    axs = panel_grid(fig, layout, bottom=LEGEND_STRIP_BOTTOM_IN, **grid)
    pid_to_ax = {pid: ax for pid, ax in enumerate(axs, start=1)}

    plot_global_scatter(pid_to_ax[1], med_ol, LATE_WEEK, "olink", show_ylabel=True)
    plot_global_scatter(pid_to_ax[2], med_so, LATE_WEEK, "soma", show_ylabel=False)
    final_boxes = grid_boxes(fig, layout, bottom=FOREST_GRID_BOTTOM_IN, **grid)
    for ax, box in zip(axs, final_boxes, strict=True):
        ax.set_position(box)
    align_to_aspect_box(pid_to_ax[1], [pid_to_ax[pid] for pid in (3, 4, 7, 8)])
    align_to_aspect_box(pid_to_ax[2], [pid_to_ax[pid] for pid in (5, 6, 9, 10)])

    for theme_name in MAIN_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        plot_theme_pair(
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            genes=main_themes[theme_name],
            med_ol=med_ol,
            med_so=med_so,
            n_max=N_PROTEINS_PER_THEME,
        )

    # The labels were placed at the pre-final grid; move the fewest that overlap at the
    # final one, as little as possible.
    for pid in (1, 2):
        separate_labels(pid_to_ax[pid])

    add_forest_legend(fig)
    fig.canvas.draw()
    for pid, name in ((1, "Olink"), (2, "SomaScan")):
        draw_banner(fig, pid_to_ax[pid], pid_to_ax[pid], bold_text=name, tail_text=SCATTER_BANNER_TAIL)
    for theme_name in MAIN_THEME_ORDER:
        panel_ol, panel_so = theme_panel_ids[theme_name]
        draw_banner(
            fig,
            pid_to_ax[panel_ol],
            pid_to_ax[panel_so],
            bold_text=MAIN_THEME_LABELS[theme_name].replace("\n", " "),
            left_edge_text="Olink",
            right_edge_text="SomaScan",
        )

    fig.canvas.draw()
    place_panel_letter(fig, pid_to_ax[1], "a", extra_dy_in=PAIR_LETTER_DY_IN)
    place_panel_letter(fig, pid_to_ax[2], "b", extra_dy_in=PAIR_LETTER_DY_IN)
    for label, theme_name in zip("cdef", MAIN_THEME_ORDER, strict=False):
        panel_ol, _ = theme_panel_ids[theme_name]
        place_panel_letter(fig, pid_to_ax[panel_ol], label, extra_dy_in=PAIR_LETTER_DY_IN)

    return fig
