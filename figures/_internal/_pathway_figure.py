"""Figure 3 — pathway enrichment dot plots (cameraPR), Olink + SomaScan."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import polars as pl
import ultraplot as uplt
from matplotlib.patheffects import withStroke

from ._common import (
    ANALYSIS_DIR,
    COVAR,
    COLOR_DOWN,
    COLOR_UP,
    COLOR_NS,
    FDR_THRESHOLD,
    NAT_W2,
    RESULTS_DIR,
    build_gene_lookup,
    build_marker_genes,
    draw_segments_left_aligned,
    draw_segments_right_aligned,
    init_figure_theme,
    load_marker_gene_pairs,
    save_figure,
)
from ._pathway_labels import pathway_label

# ===========================================================================
# PATHWAY SECTION
# ===========================================================================

COLOR_MIXED = "#7B2D8E"  # purple — gene sig in opposite directions across platforms
COLOR_LINE = "#888888"

FIGURES_DIR = Path(__file__).resolve().parents[1]
REPO = FIGURES_DIR.parent
MAPPING_CSV = FIGURES_DIR / "configs" / "pathway_figure_terms.csv"

CAMERA_FILES = {
    ("olink", 24): RESULTS_DIR
    / "surmount5_olink"
    / COVAR
    / "pathwayRes"
    / "acTrt"
    / "TZP15mgorMTDVSSEMA2.4mgorMTD@24"
    / "camera_combined.csv",
    ("olink", 72): RESULTS_DIR
    / "surmount5_olink"
    / COVAR
    / "pathwayRes"
    / "acTrt"
    / "TZP15mgorMTDVSSEMA2.4mgorMTD@72"
    / "camera_combined.csv",
    ("soma", 24): RESULTS_DIR
    / "surmount5_soma"
    / COVAR
    / "pathwayRes"
    / "acTrt"
    / "TZP15mgorMTDVSSEMA2.4mgorMTD@24"
    / "camera_combined.csv",
    ("soma", 72): RESULTS_DIR
    / "surmount5_soma"
    / COVAR
    / "pathwayRes"
    / "acTrt"
    / "TZP15mgorMTDVSSEMA2.4mgorMTD@72"
    / "camera_combined.csv",
}

# Marker size scaling: sqrt(N_genes) * SIZE_SCALE, clamped to [SIZE_MIN, SIZE_MAX]
SIZE_SCALE = 9.0
SIZE_MIN = 22.0
SIZE_MAX = 140.0
SIZE_NS = 14.0  # constant small size for non-significant markers

# X-axis cap on -log10(FDR) so outliers don't dominate
NEGLOG_CAP = 12.0

# Y-spacing
PATHWAY_GAP = 0.34  # unit between consecutive pathways in the same theme
INTER_THEME_GAP = 0.5  # extra vertical gap between themes (kept wide so themes
# stay visually grouped even as intra-theme rows are tightened)
THEME_NAME_OFFSET = 0.30  # theme-name y-data offset above first pathway in group

# Banner geometry
THEME_BANNER_PAD = 0.2  # data-units of vertical padding above/below pathway rows
THEME_BANNER_WIDTH = None  # None = gray theme strip fills the full left gutter,
# so it sits behind the theme + pathway-term label text (supp_fig11_pathway_ora
# overrides this with its own width).
LEAD_GENE_OFFSET = (
    0.002  # figure-fraction offset from pathway-right axis to lead-gene labels
)

# Protein-level data files for lead-gene direction coloring
_PROTEIN_FILES = {
    ("olink", 24): RESULTS_DIR
    / "surmount5_olink"
    / COVAR
    / "finalRes"
    / "acTrt"
    / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@24_py.csv",
    ("olink", 72): RESULTS_DIR
    / "surmount5_olink"
    / COVAR
    / "finalRes"
    / "acTrt"
    / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@72_py.csv",
    ("soma", 24): RESULTS_DIR
    / "surmount5_soma"
    / COVAR
    / "finalRes"
    / "acTrt"
    / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@24_py.csv",
    ("soma", 72): RESULTS_DIR
    / "surmount5_soma"
    / COVAR
    / "finalRes"
    / "acTrt"
    / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@72_py.csv",
}


# ---------------------------------------------------------------------------
# Pathway data loading
# ---------------------------------------------------------------------------


def load_camera_for_terms(mapping: pl.DataFrame) -> pl.DataFrame:
    """For each (term x platform x week), return its FDR/direction/N_genes if FDR<0.05.

    Returns long-format dataframe with columns:
      theme_id, display_label, original_term, database, lead_genes_in_figure,
      platform, week, direction, fdr_q, neglog10_fdr, n_genes, sig
    """
    frames = []
    for (platform, week), path in CAMERA_FILES.items():
        df = (
            pl.read_csv(path)
            .select(
                [
                    pl.col("Gene_Set_Database").alias("database"),
                    pl.col("Term").alias("original_term"),
                    pl.col("Direction").alias("direction"),
                    pl.col("NOM p-val").alias("nominal_p"),
                    pl.col("FDR q-val").alias("fdr_q"),
                    pl.col("N_Genes").alias("n_genes"),
                    pl.col("Total_Genes").alias("total_genes"),
                    pl.col("Lead_genes").alias("lead_genes_full"),
                ]
            )
            .with_columns(
                pl.lit(platform).alias("platform"),
                pl.lit(week).alias("week"),
            )
        )
        frames.append(df)
    camera = pl.concat(frames)

    # Inner-join mapping to the camera rows (only the 32 selected terms)
    long = mapping.join(camera, on=["database", "original_term"], how="inner")
    long = long.with_columns(
        (pl.col("fdr_q") < FDR_THRESHOLD).alias("sig"),
        (-(pl.col("fdr_q").log10())).alias("neglog10_fdr"),
    )
    return long


def collapse_per_cell(long: pl.DataFrame) -> pl.DataFrame:
    """For each (term x platform x week), pick the *significant* row.

    cameraPR returns one row per (term, direction). For a given cell, only one
    direction will typically be significant (Up XOR Down). We keep the sig row
    if present; otherwise we keep the better-FDR row but mark sig=False.
    """
    long = long.with_columns(
        pl.struct([pl.col("sig").not_(), pl.col("fdr_q")]).alias("_rank_key")
    )
    out = (
        long.sort("_rank_key")
        .group_by(
            [
                "theme_id",
                "display_label",
                "original_term",
                "database",
                "platform",
                "week",
            ],
            maintain_order=True,
        )
        .agg(
            [
                pl.col("theme_name").first(),
                pl.col("lead_genes_in_figure").first(),
                pl.col("direction").first(),
                pl.col("nominal_p").first(),
                pl.col("fdr_q").first(),
                pl.col("neglog10_fdr").first(),
                pl.col("n_genes").first(),
                pl.col("total_genes").first(),
                pl.col("lead_genes_full").first(),
                pl.col("sig").first(),
            ]
        )
        .drop("_rank_key", strict=False)
    )
    return out


# ---------------------------------------------------------------------------
# Protein-level significance lookup (drives lead-gene ordering + bolding)
# ---------------------------------------------------------------------------


def load_gene_protein_data() -> dict[str, list[dict]]:
    """For each gene_symbol, return per-cell across-treatment stats.

    Returns dict gene_symbol -> list of {platform, week, fdr, direction}
    entries.
    """
    marker_gene = load_marker_gene_pairs()
    out: dict[str, list[dict]] = {}
    for (platform, week), path in _PROTEIN_FILES.items():
        vol = pl.read_csv(path).select(["marker", "Assay", "fdr", "FC"])
        joined = vol.join(marker_gene, on="marker", how="inner")
        leftover = (
            vol.join(marker_gene, on="marker", how="anti")
            .with_columns(pl.col("Assay").alias("gene_symbol"))
            .select(joined.columns)
        )
        merged = pl.concat([joined, leftover])
        for r in merged.iter_rows(named=True):
            out.setdefault(r["gene_symbol"], []).append(
                {
                    "platform": platform,
                    "week": week,
                    "fdr": r["fdr"],
                    "direction": "Up" if r["FC"] > 1 else "Down",
                }
            )
    return out


def compute_lead_gene_stats(
    mapping: pl.DataFrame,
    panel: pl.DataFrame,
    gene_protein_data: dict[str, list[dict]],
    sig_threshold: float = FDR_THRESHOLD,
) -> dict[tuple[str, str], dict]:
    """For each (display_label, gene) lead-gene cell, compute:
    - fdr: gene's min protein-level FDR across all cells (sort order).
    - direction: matched direction (Up/Down/Mixed) when the gene is
      itself significant in the across-treatment contrast on a platform
      where the pathway is also significant; None otherwise.
    - matches_pathway: True iff such a co-occurring sig cell exists.
    """
    out: dict[tuple[str, str], dict] = {}
    for r in mapping.iter_rows(named=True):
        label = r["display_label"]
        path_sig = panel.filter(pl.col("display_label") == label).filter(pl.col("sig"))
        path_dirs: set[str] = set(path_sig["direction"].unique().to_list())
        path_platforms: set[str] = set(path_sig["platform"].unique().to_list())

        for gene in r["lead_genes_in_figure"].split(";"):
            cells = gene_protein_data.get(gene, [])
            if not cells:
                continue
            on_path_cells = [c for c in cells if c["platform"] in path_platforms]
            sort_min = (
                min(on_path_cells, key=lambda c: c["fdr"])["fdr"]
                if on_path_cells
                else min(cells, key=lambda c: c["fdr"])["fdr"]
            )
            gene_sig_dirs = {
                c["direction"] for c in on_path_cells if c["fdr"] < sig_threshold
            }
            matched = gene_sig_dirs & path_dirs
            if not matched:
                direction = None
            elif matched == {"Up", "Down"}:
                direction = "Mixed"
            else:
                direction = next(iter(matched))
            out[(label, gene)] = {
                "fdr": sort_min,
                "direction": direction,
                "matches_pathway": direction is not None,
            }
    return out


# ---------------------------------------------------------------------------
# Pathway plotting
# ---------------------------------------------------------------------------


def signed_neglog(row: dict) -> float:
    nl = min(row["neglog10_fdr"], NEGLOG_CAP)
    sign = -1.0 if row["direction"] == "Down" else 1.0
    return sign * nl


def marker_size(n_genes: int) -> float:
    s = SIZE_SCALE * np.sqrt(max(n_genes, 1))
    return float(np.clip(s, SIZE_MIN, SIZE_MAX))


def plot_pathway_panel(
    ax,
    panel_data: pl.DataFrame,
    y_positions: dict[str, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """Render one platform panel (e.g., Olink) with all 32 pathways as dumbbells."""
    # Vertical guide at x=0
    ax.axvline(0, color="#AAAAAA", lw=0.5, zorder=0)
    # FDR=0.05 thresholds
    thr = -np.log10(FDR_THRESHOLD)
    for x_thr in (-thr, thr):
        ax.axvline(x_thr, color="#CCCCCC", lw=0.5, zorder=0)

    label_order = list(y_positions.keys())

    for label in label_order:
        sub = panel_data.filter(pl.col("display_label") == label)
        if sub.shape[0] == 0:
            continue
        y = y_positions[label]
        rows_by_week = {r["week"]: r for r in sub.iter_rows(named=True)}
        r24 = rows_by_week.get(24)
        r72 = rows_by_week.get(72)

        def to_pt(r):
            if r is None:
                return None
            x = signed_neglog(r)
            if r["sig"]:
                color = COLOR_UP if r["direction"] == "Up" else COLOR_DOWN
                size = marker_size(r["n_genes"])
            else:
                color = COLOR_NS
                size = SIZE_NS
            return (x, color, size, r["sig"])

        p24 = to_pt(r24)
        p72 = to_pt(r72)

        # Connecting dumbbell line
        if p24 is not None and p72 is not None:
            x24, c24, _, s24 = p24
            x72, c72, _, s72 = p72
            same_sig_dir = s24 and s72 and (x24 >= 0) == (x72 >= 0)
            line_color = c24 if same_sig_dir else COLOR_LINE
            line_alpha = 0.55 if same_sig_dir else 0.35
            ax.plot(
                [x24, x72], [y, y], color=line_color, lw=0.8, alpha=line_alpha, zorder=1
            )

        # wk24 marker: open circle
        if p24 is not None:
            x, c, s, sig = p24
            ax.scatter(
                [x],
                [y],
                s=s,
                facecolors="white",
                edgecolors=c,
                linewidths=1.0 if sig else 0.6,
                zorder=3 if sig else 2,
                marker="o",
            )
        # wk72 marker: filled circle
        if p72 is not None:
            x, c, s, sig = p72
            ax.scatter(
                [x],
                [y],
                s=s,
                facecolors=c,
                edgecolors="white",
                linewidths=0.6,
                zorder=3 if sig else 2,
                marker="o",
            )

    ax.format(
        xlim=xlim,
        ylim=ylim,
        xlabel="signed −log₁₀(FDR)",
    )
    ax.invert_yaxis()


def compute_y_positions(mapping: pl.DataFrame):
    """Assign y positions with tight intra-theme spacing + inter-theme gap.

    Returns:
        y_positions: display_label -> y coordinate
        theme_ranges: theme_id -> {y_min, y_max, name}
        ylim: (top_pad, bottom_pad) tuple
    """
    y_positions: dict[str, float] = {}
    theme_ranges: dict[int, dict] = {}
    y = 0.0
    prev_theme = None
    for r in mapping.iter_rows(named=True):
        if prev_theme is not None and r["theme_id"] != prev_theme:
            y += INTER_THEME_GAP
        y_positions[r["display_label"]] = y
        if r["theme_id"] not in theme_ranges:
            theme_ranges[r["theme_id"]] = {
                "y_min": y,
                "y_max": y,
                "name": r["theme_name"],
            }
        else:
            theme_ranges[r["theme_id"]]["y_max"] = y
        y += PATHWAY_GAP
        prev_theme = r["theme_id"]
    y_max = y - PATHWAY_GAP
    ylim = (-0.5, y_max + 0.5)
    return y_positions, theme_ranges, ylim


def _y_data_to_fig(bb, y_data: float, ylim: tuple[float, float]) -> float:
    """Convert a data-y value to figure y coordinate, accounting for inverted axis."""
    y0_data, y1_data = ylim
    frac = (y_data - y0_data) / (y1_data - y0_data)  # 0 at top, 1 at bottom
    return bb.y1 - frac * (bb.y1 - bb.y0)


def add_theme_banners(
    fig, ax_left, ax_right, theme_ranges: dict, ylim: tuple[float, float]
):
    """Draw horizontal lightgray strips spanning theme pathway rows in the left gutter."""
    bb_left = ax_left.get_position()
    banner_x1 = bb_left.x0 - 0.005
    if THEME_BANNER_WIDTH is None:
        banner_x0 = 0.005  # full left gutter (standalone figure)
    else:
        banner_x0 = banner_x1 - THEME_BANNER_WIDTH
    banner_width = banner_x1 - banner_x0

    for theme_id in sorted(theme_ranges.keys()):
        r = theme_ranges[theme_id]
        y_top = _y_data_to_fig(bb_left, r["y_min"] - THEME_BANNER_PAD, ylim)
        y_bot = _y_data_to_fig(bb_left, r["y_max"] + THEME_BANNER_PAD, ylim)
        height = y_top - y_bot
        rect = mpatches.Rectangle(
            (banner_x0, y_bot),
            banner_width,
            height,
            facecolor="#EDEDED",
            edgecolor="none",
            transform=fig.transFigure,
            clip_on=False,
            zorder=0,
        )
        fig.patches.append(rect)


def add_pathway_labels(
    fig,
    ax_left,
    mapping: pl.DataFrame,
    y_positions: dict[str, float],
    ylim: tuple[float, float],
    theme_ranges: dict,
):
    """Place pathway labels in left gutter; theme names above each block.

    The y-axis is keyed by ``display_label`` (a stable join key), but the
    *rendered* label is the cleaned original database term resolved through
    ``_pathway_labels.pathway_label`` so figure text matches the published
    pathway names rather than our internal short labels.
    """
    bb = ax_left.get_position()
    label_x = bb.x0 - 0.008

    # display_label → rendered string, via the cleaned-original lookup
    label_lookup = {
        r["display_label"]: pathway_label(r["original_term"])
        for r in mapping.iter_rows(named=True)
    }

    for label, y_data in y_positions.items():
        rendered = label_lookup.get(label, label)
        fig_y = _y_data_to_fig(bb, y_data, ylim)
        fig.text(
            label_x, fig_y, rendered, fontsize=6, ha="right", va="center", color="#1a1a1a"
        )

    for theme_id in sorted(theme_ranges.keys()):
        r = theme_ranges[theme_id]
        y_data_above = r["y_min"] - THEME_NAME_OFFSET
        fig_y = _y_data_to_fig(bb, y_data_above, ylim)
        fig.text(
            label_x,
            fig_y,
            r["name"],
            fontsize=6.5,
            fontweight="bold",
            ha="right",
            va="center",
            color="#000000",
        )


def add_lead_gene_labels(
    fig,
    ax_right,
    mapping: pl.DataFrame,
    y_positions: dict[str, float],
    ylim: tuple[float, float],
    lead_gene_stats: dict[tuple[str, str], dict],
    sig_threshold: float = FDR_THRESHOLD,
    max_genes: int = 5,
):
    """Place lead-gene strings in the right margin, sorted by protein-level
    across-treatment significance and colored by direction of effect.
    """
    bb = ax_right.get_position()
    label_x = bb.x1 + LEAD_GENE_OFFSET

    base_props = dict(fontsize=5, fontstyle="italic", color=COLOR_NS)
    NS_FALLBACK = {"fdr": 1.0, "direction": None, "matches_pathway": False}

    for r in mapping.iter_rows(named=True):
        label = r["display_label"]
        if label not in y_positions:
            continue
        y_data = y_positions[label]
        fig_y = _y_data_to_fig(bb, y_data, ylim)

        genes = r["lead_genes_in_figure"].split(";")
        scored = [(g, lead_gene_stats.get((label, g), NS_FALLBACK)) for g in genes]
        scored.sort(key=lambda gf: gf[1]["fdr"])
        truncated = scored[:max_genes]

        renderer = fig.canvas.get_renderer()
        cur_x = label_x
        for i, (gene, stats) in enumerate(truncated):
            if i > 0:
                sep_props = dict(base_props)
                sep_props["fontstyle"] = "normal"
                t = fig.text(
                    cur_x,
                    fig_y,
                    ", ",
                    ha="left",
                    va="center_baseline",
                    **sep_props,
                )
                bbw = t.get_window_extent(renderer=renderer).width
                cur_x += bbw / (fig.dpi * fig.get_figwidth())
            gene_props = dict(base_props)
            if stats.get("matches_pathway", False):
                if stats["direction"] == "Up":
                    gene_props["color"] = COLOR_UP
                elif stats["direction"] == "Down":
                    gene_props["color"] = COLOR_DOWN
                elif stats["direction"] == "Mixed":
                    gene_props["color"] = COLOR_MIXED
            t = fig.text(
                cur_x,
                fig_y,
                gene,
                ha="left",
                va="center_baseline",
                **gene_props,
            )
            bbw = t.get_window_extent(renderer=renderer).width
            cur_x += bbw / (fig.dpi * fig.get_figwidth())

        # Truncation tail
        if len(scored) > max_genes:
            fig.text(
                cur_x,
                fig_y,
                f", +{len(scored) - max_genes}",
                ha="left",
                va="center_baseline",
                fontsize=5,
                fontstyle="italic",
                color="#888888",
            )


def add_pathway_platform_banner(fig, ax, label: str):
    bb = ax.get_position()
    banner_height = 0.022
    banner_y = bb.y1 + 0.005
    rect = mpatches.FancyBboxPatch(
        (bb.x0, banner_y),
        bb.x1 - bb.x0,
        banner_height,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(rect)
    mid_x = (bb.x0 + bb.x1) / 2
    fig.text(
        mid_x,
        banner_y + banner_height / 2,
        label,
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="center",
    )


# ===========================================================================
# COMBINED FIGURE
# ===========================================================================

# Shared font size across both pathway-legend blocks
LEGEND_FS = 6

# Uniform gap from a legend title to the first handle
TITLE_HANDLE_GAP_FIG = 0.004


def make_pathway_figure():
    """Pathway enrichment dot plots (Olink/SomaScan).

    Two platform columns share the curated-term y-axis: term labels on the far
    left, lead-gene labels on the far right.
    """
    init_figure_theme()

    fig, axs = uplt.subplots(
        [[1, 2]],
        figwidth=NAT_W2,
        figheight=7.0,
        share=False,
        wspace=("0.2in",),
        left="1.45in",
        right="1.55in",
        top="0.5in",
        bottom="0.6in",
    )
    axs.format(abc=False)

    mapping = pl.read_csv(MAPPING_CSV)
    long = load_camera_for_terms(mapping)
    panel = collapse_per_cell(long)
    gene_protein_data = load_gene_protein_data()
    lead_gene_stats = compute_lead_gene_stats(mapping, panel, gene_protein_data)
    y_positions, theme_ranges, ylim = compute_y_positions(mapping)

    olink_panel = panel.filter(pl.col("platform") == "olink")
    soma_panel = panel.filter(pl.col("platform") == "soma")

    def _xlim(sub: pl.DataFrame) -> tuple[float, float]:
        sig = sub.filter(pl.col("sig"))
        nl = float(sig["neglog10_fdr"].max()) if sig.shape[0] > 0 else 5.0
        nl = min(nl, NEGLOG_CAP)
        return (-(nl + 1.5), nl + 1.5)

    plot_pathway_panel(
        axs[0], olink_panel, y_positions, xlim=_xlim(olink_panel), ylim=ylim
    )
    plot_pathway_panel(
        axs[1], soma_panel, y_positions, xlim=_xlim(soma_panel), ylim=ylim
    )
    for ax in (axs[0], axs[1]):
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, right=False)

    fig.canvas.draw()
    add_pathway_platform_banner(fig, axs[0], "Olink")
    add_pathway_platform_banner(fig, axs[1], "SomaScan")
    add_theme_banners(fig, axs[0], axs[1], theme_ranges, ylim)
    add_pathway_labels(fig, axs[0], mapping, y_positions, ylim, theme_ranges)
    add_lead_gene_labels(fig, axs[1], mapping, y_positions, ylim, lead_gene_stats)

    halo = [withStroke(linewidth=3.0, foreground="white")]
    for ax in (axs[0], axs[1]):
        ax.text(
            0.02,
            0.012,
            "← SEMA > TZP",
            transform=ax.transAxes,
            fontsize=6,
            color=COLOR_DOWN,
            ha="left",
            va="bottom",
            path_effects=halo,
            zorder=5,
        )
        ax.text(
            0.98,
            0.012,
            "TZP > SEMA →",
            transform=ax.transAxes,
            fontsize=6,
            color=COLOR_UP,
            ha="right",
            va="bottom",
            path_effects=halo,
            zorder=5,
        )

    _add_olink_marker_strip(fig, axs[0])
    _add_lead_gene_legend(fig, axs[1])

    # No a/b panel letters: the Olink/SomaScan columns are the same analysis on
    # two platforms (labelled by the platform banners), not separate panels.
    return fig


def _add_olink_marker_strip(fig, ax_path_olink):
    """Marker convention strip right-aligned in the gap column to the left
    of the Olink pathway panel.
    """
    bb = ax_path_olink.get_position()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    # Y: match the rendered Olink x-axis label centre.
    xlabel_lbl = ax_path_olink.xaxis.get_label()
    lbl_bb = xlabel_lbl.get_window_extent(renderer=renderer)
    y_center = inv.transform((0, (lbl_bb.y0 + lbl_bb.y1) / 2))[1]

    # X: right edge aligns with pathway label text
    label_right_x = bb.x0 - 0.008
    STRIP_W = 0.105
    STRIP_H = 0.020
    strip_ax = fig.add_axes(
        [
            label_right_x - STRIP_W,
            y_center - STRIP_H / 2,
            STRIP_W,
            STRIP_H,
        ]
    )
    strip_ax.set_axis_off()
    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(0, 1)

    fs = LEGEND_FS
    strip_w_in = STRIP_W * fig.get_figwidth()
    title_gap_strip = TITLE_HANDLE_GAP_FIG / STRIP_W

    def _tw(s, fontsize=fs, **kwargs):
        t = strip_ax.text(0, -10, s, fontsize=fontsize, **kwargs)
        bb_ = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb_.width / (fig.dpi * strip_w_in)

    SIG_S, NS_S = 14, 14
    sig_d = (SIG_S**0.5 / 72) / strip_w_in
    ns_d = (NS_S**0.5 / 72) / strip_w_in

    HANDLE_LABEL_GAP = 0.006
    INTER_ITEM_GAP = 0.012
    WEEK_NS_GAP = 0.024  # extra spacing before the NS entry

    w_sig = _tw("Sig:", fontweight="bold")
    w_w24 = _tw("Week 24,")
    w_w72 = _tw("Week 72,")
    w_ns = _tw("NS")

    total = (
        w_sig
        + title_gap_strip
        + sig_d
        + HANDLE_LABEL_GAP
        + w_w24
        + INTER_ITEM_GAP
        + sig_d
        + HANDLE_LABEL_GAP
        + w_w72
        + WEEK_NS_GAP
        + ns_d
        + HANDLE_LABEL_GAP
        + w_ns
    )
    cx = 0.99 - total

    # "Sig:" bold
    strip_ax.text(
        cx,
        0.5,
        "Sig:",
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="left",
        color="#444444",
    )
    cx += w_sig + title_gap_strip

    # open circle Week 24 — smaller s to compensate for the thicker edge
    # stroke so it appears the same visual size as the filled Week 72 circle.
    strip_ax.scatter(
        [cx + sig_d / 2],
        [0.5],
        s=SIG_S * 0.72,
        facecolors="white",
        edgecolors="#444444",
        linewidths=0.7,
        marker="o",
        clip_on=False,
    )
    cx += sig_d + HANDLE_LABEL_GAP
    strip_ax.text(
        cx, 0.5, "Week 24,", fontsize=fs, va="center", ha="left", color="#444444"
    )
    cx += w_w24 + INTER_ITEM_GAP

    # filled circle Week 72
    strip_ax.scatter(
        [cx + sig_d / 2],
        [0.5],
        s=SIG_S,
        facecolors="#444444",
        edgecolors="white",
        linewidths=0.4,
        marker="o",
        clip_on=False,
    )
    cx += sig_d + HANDLE_LABEL_GAP
    strip_ax.text(
        cx, 0.5, "Week 72,", fontsize=fs, va="center", ha="left", color="#444444"
    )
    cx += w_w72 + WEEK_NS_GAP

    # NS dot (same size as Week 72 filled circle)
    strip_ax.scatter(
        [cx + ns_d / 2],
        [0.5],
        s=NS_S,
        facecolors=COLOR_NS,
        edgecolors="none",
        marker="o",
        clip_on=False,
    )
    cx += ns_d + HANDLE_LABEL_GAP
    strip_ax.text(cx, 0.5, "NS", fontsize=fs, va="center", ha="left", color="#666666")


def _add_lead_gene_legend(fig, ax_path_soma):
    """Compact 2-row lead-gene legend in the right margin:

      Row 1 (top):    n: *5  *20  *50            (bubble size scale)
      Row 2 (bottom): Lead genes:  upTZP, upSEMA (title + direction colors)

    Bottom of row 2 aligns with the bottom of the pathway x-axis labels.
    """
    bb = ax_path_soma.get_position()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    x = bb.x1 + LEAD_GENE_OFFSET

    # Anchor: bottom of bottom row = bottom of soma xlabel text.
    xlabel_lbl = ax_path_soma.xaxis.get_label()
    lbl_bb = xlabel_lbl.get_window_extent(renderer=renderer)
    lbl_h_frac = (lbl_bb.y1 - lbl_bb.y0) / (fig.dpi * fig.get_figheight())
    y_lbl_bottom = inv.transform((0, lbl_bb.y0))[1]
    bottom_y = y_lbl_bottom + lbl_h_frac / 2  # centre of bottom row

    ROW = 0.016
    bubble_y = bottom_y + ROW

    FS = LEGEND_FS

    # ── Row 1: bubble-size scale "Pathway genes: *5  *20  *50" ────────────
    BSTRIP_H = 0.020
    BSTRIP_W = 0.115
    bubble_strip = fig.add_axes([x, bubble_y - BSTRIP_H / 2, BSTRIP_W, BSTRIP_H])
    bubble_strip.set_axis_off()
    bubble_strip.set_xlim(0, 1)
    bubble_strip.set_ylim(0, 1)

    strip_w_in = BSTRIP_W * fig.get_figwidth()
    title_gap_strip = TITLE_HANDLE_GAP_FIG / BSTRIP_W

    def _btw(s, **kwargs):
        t = bubble_strip.text(0, -10, s, fontsize=FS, **kwargs)
        w = t.get_window_extent(renderer=renderer).width / (fig.dpi * strip_w_in)
        t.remove()
        return w

    title_label = "Pathway genes"
    w_title = _btw(title_label, fontweight="bold")
    w_colon = _btw(":")
    bubble_strip.text(
        0.0,
        0.5,
        title_label,
        fontsize=FS,
        fontweight="bold",
        va="center",
        ha="left",
        color="#444444",
    )
    bubble_strip.text(
        w_title, 0.5, ":", fontsize=FS, va="center", ha="left", color="#444444"
    )

    BUBBLE_LABEL_GAP = 0.006
    INTER_BUBBLE_GAP = 0.025

    ns = [5, 20, 50]
    sizes_pt2 = [marker_size(n) for n in ns]
    radii_strip = [((s**0.5) / 72 / 2) / strip_w_in for s in sizes_pt2]
    label_widths = [_btw(str(n)) for n in ns]

    cursor = w_title + w_colon + title_gap_strip
    for n, s, r, lw in zip(ns, sizes_pt2, radii_strip, label_widths):
        bx = cursor + r
        bubble_strip.scatter(
            [bx],
            [0.5],
            s=s,
            facecolors="#888888",
            edgecolors="white",
            linewidths=0.4,
            marker="o",
            alpha=0.85,
            clip_on=False,
        )
        label_x = bx + r + BUBBLE_LABEL_GAP
        bubble_strip.text(
            label_x, 0.5, str(n), fontsize=FS, va="center", ha="left", color="#444444"
        )
        cursor = label_x + lw + INTER_BUBBLE_GAP

    # ── Row 2: "Lead genes:" (bold) + upTZP, upSEMA (colored, italic) ────
    title_artist = fig.text(
        x,
        bottom_y,
        "Lead genes:",
        fontsize=FS,
        fontweight="bold",
        color="#444444",
        ha="left",
        va="center",
    )
    fig.canvas.draw()
    title_w = title_artist.get_window_extent(renderer=renderer).width / (
        fig.dpi * fig.get_figwidth()
    )
    draw_segments_left_aligned(
        fig,
        x + title_w + TITLE_HANDLE_GAP_FIG,
        bottom_y,
        [
            ("TZP > SEMA", COLOR_UP),
            (", ", "#444444"),
            ("SEMA > TZP", COLOR_DOWN),
        ],
        fontsize=FS,
        fontstyle="italic",
    )


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1]
    fig = make_pathway_figure()
    save_figure(fig, str(out_dir / "fig3_pathway"), formats=("pdf",))
    print("Wrote fig3_pathway.pdf")
