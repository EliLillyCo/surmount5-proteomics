"""Supplementary Fig. 8: the ORA counterpart of the Fig 3a cameraPR figure.

Layout (panel-c-equivalent only — no volcanos):
  +---------------------+---------------------+
  | Olink Explore HT    | SomaScan 11K        |
  | (ORA pathway dots)  | (ORA pathway dots)  |
  +---------------------+---------------------+
  Theme banners on the left; pathway names in the middle column;
  lead-gene strings in the right margin.

Differences from Fig 3a:
  * x-axis is unsigned −log10(adjusted P) — ORA has no direction.
  * Marker area encodes the overlap FRACTION X/Y of the ORA "X/Y" field: X, the
    significant genes in the set; Y, the set's genes measured on the platform (the
    background). Area = SIZE_SCALE * sqrt(100 * X/Y) pt², clipped to [SIZE_MIN, SIZE_MAX].
  * Pathway dots are monochrome (light gray when NS, dark green when sig);
    direction info is preserved at gene resolution in the lead-gene labels.
  * Lead-gene candidates come from the ORA overlap in pathway-significant cells;
    only genes with an assay at FDR<0.05 on a pathway-significant platform
    are labeled. Labels are ranked by lowest assay FDR on those platforms
    and limited to five per pathway.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Imports from shared modules
# ---------------------------------------------------------------------------

from figures._internal._common import (
    ROOT,
    c,
    init_figure_theme,
    panel_figure,
    panel_grid,
    save_figure,
)

from analysis.mmrm_outputs import pathway_csv
from analysis.design import EARLY_WEEK, LATE_WEEK, PLATFORMS, POST_BASELINE_WEEKS
from analysis.mmrm_outputs import PATHWAY_NUMERIC_SCHEMAS
import figures._internal._pathway_figure as P
from figures._internal._pathway_figure import (
    COLOR_DOWN,
    COLOR_MIXED,
    COLOR_NS,
    COLOR_UP,
    FDR_THRESHOLD,
    NEGLOG_CAP,
    SIZE_NS,
    _y_data_to_fig,
    add_pathway_labels,
    add_pathway_platform_banner,
    add_theme_banners,
    compute_y_positions,
    load_gene_protein_data,
)

# Theme-banner width sized to the widest row, "Complement & Coagulation Cascades · Reactome"
# (term plus gray library suffix, 6 pt and 5 pt), with the 0.048 in left margin past the
# label edge that Fig 3a has.
# 2.06 in, as a fraction of the figure width.
THEME_BANNER_WIDTH = 2.06 / P.NAT_W2

# Shared font size across the legend strips (as in Fig 3a).
LEGEND_FS = 6
TITLE_HANDLE_GAP_FIG = 0.004
# Lilly "bold_green" (web palette, as in the other figures) sets the ORA dots apart from Fig 3a's navy and gray.
DARK_SIG_COLOR = c["bold_green"]
NS_FALLBACK = {"fdr": 1.0, "direction": None, "matches_pathway": False}


def marker_size_fraction(frac: float) -> float:
    """Scale marker area from overlap fraction (X / measured gene-set size Y).

    The area scales with the square root of the percentage and is clipped to
    ``P.SIZE_MIN`` and ``P.SIZE_MAX``.
    """
    s = P.SIZE_SCALE * np.sqrt(max(frac, 0.0) * 100)
    return float(np.clip(s, P.SIZE_MIN, P.SIZE_MAX))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MAPPING_CSV = ROOT / "figures" / "configs" / "pathway_ora_figure_terms.csv"

ORA_CELLS = {
    (platform, week): pathway_csv(platform, week, "ora")
    for platform in PLATFORMS
    for week in POST_BASELINE_WEEKS
}


def _parse_overlap_xy(s: str) -> tuple[int, int]:
    """Parse the 'X/Y' overlap field. Returns (X, Y)."""
    x, y = s.split("/")
    return int(x), int(y)


def load_ora_for_terms(mapping: pl.DataFrame) -> pl.DataFrame:
    """Load all 4 ORA cells, filter to curated terms, return long-form panel
    with EVERY (curated term × platform × week) combination present — cells
    where the pathway has no ORA overlap are filled with NS placeholders so
    that downstream plotting always sees both Wk24 and Wk72 rows. A curated term that
    appears in no cell at all raises.

    Returns columns:
      display_label, theme_id, theme_name, original_term, database,
      platform, week, fdr, neglog10_fdr, overlap_count, overlap_total,
      overlap_fraction, sig, overlap_genes
    """
    curated = mapping.select(
        ["original_term", "database", "display_label", "theme_id", "theme_name"]
    )

    frames = []
    for (platform, week), path in ORA_CELLS.items():
        df = pl.read_csv(path, schema_overrides=PATHWAY_NUMERIC_SCHEMAS["ora"]).select(
            pl.col("Term").alias("original_term"),
            pl.col("Gene_Set_Database").alias("database"),
            pl.col("Adjusted P-value").alias("fdr"),
            pl.col("Overlap").alias("_overlap_str"),
            pl.col("Genes").alias("overlap_genes"),
        )
        df = df.with_columns(
            pl.lit(platform).alias("platform"),
            pl.lit(week).alias("week"),
            pl.col("_overlap_str")
            .map_elements(lambda s: _parse_overlap_xy(s)[0], return_dtype=pl.Int64)
            .alias("overlap_count"),
            pl.col("_overlap_str")
            .map_elements(lambda s: _parse_overlap_xy(s)[1], return_dtype=pl.Int64)
            .alias("overlap_total"),
        ).drop("_overlap_str")
        frames.append(df)

    long = pl.concat(frames)
    long = long.with_columns(
        (-pl.col("fdr").clip(1e-300).log10()).alias("neglog10_fdr"),
        (pl.col("fdr") < FDR_THRESHOLD).alias("sig"),
        (pl.col("overlap_count") / pl.col("overlap_total")).alias("overlap_fraction"),
    )

    missing = curated.join(
        long.select("original_term", "database").unique(),
        on=["original_term", "database"],
        how="anti",
    )
    if missing.height:
        raise ValueError(
            "curated ORA terms absent from every cell: "
            f"{missing.select('database', 'original_term').rows()}"
        )

    # Build full (curated × cell) grid so every panel-row × cell has a record.
    cells_df = pl.DataFrame(
        {
            "platform": [p for (p, w) in ORA_CELLS],
            "week": [w for (p, w) in ORA_CELLS],
        }
    )
    grid = curated.join(cells_df, how="cross")
    out = grid.join(
        long, on=["original_term", "database", "platform", "week"], how="left"
    )
    # Fill placeholders for "tested but no overlap" cells.
    out = out.with_columns(
        pl.col("fdr").fill_null(1.0),
        pl.col("neglog10_fdr").fill_null(0.0),
        pl.col("overlap_count").fill_null(0),
        pl.col("overlap_total").fill_null(0),
        pl.col("overlap_fraction").fill_null(0.0),
        pl.col("sig").fill_null(False),
        pl.col("overlap_genes").fill_null(""),
    )
    return out


def compute_lead_gene_pool(panel: pl.DataFrame) -> dict[str, list[str]]:
    """For each pathway (display_label), build the union of overlap genes
    drawn only from cells where the pathway is itself significant (FDR<0.05).

    Order: smallest pathway-FDR cells first, so the genes driving the
    strongest hit appear at the head of the list.
    """
    out: dict[str, list[str]] = {}
    ordered = panel.filter(pl.col("sig")).sort(["display_label", "fdr", "platform", "week"])
    for r in ordered.iter_rows(named=True):
        if r["overlap_genes"] is None or r["overlap_genes"] == "":
            continue
        genes = [g for g in r["overlap_genes"].split(";") if g]
        bucket = out.setdefault(r["display_label"], [])
        for g in genes:
            if g not in bucket:
                bucket.append(g)
    return out


def compute_lead_gene_stats_ora(
    panel: pl.DataFrame,
    gene_protein_data: dict[str, list[dict]],
    sig_threshold: float = FDR_THRESHOLD,
) -> dict[tuple[str, str], dict]:
    """For each (pathway, gene) in the ORA overlap union, compute:
      - fdr: gene's min protein-level FDR across all 4 cells (sort order).
      - direction: Up / Down / Mixed / None.
      - matches_pathway: True iff the gene is significant in the contrast
        on a platform where the ORA pathway is itself significant.

    Constraint: a gene only counts as matching if it is significant in the
    between-treatment contrast on the SAME platform where the ORA pathway is
    significant. Other genes are left out of the lead-gene strings.
    """
    pool = compute_lead_gene_pool(panel)

    # Which platforms is each pathway significant on?
    path_platforms: dict[str, set[str]] = {}
    for r in panel.filter(pl.col("sig")).iter_rows(named=True):
        path_platforms.setdefault(r["display_label"], set()).add(r["platform"])

    out: dict[tuple[str, str], dict] = {}
    for label, genes in pool.items():
        plats = path_platforms.get(label, set())
        for g in genes:
            cells = gene_protein_data.get(g, [])
            # Sort key: gene's best FDR among cells on pathway-sig
            # platforms (falls back to global min if untested there).
            on_path = [c for c in cells if c["platform"] in plats]
            sort_min = (
                min(on_path, key=lambda c: c["fdr"])["fdr"]
                if on_path
                else min((c["fdr"] for c in cells), default=1.0)
            )
            sig_cells = [c for c in on_path if c["fdr"] < sig_threshold]
            if not sig_cells:
                out[(label, g)] = {
                    "fdr": sort_min,
                    "direction": None,
                    "matches_pathway": False,
                }
                continue
            dirs = {c["direction"] for c in sig_cells}
            direction = "Mixed" if len(dirs) > 1 else ("Up" if "Up" in dirs else "Down")
            out[(label, g)] = {
                "fdr": sort_min,
                "direction": direction,
                "matches_pathway": True,
            }
    return out


# ---------------------------------------------------------------------------
# Pathway-panel plot
# ---------------------------------------------------------------------------


def plot_ora_panel(
    ax,
    panel_data: pl.DataFrame,
    y_positions: dict[str, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """Render one platform's ORA panel. Dumbbells: ○ Wk24, ● Wk72.

    x = unsigned −log10(FDR); marker area = overlap fraction (``marker_size_fraction``).
    """
    ax.axvline(0, color="#AAAAAA", lw=0.5, zorder=0)
    x_thr = -np.log10(FDR_THRESHOLD)
    ax.axvline(x_thr, color="#CCCCCC", lw=0.5, zorder=0)

    # group by pathway → render dumbbell connecting Wk24 + Wk72
    for label, sub in panel_data.group_by("display_label", maintain_order=True):
        label = label[0] if isinstance(label, tuple) else label
        if label not in y_positions:
            continue
        y = y_positions[label]
        rows = {int(r["week"]): r for r in sub.iter_rows(named=True)}
        # Every (curated × cell) is guaranteed to be present by
        # load_ora_for_terms — missing cells render at x=0 as small NS dots.
        r24, r72 = rows[EARLY_WEEK], rows[LATE_WEEK]

        x24 = min(float(r24["neglog10_fdr"]), NEGLOG_CAP)
        x72 = min(float(r72["neglog10_fdr"]), NEGLOG_CAP)
        sig_both = r24["sig"] and r72["sig"]
        line_color = DARK_SIG_COLOR if sig_both else "#BBBBBB"
        line_alpha = 0.7 if sig_both else 0.4
        ax.plot(
            [x24, x72], [y, y], color=line_color, lw=0.8, alpha=line_alpha, zorder=1
        )

        for week, row in ((EARLY_WEEK, r24), (LATE_WEEK, r72)):
            x = min(float(row["neglog10_fdr"]), NEGLOG_CAP)
            sig = row["sig"]
            if sig:
                size = marker_size_fraction(float(row["overlap_fraction"]))
                facecolor = "white" if week == EARLY_WEEK else DARK_SIG_COLOR
                edgecolor = DARK_SIG_COLOR if week == EARLY_WEEK else "white"
                edgewidth = 0.7 if week == EARLY_WEEK else 0.4
            else:
                size = SIZE_NS
                facecolor = COLOR_NS
                edgecolor = "white"
                edgewidth = 0.3

            ax.scatter(
                [x],
                [y],
                s=size,
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidths=edgewidth,
                zorder=3 if sig else 2,
                marker="o",
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("−log₁₀(FDR)")
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Lead-gene labels on the right margin
# ---------------------------------------------------------------------------


def add_lead_gene_labels_ora(
    fig,
    ax_right,
    panel: pl.DataFrame,
    mapping: pl.DataFrame,
    y_positions: dict[str, float],
    ylim: tuple[float, float],
    lead_gene_stats: dict,
):
    """Render the per-pathway lead-gene strings in the right margin.

    Genes are sorted by min protein-level FDR; significant + direction-matched
    genes are colored navy (Up=TZP > SEMA), gray (Down=SEMA > TZP) or purple (Mixed);
    the rest are dropped.
    """
    bb = ax_right.get_position()
    label_x = bb.x1 + P.LEAD_GENE_OFFSET

    pool = compute_lead_gene_pool(panel)
    base_props = dict(fontsize=5, fontstyle="italic", color=COLOR_NS)

    for r in mapping.iter_rows(named=True):
        label = r["display_label"]
        if label not in y_positions:
            continue
        y_data = y_positions[label]
        fig_y = _y_data_to_fig(bb, y_data, ylim)

        genes = pool.get(label, [])
        if not genes:
            continue
        scored = [(g, lead_gene_stats.get((label, g), NS_FALLBACK)) for g in genes]
        # Drop genes that aren't sig on a pathway-sig platform.
        scored = [s for s in scored if s[1].get("matches_pathway")]
        scored.sort(key=lambda gf: gf[1]["fdr"])

        # Truncate to keep the right margin readable.
        max_genes = 5
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
        if len(scored) > max_genes:
            t = fig.text(
                cur_x,
                fig_y,
                f", +{len(scored) - max_genes}",
                ha="left",
                va="center_baseline",
                fontsize=5,
                fontstyle="italic",
                color="#888888",
            )


# ---------------------------------------------------------------------------
# Legend strips (sig markers + lead-gene block), as in Fig 3a
# ---------------------------------------------------------------------------


def _add_olink_marker_strip(fig, ax_olink, ax_soma):
    """Sig-marker strip in the left margin, right-aligned to the right edge
    of the theme banner (= just left of the Olink panel). The Olink xlabel
    keeps its default centered position under the panel.
    """
    bb = ax_olink.get_position()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    # Use the soma panel's rendered xlabel for the y reference — it stays
    # present, so its window extent is meaningful.
    soma_xlabel = ax_soma.xaxis.get_label()
    xlbl_bb = soma_xlabel.get_window_extent(renderer=renderer)
    y_center = inv.transform((0, (xlbl_bb.y0 + xlbl_bb.y1) / 2))[1]

    # Strip right edge = theme-banner right edge (= bb.x0 - 0.005).
    # Strip width is computed below from total content width so items right-align.
    strip_x1 = bb.x0 - 0.005

    # Pre-measure all content in pixels using a throwaway fig text.
    def _measure(s, **kwargs):
        kwargs.setdefault("fontsize", LEGEND_FS)
        t = fig.text(0, -10, s, **kwargs)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return w  # pixels

    SIG_S, NS_S = 14, 14
    sig_d_px = ((SIG_S**0.5) / 72) * fig.dpi
    ns_d_px = ((NS_S**0.5) / 72) * fig.dpi
    fig_w_px = fig.dpi * fig.get_figwidth()

    GAP_TITLE_PX = TITLE_HANDLE_GAP_FIG * fig_w_px
    GAP_HANDLE_LABEL_PX = 0.006 * fig_w_px
    GAP_INTER_ITEM_PX = 0.012 * fig_w_px
    GAP_WEEK_NS_PX = 0.024 * fig_w_px  # extra spacing before NS entry

    w_sig_px = _measure("Sig:", fontweight="bold")
    w_w24_px = _measure(f"Week {EARLY_WEEK},")
    w_w72_px = _measure(f"Week {LATE_WEEK},")
    w_ns_px = _measure("ns")

    total_px = (
        w_sig_px
        + GAP_TITLE_PX
        + sig_d_px
        + GAP_HANDLE_LABEL_PX
        + w_w24_px
        + GAP_INTER_ITEM_PX
        + sig_d_px
        + GAP_HANDLE_LABEL_PX
        + w_w72_px
        + GAP_WEEK_NS_PX
        + ns_d_px
        + GAP_HANDLE_LABEL_PX
        + w_ns_px
    )
    STRIP_W = total_px / fig_w_px
    STRIP_H = 0.020
    strip_x0 = strip_x1 - STRIP_W
    strip_ax = fig.add_axes(
        [strip_x0, y_center - STRIP_H / 2, STRIP_W, STRIP_H],
    )
    strip_ax.set_axis_off()
    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(0, 1)

    fs = LEGEND_FS

    # Convert pixel widths/gaps to strip fraction
    pf = lambda px: px / total_px
    sig_d = pf(sig_d_px)
    ns_d = pf(ns_d_px)
    TG = pf(GAP_TITLE_PX)
    HLG = pf(GAP_HANDLE_LABEL_PX)
    ITG = pf(GAP_INTER_ITEM_PX)
    WNSG = pf(GAP_WEEK_NS_PX)

    cx = 0.0
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
    cx += pf(w_sig_px) + TG

    strip_ax.scatter(
        [cx + sig_d / 2],
        [0.5],
        s=SIG_S,
        facecolors="white",
        edgecolors="#444444",
        linewidths=0.7,
        marker="o",
        clip_on=False,
    )
    cx += sig_d + HLG
    strip_ax.text(
        cx, 0.5, f"Week {EARLY_WEEK},", fontsize=fs, va="center", ha="left", color="#444444"
    )
    cx += pf(w_w24_px) + ITG

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
    cx += sig_d + HLG
    strip_ax.text(
        cx, 0.5, f"Week {LATE_WEEK},", fontsize=fs, va="center", ha="left", color="#444444"
    )
    cx += pf(w_w72_px) + WNSG

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
    cx += ns_d + HLG
    strip_ax.text(cx, 0.5, "ns", fontsize=fs, va="center", ha="left", color="#666666")


def _add_lead_gene_legend(fig, ax_soma):
    """Right-margin legend: 'Overlap: ●5% ●20% ●50%' / 'Lead genes: TZP > SEMA, ...'.

    'Overlap' is the overlap fraction X/Y: the share of the pathway's measured genes that
    are significant proteins.
    """
    bb = ax_soma.get_position()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    x = bb.x1 + P.LEAD_GENE_OFFSET

    xlabel_lbl = ax_soma.xaxis.get_label()
    lbl_bb = xlabel_lbl.get_window_extent(renderer=renderer)
    lbl_h_frac = (lbl_bb.y1 - lbl_bb.y0) / (fig.dpi * fig.get_figheight())
    y_lbl_bottom = inv.transform((0, lbl_bb.y0))[1]
    bottom_y = y_lbl_bottom + lbl_h_frac / 2

    ROW = 0.016  # vertical spacing — wide enough to clear the bubble strip
    bubble_y = bottom_y + ROW

    FS = LEGEND_FS

    BSTRIP_H = 0.020
    BSTRIP_W = 0.105
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

    # "Overlap:" — bold label, regular colon
    title_label = "Overlap"
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
    INTER_BUBBLE_GAP = 0.022

    # Overlap-fraction legend reference values.
    fractions = [0.05, 0.20, 0.50]
    labels = ["5%", "20%", "50%"]
    sizes_pt2 = [marker_size_fraction(f) for f in fractions]
    radii_strip = [((s**0.5) / 72 / 2) / strip_w_in for s in sizes_pt2]
    label_widths = [_btw(lab) for lab in labels]

    cursor = w_title + w_colon + title_gap_strip
    for lab, s, r, lw in zip(labels, sizes_pt2, radii_strip, label_widths):
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
            label_x, 0.5, lab, fontsize=FS, va="center", ha="left", color="#444444"
        )
        cursor = label_x + lw + INTER_BUBBLE_GAP

    # Row 2: "Lead genes:" bold + " TZP > SEMA, SEMA > TZP" colored italic
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
    _draw_segments_left_aligned(
        fig,
        x + title_w + TITLE_HANDLE_GAP_FIG,
        bottom_y,
        [
            ("TZP > SEMA", COLOR_UP),
            (", ", "#444444"),
            ("SEMA > TZP", COLOR_DOWN),
            (", ", "#444444"),
            ("Mixed", COLOR_MIXED),
        ],
        fontsize=FS,
        fontstyle="italic",
    )


def _draw_segments_left_aligned(fig, x_left, y, segments, **kwargs):
    """Place text segments left-aligned starting at x_left, in fig coords."""
    renderer = fig.canvas.get_renderer()
    cur_x = x_left
    for text, color in segments:
        t = fig.text(
            cur_x,
            y,
            text,
            ha="left",
            va="center",
            color=color,
            **kwargs,
        )
        bbw = t.get_window_extent(renderer=renderer).width
        cur_x += bbw / (fig.dpi * fig.get_figwidth())


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure():
    """Build Supplementary Fig 8: ORA dot plots of the curated terms, Olink and SomaScan.

    Reads the curated terms (``figures/configs/pathway_ora_figure_terms.csv``), the four
    ORA result cells (platform x week) and the protein-level results for the lead genes.

    Returns:
        The figure.
    """
    init_figure_theme()

    mapping = pl.read_csv(MAPPING_CSV)
    panel = load_ora_for_terms(mapping)

    gene_protein_data = load_gene_protein_data()
    lead_gene_stats = compute_lead_gene_stats_ora(panel, gene_protein_data)

    y_positions, theme_ranges, ylim = compute_y_positions(mapping)

    olink_panel = panel.filter(pl.col("platform") == "olink")
    soma_panel = panel.filter(pl.col("platform") == "soma")

    def _xlim(sub: pl.DataFrame) -> tuple[float, float]:
        sig = sub.filter(pl.col("sig"))
        nl = float(sig["neglog10_fdr"].max()) if sig.shape[0] > 0 else 5.0
        nl = min(nl, NEGLOG_CAP)
        # Small negative pad keeps NS-placeholder dots (drawn at x=0) clear
        # of the left axis spine; right pad keeps the largest markers
        # visible at the extreme x.
        return (-0.3, nl + 1.4)

    # The figure height follows the term list at Fig 3a's physical row pitch
    # (P.FIG3_IN_PER_Y_UNIT; both figures use PATHWAY_GAP = 0.34).
    top_in, bottom_in = 0.4, 0.9
    fig = panel_figure(P.NAT_W2, top_in + bottom_in + P.FIG3_IN_PER_Y_UNIT * (ylim[1] - ylim[0]))
    axs = panel_grid(
        fig,
        [[1, 2]],
        left=2.4,  # theme banners + pathway labels in left margin
        right=1.4,  # lead-gene strings
        top=top_in,
        bottom=bottom_in,
        wspace=0.15,
    )

    plot_ora_panel(axs[0], olink_panel, y_positions, xlim=_xlim(olink_panel), ylim=ylim)
    plot_ora_panel(axs[1], soma_panel, y_positions, xlim=_xlim(soma_panel), ylim=ylim)
    for ax in (axs[0], axs[1]):
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, right=False)

    fig.canvas.draw()

    add_pathway_platform_banner(fig, axs[0], "Olink")
    add_pathway_platform_banner(fig, axs[1], "SomaScan")

    add_theme_banners(fig, axs[0], axs[1], theme_ranges, ylim, strip_width=THEME_BANNER_WIDTH)
    add_pathway_labels(
        fig, axs[0], mapping, y_positions, ylim, theme_ranges, source_suffix=True
    )
    add_lead_gene_labels_ora(
        fig, axs[1], panel, mapping, y_positions, ylim, lead_gene_stats
    )

    _add_olink_marker_strip(fig, axs[0], axs[1])
    _add_lead_gene_legend(fig, axs[1])

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_dir = Path(__file__).resolve().parent
    save_figure(fig, str(out_dir / "supp_fig_pathway_ora"), formats=("pdf",))
    print("Wrote supp_fig_pathway_ora.pdf")
