"""Figure 3 — pathway enrichment dot plots (cameraPR), Olink + SomaScan."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import polars as pl

from analysis.design import EARLY_WEEK, LATE_WEEK, PLATFORMS, POST_BASELINE_WEEKS
from analysis.mmrm_outputs import across_treatment_csv, pathway_csv


from ._common import (
    COLOR_DOWN,
    COLOR_UP,
    COLOR_NS,
    FDR_THRESHOLD,
    NAT_W2,
    draw_segments_left_aligned,
    init_figure_theme,
    load_marker_gene_pairs,
    panel_figure,
    panel_grid,
)
from ._disease_reversal_figure import make_reversal_figure
from ._figure_bundle import main_fit_problem
from analysis.mmrm_outputs import PATHWAY_NUMERIC_SCHEMAS
from analysis.pathway_labels import SOURCE_NAMES, pathway_label
from ._theme import FS_MIN, FS_PANEL, GRAY_MID

# ===========================================================================
# PATHWAY SECTION
# ===========================================================================

COLOR_MIXED = "#7B2D8E"  # purple — gene sig in opposite directions across platforms
COLOR_LINE = "#888888"

FIGURES_DIR = Path(__file__).resolve().parents[1]

MAPPING_CSV = FIGURES_DIR / "configs" / "pathway_figure_terms.csv"

CAMERA_FILES = {
    (platform, week): pathway_csv(platform, week, "camera")
    for platform in PLATFORMS
    for week in POST_BASELINE_WEEKS
}

# Marker size scaling: sqrt(N_genes) * SIZE_SCALE, clamped to [SIZE_MIN, SIZE_MAX]
SIZE_SCALE = 9.0
SIZE_MIN = 22.0
SIZE_MAX = 140.0
SIZE_NS = 14.0  # constant small size for non-significant markers

# Upper limit of the x axis in -log10(FDR).
NEGLOG_CAP = 12.0

# Y-spacing
PATHWAY_GAP = 0.34  # unit between consecutive pathways in the same theme
INTER_THEME_GAP = 0.5  # extra vertical gap between themes (kept wide so themes
# stay visually grouped even as intra-theme rows are tightened)
THEME_NAME_OFFSET = 0.30  # theme-name y-data offset above first pathway in group

# Banner geometry
THEME_BANNER_PAD = 0.2  # data-units of vertical padding above/below pathway rows
LEAD_GENE_OFFSET = (
    0.002  # figure-fraction offset from pathway-right axis to lead-gene labels
)

# Fig 3a page geometry (inches). Figure height follows the term list at a fixed physical
# row pitch (``FIG3_IN_PER_Y_UNIT``); gutters hold the term labels (left) and the lead-gene
# labels (right); the top and bottom margins hold the platform banners and the x-axis label.
FIG3_LEFT_IN = 2.17
FIG3_RIGHT_IN = 1.7
FIG3_WSPACE_IN = 0.05
FIG3_TOP_IN = 0.5
FIG3_BOTTOM_IN = 0.6
# 5.9 / 14.66 in per y unit keeps Fig 3a and the 85 mm Fig 3b panel within the 247 mm page
# (``check_composite_height``). More than about 30 terms would need a smaller value.
FIG3_IN_PER_Y_UNIT = 5.9 / 14.66

# Protein-level data files for lead-gene direction coloring
_PROTEIN_FILES = {
    (platform, week): across_treatment_csv(platform, week)
    for platform in PLATFORMS
    for week in POST_BASELINE_WEEKS
}


# ---------------------------------------------------------------------------
# Pathway data loading
# ---------------------------------------------------------------------------


def load_camera_for_terms(mapping: pl.DataFrame) -> pl.DataFrame:
    """For each (term x platform x week), return its FDR/direction/N_genes if FDR<0.05.

    Returns long-format dataframe with columns:
      theme_id, display_label, original_term, database, platform, week, direction,
      nominal_p, fdr_q, neglog10_fdr, n_genes, total_genes, lead_genes_full, sig
    """
    frames = []
    for (platform, week), path in CAMERA_FILES.items():
        df = (
            pl.read_csv(path, schema_overrides=PATHWAY_NUMERIC_SCHEMAS["camera"])
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

    # Inner-join mapping to the camera rows (only the configured terms)
    long = mapping.join(camera, on=["database", "original_term"], how="inner")
    long = long.with_columns(
        (pl.col("fdr_q") < FDR_THRESHOLD).alias("sig"),
        (-(pl.col("fdr_q").log10())).alias("neglog10_fdr"),
    )
    return long


def collapse_per_cell(long: pl.DataFrame) -> pl.DataFrame:
    """For each (term x platform x week), keep one of cameraPR's per-direction rows.

    Prefer a significant row; otherwise select the row with the lowest FDR (``sig`` False).
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
    """For each gene_symbol, return per-cell between-treatment stats.

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


def figure_lead_genes(panel: pl.DataFrame) -> dict[str, list[str]]:
    """Each term's drawn lead genes: the cameraPR lead genes of its significant cells.

    The union of ``lead_genes_full`` over the term's cells with ``sig``, taken cell by cell in
    ascending pathway FDR (ties by platform, then week) and de-duplicated in first-seen order. ``add_lead_gene_labels`` then
    orders the drawn genes by protein FDR.

    Args:
        panel: ``collapse_per_cell`` output.

    Returns:
        ``display_label`` -> lead genes; a term with no significant cell is absent.
    """
    out: dict[str, list[str]] = {}
    ordered = panel.filter(pl.col("sig")).sort("display_label", "fdr_q", "platform", "week", maintain_order=True)
    for r in ordered.iter_rows(named=True):
        bucket = out.setdefault(r["display_label"], [])
        for gene in r["lead_genes_full"].split(";"):
            if gene and gene not in bucket:
                bucket.append(gene)
    return out


def compute_lead_gene_stats(
    mapping: pl.DataFrame,
    panel: pl.DataFrame,
    gene_protein_data: dict[str, list[dict]],
    sig_threshold: float = FDR_THRESHOLD,
) -> dict[tuple[str, str], dict]:
    """For each (display_label, gene) of ``figure_lead_genes(panel)``, compute:
    - fdr: gene's min protein-level FDR across all cells (sort order).
    - direction: matched direction (Up/Down/Mixed) when the gene is
      itself significant in the between-treatment contrast on a platform
      where the pathway is also significant; None otherwise.
    - matches_pathway: True iff such a co-occurring sig cell exists.
    """
    out: dict[tuple[str, str], dict] = {}
    lead_genes = figure_lead_genes(panel)
    for r in mapping.iter_rows(named=True):
        label = r["display_label"]
        path_sig = panel.filter(pl.col("display_label") == label).filter(pl.col("sig"))
        path_dirs: set[str] = set(path_sig["direction"].unique().to_list())
        path_platforms: set[str] = set(path_sig["platform"].unique().to_list())

        for gene in lead_genes.get(label, []):
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


def check_lead_gene_rule(mapping: pl.DataFrame, lead_gene_stats: dict[tuple[str, str], dict]) -> None:
    """Raise when a configured term breaks the Fig 3a display rule.

    The rule: a term is shown only if one of its lead genes is itself significant on a
    platform where the term is significant (``matches_pathway``, the colored lead-gene label).

    Args:
        mapping: The configured terms (``display_label`` column).
        lead_gene_stats: ``compute_lead_gene_stats`` output, keyed by ``(label, gene)``.

    Raises:
        ValueError: Naming every term of ``mapping`` that breaks the rule.
    """
    shown = {label for (label, _), s in lead_gene_stats.items() if s["matches_pathway"]}
    bad = [label for label in mapping["display_label"].to_list() if label not in shown]
    if bad:
        raise ValueError(
            f"terms with no lead gene significant on a platform where the term is significant: {bad}"
        )


# ---------------------------------------------------------------------------
# Pathway plotting
# ---------------------------------------------------------------------------


def signed_neglog(row: dict) -> float:
    """The dot's x position: -log10 FDR, capped at ``NEGLOG_CAP``, negative for a ``Down`` set."""
    nl = min(row["neglog10_fdr"], NEGLOG_CAP)
    sign = -1.0 if row["direction"] == "Down" else 1.0
    return sign * nl


def marker_size(n_genes: int) -> float:
    """Dot area in pt^2: ``SIZE_SCALE * sqrt(n_genes)`` (at least 1 gene), clipped to [``SIZE_MIN``, ``SIZE_MAX``]."""
    s = SIZE_SCALE * np.sqrt(max(n_genes, 1))
    return float(np.clip(s, SIZE_MIN, SIZE_MAX))


def plot_pathway_panel(
    ax,
    panel_data: pl.DataFrame,
    y_positions: dict[str, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """Render one platform panel (e.g., Olink) with every configured term as a dumbbell."""
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
        r24 = rows_by_week.get(EARLY_WEEK)
        r72 = rows_by_week.get(LATE_WEEK)

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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("signed −log₁₀(FDR)")
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
    fig,
    ax_left,
    ax_right,
    theme_ranges: dict,
    ylim: tuple[float, float],
    strip_width: float | None = None,
):
    """Draw horizontal lightgray strips spanning theme pathway rows in the left gutter.

    ``strip_width`` is the strip's width as a fraction of the figure width; None fills the
    whole left gutter, so the strip sits behind the theme and pathway-term labels.
    """
    bb_left = ax_left.get_position()
    banner_x1 = bb_left.x0 - 0.005
    if strip_width is None:
        banner_x0 = 0.005  # full left gutter (standalone figure)
    else:
        banner_x0 = banner_x1 - strip_width
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
    source_suffix: bool = False,
):
    """Place pathway labels in left gutter; theme names above each block.

    The y-axis is keyed by ``display_label``; rendered names are resolved with
    ``analysis.pathway_labels.pathway_label`` from ``original_term``.

    With ``source_suffix`` each term carries a gray `` · <library>`` suffix
    (``SOURCE_NAMES``) right-aligned at the label edge; the term shifts left by
    the suffix width.
    """
    bb = ax_left.get_position()
    label_x = bb.x0 - 0.008

    # display_label → rendered string, via the cleaned-original lookup
    label_lookup = {
        r["display_label"]: pathway_label(r["original_term"])
        for r in mapping.iter_rows(named=True)
    }

    if source_suffix:
        source_lookup = {
            r["display_label"]: SOURCE_NAMES[r["database"]]
            for r in mapping.iter_rows(named=True)
        }
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        px_per_fig_x = fig.dpi * fig.get_figwidth()

    for label, y_data in y_positions.items():
        rendered = label_lookup.get(label, label)
        fig_y = _y_data_to_fig(bb, y_data, ylim)
        term_x = label_x
        va = "center"
        if source_suffix:
            va = "center_baseline"
            suffix = fig.text(
                label_x,
                fig_y,
                f" · {source_lookup[label]}",
                fontsize=FS_MIN,
                ha="right",
                va=va,
                color=GRAY_MID,
            )
            term_x = label_x - suffix.get_window_extent(renderer).width / px_per_fig_x
        fig.text(term_x, fig_y, rendered, fontsize=6, ha="right", va=va, color="#1a1a1a")

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
    lead_genes: dict[str, list[str]],
    sig_threshold: float = FDR_THRESHOLD,
    max_genes: int = 5,
):
    """Place lead-gene strings in the right margin, sorted by protein-level
    between-treatment significance and colored by direction of effect.

    ``lead_genes`` is ``figure_lead_genes(panel)``.
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

        genes = lead_genes.get(label, [])
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
    """Draw a gray banner with ``label`` in bold across the top of ``ax``, in figure coordinates."""
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
    """Pathway enrichment dot plots (Olink/SomaScan), Fig 3a.

    Two platform columns share the curated-term y-axis: term labels on the far
    left, lead-gene labels on the far right. The height follows the number of
    terms and themes (``FIG3_IN_PER_Y_UNIT``); every term must pass
    ``check_lead_gene_rule``.
    """
    init_figure_theme()

    mapping = pl.read_csv(MAPPING_CSV)
    long = load_camera_for_terms(mapping)
    panel = collapse_per_cell(long)
    gene_protein_data = load_gene_protein_data()
    lead_gene_stats = compute_lead_gene_stats(mapping, panel, gene_protein_data)
    check_lead_gene_rule(mapping, lead_gene_stats)
    y_positions, theme_ranges, ylim = compute_y_positions(mapping)

    fig = panel_figure(
        NAT_W2, FIG3_TOP_IN + FIG3_BOTTOM_IN + FIG3_IN_PER_Y_UNIT * (ylim[1] - ylim[0])
    )
    axs = panel_grid(
        fig,
        [[1, 2]],
        left=FIG3_LEFT_IN,
        right=FIG3_RIGHT_IN,
        top=FIG3_TOP_IN,
        bottom=FIG3_BOTTOM_IN,
        wspace=FIG3_WSPACE_IN,
    )

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
    add_pathway_labels(
        fig, axs[0], mapping, y_positions, ylim, theme_ranges, source_suffix=True
    )
    add_lead_gene_labels(
        fig, axs[1], mapping, y_positions, ylim, lead_gene_stats, figure_lead_genes(panel)
    )

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
            zorder=5,
        )

    _add_olink_marker_strip(fig, axs[0])
    _add_lead_gene_legend(fig, axs[1])

    # No a/b panel letters: the Olink/SomaScan columns are the same analysis on
    # two platforms (labeled by the platform banners), not separate panels.
    return fig


def _add_olink_marker_strip(fig, ax_path_olink):
    """Marker convention strip right-aligned in the gap column to the left
    of the Olink pathway panel.
    """
    bb = ax_path_olink.get_position()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    # Y: match the rendered Olink x-axis label center.
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
    w_w24 = _tw(f"Week {EARLY_WEEK},")
    w_w72 = _tw(f"Week {LATE_WEEK},")
    w_ns = _tw("ns")

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
        cx, 0.5, f"Week {EARLY_WEEK},", fontsize=fs, va="center", ha="left", color="#444444"
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
        cx, 0.5, f"Week {LATE_WEEK},", fontsize=fs, va="center", ha="left", color="#444444"
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
    strip_ax.text(cx, 0.5, "ns", fontsize=fs, va="center", ha="left", color="#666666")


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
    bottom_y = y_lbl_bottom + lbl_h_frac / 2  # center of bottom row

    FS = LEGEND_FS
    ns = [5, 20, 50]

    # Row spacing in inches, so it holds at any figure height: the largest bubble's radius,
    # half the text height and a 1.5 pt gap keep the bubbles clear of the "Lead genes" row.
    r_max_pt = marker_size(max(ns)) ** 0.5 / 2
    ROW = (r_max_pt + FS / 2 + 1.5) / 72 / fig.get_figheight()
    bubble_y = bottom_y + ROW

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


# ===========================================================================
# FIGURE 3 COMPOSITE: panel a over panel b
# ===========================================================================
# Each panel is its own 183 mm wide figure with its letter at the top-left: panel a's height
# follows its term list (``make_pathway_figure``), panel b is 85 mm
# (``_disease_reversal_figure``). ``save_figure_stack`` stacks them edge to edge into one
# PDF/PNG, which must fit the 247 mm main-figure page depth.

# Panel letter position, figure fraction from each panel's top-left corner. The panels have
# no outer margin to hang a letter in, so ``_annotations.place_panel_letter`` (outside the
# axes, bottom-aligned, right-aligned) would not land here.
LETTER_XY = (0.004, 0.996)
PT_PER_INCH = 72.0


def check_composite_height(heights_in: list[float]) -> float:
    """Total height of panels stacked edge to edge, in inches.

    Args:
        heights_in: Panel heights, in inches.

    Returns:
        The stack height.

    Raises:
        ValueError: When the stack is deeper than the main-figure page (``main_fit_problem``).
    """
    total = sum(heights_in)
    problem = main_fit_problem(NAT_W2 * PT_PER_INCH, total * PT_PER_INCH)
    if problem:
        raise ValueError(f"Fig 3 stack: {problem}")
    return total


def add_panel_letter(fig, letter: str):
    """The 8 pt bold panel letter at the figure's top-left corner."""
    return fig.text(*LETTER_XY, letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top", color="black")


def make_figure3() -> list:
    """Panels a and b, lettered, in stacking order."""
    figs = [make_pathway_figure(), make_reversal_figure()]
    for fig, letter in zip(figs, "ab"):
        add_panel_letter(fig, letter)
    check_composite_height([fig.get_figheight() for fig in figs])
    return figs
