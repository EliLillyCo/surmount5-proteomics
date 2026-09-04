"""Trajectory heatmap: temporal response classification for proteomics.

4-column layout: TZP Wk24 | Wk72 | SEMA Wk24 | Wk72
Rows grouped by TZP trajectory class, sorted by magnitude within each class.
Values scaled per protein to [-1, 1].
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import polars as pl

from ._common import (
    ANALYSIS_DIR,
    ARM_COLORS,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_NARR,
    FS_PANEL,
    LILLY_COLORS,
    PALETTE_VARIANT,
    TRAJECTORY_COLORS,
    TRAJECTORY_ORDER,
    apply_lilly_theme,
    build_cmap,
    init_figure_theme,
    load_marker_to_gene,
    save_figure,
)

apply_lilly_theme()

import PyComplexHeatmap as pch  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

c = LILLY_COLORS[PALETTE_VARIANT]


def _mix(c1: str, c2: str, t: float = 0.5) -> str:
    r1, g1, b1 = mpl.colors.to_rgb(c1)
    r2, g2, b2 = mpl.colors.to_rgb(c2)
    return mpl.colors.to_hex(
        (
            r1 + (r2 - r1) * t,
            g1 + (g2 - g1) * t,
            b1 + (b2 - b1) * t,
        )
    )


COLUMN_ORDER = ["TZP_Wk24", "TZP_Wk72", "SEMA_Wk24", "SEMA_Wk72"]

# Row split gap between trajectory classes
ROW_SPLIT_GAP = 0.3

# Genes to label on the right edge of the heatmap. A curated spread across
# trajectory classes that gives the overview some texture, always including the
# exemplar proteins plotted in panel c (sourced from _trajectory_lines.PANELS)
# so the heatmap landscape cross-references the line plots below.
from ._trajectory_lines import PANELS as _LINE_PANELS

FIGURES_DIR = Path(__file__).resolve().parents[1]

_CURATED_LABELS: list[str] = [
    # Late-onset ↓
    "IL6",
    "ECHS1",
    "CCL18",
    "ACAT1",
    "IGFBP4",
    "PLIN1",
    # Progressive ↓
    "LEP",
    "INS",
    "CRP",
    "FGF21",
    "OXT",
    "C3",
    "LBP",
    # Sustained ↓
    "PCSK9",
    "GSTA1",
    "FABP1",
    "DLK1",
    "LDLR",
    "MSTN",
    "COL26A1",
    "APOC3",
    "CES1",
    # Transient ↓
    "NPPB",
    "HAVCR1",
    "ACE2",
    "APOA1",
    # Reversal ↓↑
    "IL1R1",
    "LPL",
    "GUCA2A",
    "ANGPTL3",
    # Reversal ↑↓
    "IL1RN",
    "FABP4",
    "FABP3",
    # Transient ↑
    "INHBB",
    "GAST",
    "CCL17",
    # Sustained ↑
    "PNLIPRP1",
    "CPA1",
    "REG1B",
    "PPY",
    "FGF19",
    "HAMP",
    "WFIKKN2",
    "GDF15",
    # Progressive ↑
    "IGFBP1",
    "IGFBP2",
    "SHBG",
    "ADIPOQ",
    "GH1",
    "HSD11B1",
    # Sustained ↑ (additional)
    "DMP1",
    # Late-onset ↑
    "APOA2",
    "TG",
    "PLTP",
    "HBA1",
    "CHL1",
]
HIGHLIGHT_GENES: list[str] = sorted(
    set(_CURATED_LABELS) | set(_LINE_PANELS["olink"]) | set(_LINE_PANELS["soma"])
)


def _fig_transform(fig):
    """Return the figure-relative-to-display transform that's correct for both
    Figure (uses transFigure) and SubFigure (uses transSubfigure — its own
    transFigure is actually the *outer* figure's transform).
    """
    return getattr(fig, "transSubfigure", fig.transFigure)


def _sqrt_compact_targets(
    class_sizes: dict[str, int], max_display: int
) -> dict[str, int]:
    """Compute per-class display targets using √-scaling, calibrated so the
    largest class displays at `max_display` rows. Smaller classes whose sqrt
    target would exceed their true size are left uncompacted.

    Only classes whose target is *meaningfully* smaller than their true size are
    compacted (and thus get the ``*`` marker); a trivial 1-2 row reduction is
    skipped so small blocks keep a plain count that fits in their banner.
    """
    import math

    if not class_sizes:
        return {}
    max_n = max(class_sizes.values())
    if max_n <= max_display:
        return {}
    scale = max_display / math.sqrt(max_n)
    targets: dict[str, int] = {}
    for traj, n in class_sizes.items():
        sqrt_target = max(1, int(round(math.sqrt(n) * scale)))
        if sqrt_target < 0.9 * n:
            targets[traj] = sqrt_target
    return targets


def load_and_prepare(
    platform: str = "olink",
    compact_classes: dict[str, int] | None = None,
    compact_max_display: int | None = None,
    preserve_genes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load classification results and build matrix + annotations.

    compact_classes: optional {class_name: target_n} to stride-subsample large
    classes for visual scaling. Returned `true_counts` always reflects the full
    class size so labels can show the original n with a compaction marker.
    compact_max_display: if `compact_classes` is None, derive targets via
    √-scaling so the largest class displays at this row count.
    """
    parquet_path = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "outputs"
        / f"trajectory_{platform}.parquet"
    )
    df = pl.read_parquet(parquet_path)

    non_null = df.filter(pl.col("trajectory_TZP") != "Null")

    non_null = non_null.with_columns(
        pl.max_horizontal(
            pl.col("PCBL_TZP_24").abs(),
            pl.col("PCBL_TZP_72").abs(),
            pl.col("PCBL_SEMA_24").abs(),
            pl.col("PCBL_SEMA_72").abs(),
        ).alias("max_abs_pcbl")
    )

    non_null = non_null.with_columns(
        (pl.col("PCBL_TZP_24") / pl.col("max_abs_pcbl")).alias("TZP_Wk24"),
        (pl.col("PCBL_TZP_72") / pl.col("max_abs_pcbl")).alias("TZP_Wk72"),
        (pl.col("PCBL_SEMA_24") / pl.col("max_abs_pcbl")).alias("SEMA_Wk24"),
        (pl.col("PCBL_SEMA_72") / pl.col("max_abs_pcbl")).alias("SEMA_Wk72"),
    )

    trajectory_rank = {name: i for i, name in enumerate(TRAJECTORY_ORDER)}
    non_null = non_null.with_columns(
        pl.col("trajectory_TZP")
        .cast(pl.Utf8)
        .replace_strict(
            {k: v for k, v in trajectory_rank.items()}, return_dtype=pl.Int32
        )
        .alias("traj_rank")
    )

    # Sort within each class by peak-timepoint magnitude
    sort_by_wk24 = {"Transient ↓", "Transient ↑"}
    sort_by_wk72 = {
        "Late-onset ↓",
        "Progressive ↓",
        "Sustained ↓",
        "Late-onset ↑",
        "Progressive ↑",
        "Sustained ↑",
        "Reversal ↑↓",
        "Reversal ↓↑",
    }

    if compact_classes is None and compact_max_display is not None:
        class_sizes = {
            row["trajectory_TZP"]: row["n"]
            for row in non_null.group_by("trajectory_TZP")
            .agg(pl.len().alias("n"))
            .iter_rows(named=True)
        }
        compact_classes = _sqrt_compact_targets(class_sizes, compact_max_display)
    compact_classes = compact_classes or {}

    # Resolve preserve_genes → set of marker IDs we must keep through compaction.
    preserve_markers: set[str] = set()
    if preserve_genes:
        marker_to_gene = load_marker_to_gene(platform)
        gene_set = set(preserve_genes)
        preserve_markers = {m for m, g in marker_to_gene.items() if g in gene_set}

    true_counts: dict[str, int] = {}
    frames = []
    for traj in TRAJECTORY_ORDER:
        subset = non_null.filter(pl.col("trajectory_TZP").cast(pl.Utf8) == traj)
        if traj in sort_by_wk24:
            if "↓" in traj:
                subset = subset.sort("PCBL_TZP_24", descending=False)
            else:
                subset = subset.sort("PCBL_TZP_24", descending=True)
        else:
            if "↓" in traj:
                subset = subset.sort("PCBL_TZP_72", descending=False)
            else:
                subset = subset.sort("PCBL_TZP_72", descending=True)

        true_counts[traj] = subset.height

        target = compact_classes.get(traj)
        if target is not None and subset.height > target:
            denom = max(1, target - 1)
            sampled = {
                int(round(i * (subset.height - 1) / denom)) for i in range(target)
            }
            # Force-include any rows whose marker is in preserve set; keep order.
            if preserve_markers:
                markers = subset["marker"].to_list()
                preserved_idx = {
                    i for i, mk in enumerate(markers) if mk in preserve_markers
                }
                sampled |= preserved_idx
            indices = sorted(sampled)
            subset = subset[indices]

        frames.append(subset)

    sorted_df = pl.concat(frames)

    matrix = sorted_df.select(["marker"] + COLUMN_ORDER).to_pandas().set_index("marker")
    matrix = matrix[COLUMN_ORDER]

    row_anno = pd.DataFrame(
        {"Trajectory": sorted_df["trajectory_TZP"].cast(pl.Utf8).to_list()},
        index=matrix.index,
    )

    col_anno = pd.DataFrame(
        {
            "Arm": ["TZP", "TZP", "SEMA", "SEMA"],
            "Visit": ["Wk24", "Wk72", "Wk24", "Wk72"],
        },
        index=COLUMN_ORDER,
    )

    return matrix, row_anno, col_anno, true_counts


def render_heatmap(
    matrix: pd.DataFrame,
    row_anno: pd.DataFrame,
    col_anno: pd.DataFrame,
    platform: str = "olink",
    true_counts: dict[str, int] | None = None,
    parent_fig=None,
    subplot_spec=None,
    show_legend: bool = True,
    panel_label: str | None = None,
    save: bool = True,
) -> None:
    """Render and save the trajectory heatmap.

    `true_counts` (optional): full per-class size before any visual subsampling.
    When provided, count text uses the true n with a `*` marker and a footnote
    is added at the bottom of the figure.
    `parent_fig`/`subplot_spec` (optional): if both provided, render the panel
    into the given subplot of the parent figure (for combined multi-panel
    layouts). When omitted, a fresh figure is created.
    `show_legend`: hide the trajectory legend + colorbar (e.g. for the first
    panel of a combined figure where only the second panel carries the legend).
    `panel_label`: optional "a"/"b" label drawn at the panel's upper-left.
    `save`: if False, skip saving the figure (caller saves the combined output).
    """
    import matplotlib.pyplot as plt

    apply_lilly_theme()

    arm_palette = ARM_COLORS
    visit_palette = {
        "Wk24": _mix(c["bold_grey"], "white", 0.55),
        "Wk72": c["bold_grey"],
    }

    displayed_counts = row_anno["Trajectory"].value_counts()
    true_counts = true_counts or {}

    # Use a unique placeholder kwarg so we can locate and hide the label post-render.
    side_label = "_traj_side_"

    traj_cat = pd.Categorical(
        row_anno["Trajectory"], categories=TRAJECTORY_ORDER, ordered=True
    )
    traj_series = pd.Series(traj_cat, index=row_anno.index, name="Trajectory")

    row_ha = pch.HeatmapAnnotation(
        **{
            side_label: pch.anno_simple(
                traj_series,
                colors=TRAJECTORY_COLORS,
                add_text=False,
                legend=show_legend,
                legend_kws={"title": "Trajectory", "labelcolor": "black"},
            )
        },
        axis=0,
        verbose=0,
        label_kws={"fontsize": 0},
    )

    col_ha = pch.HeatmapAnnotation(
        Arm=pch.anno_simple(
            col_anno["Arm"],
            colors=arm_palette,
            add_text=True,
            legend=False,
            text_kws={"fontsize": 7, "weight": "bold", "color": "white"},
        ),
        Visit=pch.anno_simple(
            col_anno["Visit"],
            colors=visit_palette,
            add_text=True,
            legend=False,
            text_kws={"fontsize": 6},
        ),
        axis=1,
        verbose=0,
        label_kws={"fontsize": 7, "weight": "bold"},
    )

    embedded = parent_fig is not None and subplot_spec is not None
    if embedded:
        fig = parent_fig
        panel_ax = fig.add_subplot(subplot_spec)
        panel_ax.set_axis_off()
    else:
        plt.figure(figsize=(5.5, 5.5))
        fig = plt.gcf()
        panel_ax = None

    # In embedded mode we render legends manually to avoid PCH's bottom-row
    # fallback when the right-side space is constrained by gridspec.
    pch_legend = show_legend and not embedded
    cm_kwargs = dict(
        data=matrix,
        left_annotation=row_ha,
        top_annotation=col_ha,
        row_cluster=False,
        col_cluster=False,
        col_split=col_anno["Arm"],
        col_split_gap=1.5,
        row_split=row_anno["Trajectory"],
        row_split_order=TRAJECTORY_ORDER,
        row_split_gap=ROW_SPLIT_GAP,
        cmap=build_cmap("Lilly_Diverging"),
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        show_rownames=False,
        show_colnames=False,
        label="Scaled PCBL",
        legend=pch_legend,
        plot_legend=pch_legend,
        legend_anchor="ax",
        legend_side="right",
        legend_hpad=15,
        legend_hgap=2,
        legend_vgap=1.5,
        legend_kws={"extend": "neither"},
        xticklabels_kws={"labelsize": 0},
        yticklabels_kws={"labelsize": 0},
    )
    # Suppress the categorical legend on the row strip in embedded mode too.
    if embedded:
        # Rebuild row_ha without legend to avoid PCH collecting its categorical legend.
        row_ha = pch.HeatmapAnnotation(
            **{
                side_label: pch.anno_simple(
                    traj_series,
                    colors=TRAJECTORY_COLORS,
                    add_text=False,
                    legend=False,
                )
            },
            axis=0,
            verbose=0,
            label_kws={"fontsize": 0},
        )
        cm_kwargs["left_annotation"] = row_ha
        cm = pch.ClusterMapPlotter(plot=False, **cm_kwargs)
        cm.plot(ax=panel_ax, subplot_spec=subplot_spec)
    else:
        cm = pch.ClusterMapPlotter(**cm_kwargs)

    # Hide any side-label text that leaks through fontsize=0.
    for txt in fig.findobj(mpl.text.Text):
        if txt.get_text() == side_label:
            txt.set_visible(False)

    # Identify the colorbar axes by the label we set on it; preserve its ticks.
    cbar_axes = [ax for ax in fig.get_axes() if ax.get_ylabel() == "Scaled PCBL"]

    # Remove ticks from non-colorbar axes
    for ax in fig.get_axes():
        if ax in cbar_axes:
            continue
        ax.tick_params(
            left=False,
            right=False,
            top=False,
            bottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=False,
            length=0,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Restrict colorbar ticks to -1, 0, +1; suppress any minor/intermediate ticks.
    for ax in cbar_axes:
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([-1.0, 0.0, 1.0]))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_yticklabels(["−1", "0", "+1"])
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            length=2,
            width=0.4,
            color=c["black"],
            labelcolor=c["black"],
            labelsize=6,
            pad=1.5,
        )
        ax.set_xticks([])

    # Remove the outer figure frame and inner bounding box
    fig.patch.set_linewidth(0)
    fig.patch.set_edgecolor("none")
    all_axes = fig.get_axes()
    if len(all_axes) > 1:
        for spine in all_axes[1].spines.values():
            spine.set_visible(False)

    # Add black borders to heatmap blocks
    for ax in cm.heatmap_axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor(c["black"])

    # Add borders to column annotation axes (Arm/Visit bars)
    if hasattr(cm, "top_annotation") and cm.top_annotation is not None:
        for ax in cm.top_annotation.axes.flat:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_edgecolor(c["black"])

    # Add borders to row annotation axes and overlay rotated counts
    if hasattr(cm, "left_annotation") and cm.left_annotation is not None:
        anno_axes = cm.left_annotation.axes.flat
        for i, ax in enumerate(anno_axes):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_edgecolor(c["black"])
            if i < len(TRAJECTORY_ORDER):
                traj = TRAJECTORY_ORDER[i]
                displayed = displayed_counts.get(traj, 0)
                true_n = true_counts.get(traj, displayed)
                if displayed > 0:
                    is_compacted = true_n != displayed
                    label = f"{true_n}*" if is_compacted else str(true_n)
                    bg_color = TRAJECTORY_COLORS[traj]
                    r, g, b = mpl.colors.to_rgb(bg_color)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    txt_color = "white" if lum < 0.45 else c["black"]
                    ax.text(
                        0.5,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        fontweight="bold",
                        color=txt_color,
                        rotation=90,
                        clip_on=False,
                        zorder=100,
                    )

    # Fix legend: black text for all labels, thicker borders so pale swatches read,
    # and add breathing room between the "Trajectory" title and the handles.
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            text.set_color(c["black"])
        for patch in legend.legend_handles:
            patch.set_edgecolor(c["black"])
            patch.set_linewidth(0.6)
        # Small gap between the title and the handle stack.
        if hasattr(legend, "_legend_box") and legend._legend_box is not None:
            legend._legend_box.sep = 3

    # Horizontal divider at the down→up inflection (between Reversal ↑↓ and Transient ↑).
    down_last = TRAJECTORY_ORDER.index("Reversal ↑↓")
    up_first = TRAJECTORY_ORDER.index("Transient ↑")
    bbox_above = cm.heatmap_axes[down_last, 0].get_position()
    bbox_below = cm.heatmap_axes[up_first, 0].get_position()
    y_div = (bbox_above.y0 + bbox_below.y1) / 2
    x_left = cm.heatmap_axes[down_last, 0].get_position().x0
    x_right = cm.heatmap_axes[down_last, -1].get_position().x1
    import matplotlib.lines as mlines

    fig.add_artist(
        mlines.Line2D(
            [x_left, x_right],
            [y_div, y_div],
            transform=_fig_transform(fig),
            color=c["black"],
            linewidth=0.8,
            zorder=200,
            clip_on=False,
        )
    )

    # Highlight gene labels on the right of the heatmap with leader lines.
    marker_to_gene = load_marker_to_gene(platform)
    matrix_genes = [marker_to_gene.get(m, m) for m in matrix.index]
    traj_per_row = row_anno["Trajectory"].tolist()

    # Class block bounds in matrix-row index space
    block_bounds: dict[str, tuple[int, int]] = {}
    for i, t in enumerate(traj_per_row):
        s, e = block_bounds.get(t, (i, i))
        block_bounds[t] = (s, i)

    # For each highlighted gene, choose the row (probe) with the strongest raw
    # signal — load max_abs_pcbl from the source parquet to pick by true magnitude
    # (not by scaled values, which are peak-normalized to ±1).
    parquet_path = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "outputs"
        / f"trajectory_{platform}.parquet"
    )
    raw = pl.read_parquet(parquet_path).with_columns(
        pl.max_horizontal(
            pl.col("PCBL_TZP_24").abs(),
            pl.col("PCBL_TZP_72").abs(),
            pl.col("PCBL_SEMA_24").abs(),
            pl.col("PCBL_SEMA_72").abs(),
        ).alias("max_abs_raw")
    )
    marker_to_strength = dict(
        zip(raw["marker"].to_list(), raw["max_abs_raw"].to_list())
    )

    highlight_set = set(HIGHLIGHT_GENES)
    best_row: dict[str, tuple[int, float]] = {}
    for row_idx, (marker, gene) in enumerate(zip(matrix.index, matrix_genes)):
        if gene not in highlight_set:
            continue
        strength = float(marker_to_strength.get(marker, 0.0))
        prev = best_row.get(gene)
        if prev is None or strength > prev[1]:
            best_row[gene] = (row_idx, strength)

    items: list[tuple[float, str, str]] = []
    for gene, (row_idx, _) in best_row.items():
        traj = traj_per_row[row_idx]
        if traj not in block_bounds:
            continue
        block_start, block_end = block_bounds[traj]
        block_count = block_end - block_start + 1
        in_block_idx = row_idx - block_start
        class_idx = TRAJECTORY_ORDER.index(traj)
        ax = cm.heatmap_axes[class_idx, -1]
        bb = ax.get_position()
        frac = (in_block_idx + 0.5) / block_count
        ideal_y = bb.y1 - frac * (bb.y1 - bb.y0)
        items.append((ideal_y, gene, traj))

    # Bidirectional placement with section gaps at class boundaries. Gaps are
    # expressed in (sub)figure fraction but calibrated to a fixed *physical*
    # line-height so the labels never overlap regardless of the panel's height
    # (the composite renders this into a SubFigure of varying height).
    items.sort(key=lambda x: -x[0])
    _subfig_h_in = fig.bbox.height / fig.dpi
    min_gap = (7.5 / 72.0) / _subfig_h_in  # ~7.5 pt line-height for 6 pt labels
    section_gap = (
        12.0 / 72.0
    ) / _subfig_h_in  # extra spacing between trajectory classes
    n = len(items)

    # Bounds = top/bottom of the heatmap area
    top_y = max(
        cm.heatmap_axes[0, -1].get_position().y1
        for j in range(cm.heatmap_axes.shape[1])
    )
    bottom_y = min(
        cm.heatmap_axes[-1, -1].get_position().y0
        for j in range(cm.heatmap_axes.shape[1])
    )

    ideals = [it[0] for it in items]
    classes = [it[2] for it in items]

    def required_gap(i: int) -> float:
        # Gap between item i and item i+1
        return section_gap if classes[i] != classes[i + 1] else min_gap

    ys = list(ideals)

    # Iteratively resolve overlaps. Cluster spans labels where each pair is
    # closer than its required gap; distribute them with their class-aware gaps
    # centered on the cluster's ideal centroid.
    for _ in range(n * 3):
        changed = False
        i = 0
        while i < n - 1:
            if ys[i] - ys[i + 1] >= required_gap(i):
                i += 1
                continue
            j = i
            while j + 1 < n and ys[j] - ys[j + 1] < required_gap(j):
                j += 1
            k = j - i + 1
            # Cumulative gaps within the cluster, accounting for section breaks
            cum = [0.0]
            for m in range(i, j):
                cum.append(cum[-1] + required_gap(m))
            total_span = cum[-1]
            center = sum(ideals[i : j + 1]) / k
            top_in_cluster = center + total_span / 2
            for m in range(k):
                ys[i + m] = top_in_cluster - cum[m]
            changed = True
            i = j + 1
        if not changed:
            break

    # Clamp to figure bounds (two-pass) with class-aware gaps preserved
    if ys:
        ys[0] = min(ys[0], top_y)
        for i in range(1, n):
            ys[i] = min(ys[i], ys[i - 1] - required_gap(i - 1))
        ys[-1] = max(ys[-1], bottom_y)
        for i in range(n - 2, -1, -1):
            ys[i] = max(ys[i], ys[i + 1] + required_gap(i))
        # Last-resort guard: if the labels still don't fit between the heatmap
        # top and bottom (more labels than the panel height allows at min gap),
        # distribute them evenly within the bounds rather than letting the
        # cluster spill past the top edge of the figure.
        if n > 1 and ys[0] > top_y + 1e-9:
            step = (top_y - bottom_y) / (n - 1)
            ys = [top_y - i * step for i in range(n)]

    placed = [(ideals[i], ys[i], items[i][1]) for i in range(n)]

    # Anchor x to the right edge of the rightmost heatmap column block
    x_hm_right = max(
        cm.heatmap_axes[i, -1].get_position().x1
        for i in range(cm.heatmap_axes.shape[0])
    )
    x_leader_end = x_hm_right + 0.015
    x_label = x_hm_right + 0.020

    for ideal_y, adj_y, gene in placed:
        fig.add_artist(
            mlines.Line2D(
                [x_hm_right + 0.001, x_leader_end],
                [ideal_y, adj_y],
                transform=_fig_transform(fig),
                color=c["bold_grey"],
                linewidth=0.6,
                zorder=150,
                clip_on=False,
            )
        )
        fig.text(
            x_label,
            adj_y,
            gene,
            transform=_fig_transform(fig),
            fontsize=FS_BODY,
            ha="left",
            va="center",
            color=c["black"],
            zorder=151,
            family=mpl.rcParams["font.sans-serif"][0],
        )

    platform_label = "Olink" if platform == "olink" else "SomaScan"

    # Compute the panel's bounding box (for embedded titles / footnotes / labels)
    hm_left = min(
        cm.heatmap_axes[i, 0].get_position().x0 for i in range(cm.heatmap_axes.shape[0])
    )
    hm_top = max(
        cm.heatmap_axes[0, j].get_position().y1 for j in range(cm.heatmap_axes.shape[1])
    )
    hm_bottom = min(
        cm.heatmap_axes[-1, j].get_position().y0
        for j in range(cm.heatmap_axes.shape[1])
    )
    # 0.09 offset ensures the cap at 0.97 actually kicks in once the heatmap
    # is shifted down (col_ha takes ~0.05 of subfig, so smaller offsets would
    # follow hm_top and leave a tight visual gap to the arm bar).
    title_y = min(0.97, hm_top + 0.09)

    # In embedded mode, the trajectory legend + colorbar are drawn by the
    # caller (render_combined) in a shared bottom row.

    if embedded:
        # Per-panel title above the heatmap
        fig.text(
            (hm_left + x_hm_right) / 2,
            title_y,
            f"Temporal Proteomics ({platform_label})",
            ha="center",
            va="bottom",
            fontsize=FS_EMPH,
            fontweight="bold",
        )
        if panel_label:
            # Panel "a" aligns with panel "c" (drawn at x=0.005 in the
            # line-plot subfig); panel "b" hugs its own heatmap left edge.
            label_x = 0.005 if panel_label == "a" else hm_left - 0.04
            fig.text(
                label_x,
                title_y,
                panel_label,
                ha="left",
                va="bottom",
                fontsize=FS_PANEL,
                fontweight="bold",
            )
    else:
        fig.suptitle(
            f"Temporal Proteomics ({platform_label})",
            fontsize=FS_EMPH,
            fontweight="bold",
            y=0.93,
        )

    # Footnote when standalone: in embedded mode, render_combined draws the
    # footnote in the shared bottom legend row.
    compacted = any(
        true_counts.get(t, 0) and true_counts[t] != displayed_counts.get(t, 0)
        for t in TRAJECTORY_ORDER
    )
    if compacted and not embedded:
        last_idx = len(TRAJECTORY_ORDER) - 1
        ftnt_bottom = min(
            cm.heatmap_axes[last_idx, j].get_position().y0
            for j in range(cm.heatmap_axes.shape[1])
        )
        fig.text(
            0.5,
            max(0.005, ftnt_bottom - 0.012),
            "* class block heights compacted for visual proportion; counts shown are true class sizes",
            ha="center",
            va="top",
            fontsize=6,
            color=c["black"],
            style="italic",
        )

    if save:
        out_path = str(FIGURES_DIR / f"trajectory_heatmap_{platform}")
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved → {out_path}.pdf")
        plt.close()


# Per-platform compaction targets. Largest class displays at this row count;
# smaller classes √-scale proportionally so visual rank is preserved. Olink's
# largest class is 424, so any cap >= 500 is a no-op there.
# Compact the largest classes (√-scaling) so the small reversal blocks get
# enough height for their rotated count banners. Lower caps = more compression
# of the biggest classes, more relative room for the small ones.
COMPACT_MAX_DISPLAY: dict[str, int] = {
    "olink": 300,
    "soma": 400,
}


def render_combined(parent_fig=None, save: bool = True) -> None:
    """Render the side-by-side heatmap panels (a) Olink and (b) SomaScan, plus
    the shared horizontal legend strip below.

    parent_fig: if provided, draw into that Figure/SubFigure (used by render_full
    to compose this with the trajectory-line panel). When None, create a
    standalone figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_lilly_theme()

    if parent_fig is None:
        fig = plt.figure(figsize=(11, 5.7))
    else:
        fig = parent_fig

    # Position the heatmaps + bottom legend strip in inches so the layout is
    # invariant to whatever subfigure height the composite hands us.
    fig.canvas.draw()
    H_in = fig.bbox.height / fig.dpi
    STRIP_IN = 0.50  # reserved height for the bottom legend strip
    TITLE_IN = 0.30  # reserved above the heatmaps for the title row
    gs_panels = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=0.189,
        top=1.0 - TITLE_IN / H_in,
        bottom=STRIP_IN / H_in,
        left=0.031,
        right=0.93,
    )

    matrix_o, row_anno_o, col_anno_o, true_o = load_and_prepare(
        "olink",
        compact_max_display=COMPACT_MAX_DISPLAY.get("olink"),
        preserve_genes=HIGHLIGHT_GENES,
    )
    matrix_s, row_anno_s, col_anno_s, true_s = load_and_prepare(
        "soma",
        compact_max_display=COMPACT_MAX_DISPLAY.get("soma"),
        preserve_genes=HIGHLIGHT_GENES,
    )

    render_heatmap(
        matrix_o,
        row_anno_o,
        col_anno_o,
        "olink",
        true_counts=true_o,
        parent_fig=fig,
        subplot_spec=gs_panels[0, 0],
        show_legend=False,
        panel_label="a",
        save=False,
    )
    render_heatmap(
        matrix_s,
        row_anno_s,
        col_anno_s,
        "soma",
        true_counts=true_s,
        parent_fig=fig,
        subplot_spec=gs_panels[0, 1],
        show_legend=False,
        panel_label="b",
        save=False,
    )

    # ---------------------------------------------------------------------
    # Bottom legend strip: trajectory classes as two rows of five swatches on
    # the left, the Scaled-PCBL colorbar block on the right. Two rows because at
    # 183 mm a single row of ten class labels does not fit.
    # ---------------------------------------------------------------------
    from matplotlib.patches import Patch

    # Two tight rows of five trajectory swatches on the left; the Scaled-PCBL
    # key on one line at the upper right; the compaction footnote on the lower
    # right, directly under the SomaScan heatmap.
    ROW1_Y = 0.32 / H_in
    ROW2_Y = 0.14 / H_in
    MID_Y = (ROW1_Y + ROW2_Y) / 2

    # "Trajectory" label at the far left, vertically centered on the two rows.
    fig.text(
        0.031,
        MID_Y,
        "Trajectory",
        fontsize=FS_NARR,
        fontweight="bold",
        color=c["black"],
        ha="left",
        va="center",
    )

    # Row 1 = the five "down" classes; row 2 = the "up" classes reversed so each
    # ↑ class sits directly under its ↓ counterpart (Late-onset, Progressive,
    # Sustained, Transient, Reversal). TRAJECTORY_ORDER is symmetric (downs then
    # ups mirrored), so the second half must be reversed to realign by class.
    half = len(TRAJECTORY_ORDER) // 2
    for subset, row_y in (
        (TRAJECTORY_ORDER[:half], ROW1_Y),
        (TRAJECTORY_ORDER[half:][::-1], ROW2_Y),
    ):
        handles = [
            Patch(
                facecolor=TRAJECTORY_COLORS[t],
                edgecolor=c["black"],
                linewidth=0.6,
                label=t,
            )
            for t in subset
        ]
        leg = fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.135, row_y),
            bbox_transform=_fig_transform(fig),
            ncol=len(subset),
            fontsize=FS_BODY,
            frameon=False,
            handlelength=0.9,
            handletextpad=0.25,
            columnspacing=0.6,
            labelspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        for text in leg.get_texts():
            text.set_color(c["black"])

    # Scaled-PCBL key — label, −1, colorbar, +1 all on ONE line (upper right).
    cbar_x0, cbar_x1 = 0.842, 0.902
    cbar_h = 0.10 / H_in
    cax = fig.add_axes([cbar_x0, ROW1_Y - cbar_h / 2, cbar_x1 - cbar_x0, cbar_h])
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0),
        cmap=build_cmap("Lilly_Diverging"),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.set_xticks([])
    cbar.ax.tick_params(
        bottom=False, top=False, labelbottom=False, labeltop=False, length=0
    )
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(c["black"])
    fig.text(
        cbar_x0 - 0.040,
        ROW1_Y,
        "Scaled PCBL",
        fontsize=FS_NARR,
        fontweight="bold",
        color=c["black"],
        ha="right",
        va="center",
    )
    fig.text(
        cbar_x0 - 0.006,
        ROW1_Y,
        "−1",
        fontsize=FS_BODY,
        color=c["black"],
        ha="right",
        va="center",
    )
    fig.text(
        cbar_x1 + 0.006,
        ROW1_Y,
        "+1",
        fontsize=FS_BODY,
        color=c["black"],
        ha="left",
        va="center",
    )

    # Compaction footnote (the * on class counts) — directly under the SomaScan
    # heatmap, right-aligned with its right edge.
    if any(
        true_s.get(t, 0)
        and true_s[t] != row_anno_s["Trajectory"].value_counts().get(t, 0)
        for t in TRAJECTORY_ORDER
    ):
        fig.text(
            0.93,
            ROW2_Y,
            "* block heights compacted; counts are exact",
            ha="right",
            va="center",
            fontsize=FS_AUX,
            color=c["bold_grey"],
            style="italic",
        )

    if save:
        out_path = str(FIGURES_DIR / "trajectory_heatmap_combined")
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved → {out_path}.pdf")
        plt.close()


def render_full() -> None:
    """Render the full composite: heatmap panels (a, b) on top, trajectory-line
    panels (c) on the bottom — a single PDF for the manuscript figure.
    """
    import matplotlib.pyplot as plt
    from . import _trajectory_lines as tl
    from ._common import NAT_W2

    init_figure_theme()

    # Nature double-column width (183 mm). Two acts stacked: a tall heatmap
    # "landscape" overview (a, b) over a compact 2-row trajectory-line strip (c).
    FIG_W = NAT_W2
    HEATMAP_H = 5.30
    LINES_H = (
        3.60  # sized to the width-derived line-panel height (see _trajectory_lines)
    )
    fig = plt.figure(figsize=(FIG_W, HEATMAP_H + LINES_H))
    subfigs = fig.subfigures(
        2,
        1,
        height_ratios=[HEATMAP_H, LINES_H],
        hspace=0.04,
    )
    render_combined(parent_fig=subfigs[0], save=False)
    tl.render(parent_fig=subfigs[1], save=False, panel_label="c")

    out_path = str(FIGURES_DIR / "fig1_trajectory")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved → {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    init_figure_theme()
    render_full()
