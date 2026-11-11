"""Figure 1: trajectory classes (a) and the %CFB trajectories of the configured genes (b, c).

Panel a: butterfly bar charts of trajectory-class counts per arm (TZP left, SEMA
    right) for Olink and SomaScan, with a "same class" count column outside each
    frame. Counts include every assay in the trajectory table.
Panels b and c draw the configured trajectory genes in platform rows.

Trajectory panels reuse ``_internal._trajectory_lines`` (lines, 95% CIs, % labels,
between-treatment FDR stars, the "Not covered" note and ``draw_class_banner``). A single
"% Change from baseline (95% CI)" label heads each row block; the bottom strip carries
the FDR key (left), "Week" (center) and the arm legend (right).

Panel c one-platform status is asserted against ``uniprot_map.parquet`` (the repo's
``genes_per_platform``) and against both platforms' per-assay vendor annotation (stage
``vendor_assay_tables``), by every gene symbol and every UniProt of multi-target assays; a
disagreement stops the build.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.transforms import blended_transform_factory

import figures._internal._trajectory_data as td
import figures._internal._trajectory_lines as tl
from analysis.design import ARMS, EARLY_WEEK, LATE_WEEK, PLATFORMS
from figures._internal._common import (
    BETWEEN_TREATMENT_FDR_KEY,
    ANALYSIS_DIR,
    ARM_COLORS,
    ARM_LABELS,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_MIN,
    FS_PANEL,
    NAT_W2,
    TRAJ_PANEL_ASPECT,
    TRAJECTORY_COLORS,
    c,
    init_figure_theme,
    marker_labels,
    nature_figsize,
    save_figure,
)
from figures._internal._theme import PLATFORM_SHORT

THIS_DIR = Path(__file__).resolve().parent


INK = c["black"]
ID_GREY = "#7A7A7A"
PLAT = PLATFORM_SHORT
OTHER = {"olink": "soma", "soma": "olink"}

# Panel a row order (top to bottom): up-classes, the two reversals, down-classes.
CLASS_ORDER = [
    "Late-onset ↑",
    "Sustained ↑",
    "Progressive ↑",
    "Transient ↑",
    "Reversal ↑↓",
    "Reversal ↓↑",
    "Transient ↓",
    "Progressive ↓",
    "Sustained ↓",
    "Late-onset ↓",
]
REVERSAL_GAP = 0.4  # extra row gap after the last up-first class

# Genes drawn in panels b (both platforms) and c (one platform each).
REPLICATED = ["LEP", "IGFBP2", "PNLIPRP1", "FABP4"]
SINGLE = {
    "olink": ["PAPPA", "GAST", "COL26A1", "AMDHD1"],
    "soma": ["ADIPOQ", "AZGP1", "NPW", "CRP"],
}

# Per-assay vendor annotation (analysis/external/vendor_assay_tables.py): pipe-joined
# UniProt_ID and Gene_Symbol per assay (multi-target assays carry several of each).
PANEL_REF = {
    "olink": (
        ANALYSIS_DIR.parent / "data" / "external" / "vendor_assay_tables" / "olink_explore_ht_annotation.tsv",
        "OlinkID",
    ),
    "soma": (
        ANALYSIS_DIR.parent / "data" / "external" / "vendor_assay_tables" / "somascan_11k_annotation.tsv",
        "SeqId",
    ),
}

# Layout (inches).
LEFT_IN, RIGHT_IN, WSPACE = 0.61, 0.11, 0.35
HEAD = 0.46  # gene name + assay ID + banner above a row
GY = 0.64  # x-tick labels of the upper row + header of the lower row
FA_TOP, FA_H = 0.42, 1.60  # panel a frame
FA_LEFT = 0.86  # panel a Olink frame, left edge (room for the class labels)
FA_W = 2.52  # panel a frame width
FA_GUTTER = 0.80  # same-class column + gap between the two frames
SOMA_X = FA_LEFT + FA_W + FA_GUTTER
LETTER_X = 0.02
TAG_X = 0.27  # platform tags one column in, so they never meet the y-label
SAME_PAD_IN = 0.05  # frame -> "same class" column
YLABEL = tl.YLABEL
FDR_KEY = BETWEEN_TREATMENT_FDR_KEY

# "Not covered" note corners (the positions tl.draw_gene_panel chooses among).
CORNERS = {
    "top-left": (0, "left", 0.96, "top"),
    "top-right": (80, "right", 0.96, "top"),
    "bottom-left": (0, "left", 0.04, "bottom"),
    "bottom-right": (80, "right", 0.04, "bottom"),
}


# ---------------------------------------------------------------------------
# One-platform coverage check (panel c)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _panel_ref(platform: str) -> pl.DataFrame:
    path, _ = PANEL_REF[platform]
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def _split_values(frame: pl.DataFrame, column: str) -> set[str]:
    return set(
        frame.select(pl.col(column).str.split("|").explode()).drop_nulls()[column]
    )


def assert_one_platform(platform: str, gene: str, marker: str) -> None:
    """Assert that ``gene`` (assay ``marker`` on ``platform``) is absent from the other platform.

    Checked twice: the repo's ``uniprot_map`` coverage, and the raw panel reference of
    the other platform by every gene symbol and every UniProt of the assay's targets.
    """
    other = OTHER[platform]
    assert (
        gene not in td.genes_per_platform()[other]
    ), f"uniprot_map lists {gene} on {PLAT[other]}"
    _, id_col = PANEL_REF[platform]
    own = _panel_ref(platform).filter(pl.col(id_col) == marker)
    assert own.height == 1, f"{marker} is not one row of the {PLAT[platform]} reference"
    genes = _split_values(own, "Gene_Symbol")
    uniprots = _split_values(own, "UniProt_ID")
    assert gene in genes, f"{gene} not among the targets of {marker}: {sorted(genes)}"
    other_ref = _panel_ref(other)
    gene_hits = genes & _split_values(other_ref, "Gene_Symbol")
    uniprot_hits = uniprots & _split_values(other_ref, "UniProt_ID")
    assert not gene_hits and not uniprot_hits, (
        f"{gene} ({marker}) is also on {PLAT[other]} in the raw reference: "
        f"genes {sorted(gene_hits)}, UniProts {sorted(uniprot_hits)}"
    )


# ---------------------------------------------------------------------------
# Panel a: class-count butterflies
# ---------------------------------------------------------------------------
def draw_butterfly(ax, platform: str, *, show_labels: bool = True) -> None:
    """Class counts per arm, TZP left | SEMA right, with a "same class" column outside.

    Every assay in the trajectory table counts; "Null" is the complement of the
    changed-from-baseline counts in the footnote.
    """
    table = td.load_trajectory_table(platform)
    tzp, sema = pl.col("trajectory_TZP"), pl.col("trajectory_SEMA")
    counts = {
        k: (
            table.filter(tzp == k).height,
            table.filter(sema == k).height,
            table.filter((tzp == k) & (sema == k)).height,
        )
        for k in CLASS_ORDER
    }
    ys, y = [], len(CLASS_ORDER) - 1.0
    for k in CLASS_ORDER:
        ys.append(y)
        y -= 1
        if k == "Reversal ↑↓":
            y -= REVERSAL_GAP
    m = max(max(n_t, n_s) for n_t, n_s, _ in counts.values())
    for yi, k in zip(ys, CLASS_ORDER):
        n_t, n_s, _ = counts[k]
        for width in (-n_t, n_s):
            ax.barh(
                yi,
                width,
                height=0.74,
                color=TRAJECTORY_COLORS[k],
                edgecolor=INK,
                lw=0.25,
            )
        ax.text(-n_t - 0.03 * m, yi, f"{n_t:,}", ha="right", va="center", fontsize=FS_MIN)
        ax.text(n_s + 0.03 * m, yi, f"{n_s:,}", ha="left", va="center", fontsize=FS_MIN)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlim(-1.32 * m, 1.32 * m)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xticks([])
    ax.set_yticks(ys)
    ax.set_yticklabels(CLASS_ORDER if show_labels else [], fontsize=FS_AUX)
    ax.tick_params(axis="y", which="both", length=0, pad=1)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    axes_x = ax.get_xaxis_transform()
    ax.text(
        0, 1.075, PLAT[platform], transform=axes_x, ha="center", va="bottom",
        fontsize=FS_EMPH, fontweight="bold",
    )
    for arm, x in (("TZP", -0.66 * m), ("SEMA", 0.66 * m)):
        ax.text(
            x, 1.01, arm, transform=axes_x, ha="center", va="bottom", fontsize=FS_BODY,
            fontweight="bold", color=ARM_COLORS[arm],
        )
    # "same class" column outside the frame, left-aligned against the right border
    frame_w_in = ax.get_position().width * ax.figure.get_figwidth()
    x_col = 1 + SAME_PAD_IN / frame_w_in
    data_y = blended_transform_factory(ax.transAxes, ax.transData)
    for yi, k in zip(ys, CLASS_ORDER):
        ax.text(
            x_col, yi, f"{counts[k][2]:,}", transform=data_y, ha="left", va="center",
            fontsize=FS_MIN, fontweight="bold", clip_on=False,
        )
    ax.text(
        x_col, 1.01, "same\nclass", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=FS_MIN, linespacing=0.9, clip_on=False,
    )
    n_changed_tzp = table.filter(tzp != "Null").height
    n_changed_sema = table.filter(sema != "Null").height
    ax.text(
        0.5, -0.015,
        f"Changed from baseline (week {EARLY_WEEK} or {LATE_WEEK}):\n"
        f"TZP {n_changed_tzp:,} · SEMA {n_changed_sema:,} of {table.height:,} assays",
        transform=ax.transAxes, ha="center", va="top", fontsize=FS_MIN,
        color="#444444", linespacing=1.15,
    )


# ---------------------------------------------------------------------------
# Panels b, c: trajectory panels
# ---------------------------------------------------------------------------
def draw_trajectory_panel(ax, platform: str, gene: str, *, one_platform: bool) -> None:
    """``draw_gene_panel`` (FDR stars on) + class banner + gene name and bare assay ID."""
    marker = td.best_marker_for_gene(platform, gene)
    assert marker is not None, f"{gene} is not on {PLAT[platform]}"
    if one_platform:
        assert_one_platform(platform, gene, marker)
    else:
        assert gene in td.genes_per_platform()[OTHER[platform]], (
            f"{gene} is not on {PLAT[OTHER[platform]]}"
        )
    classes = tl.draw_gene_panel(
        ax, platform, gene, not_covered_by=PLAT[OTHER[platform]] if one_platform else None
    )
    tzp_class, sema_class = classes
    for text in ax.texts:  # "-0%" / "+0%" -> "0%"
        if text.get_text() in ("-0%", "+0%"):
            text.set_text("0%")
    tl.draw_class_banner(ax, tzp_class, sema_class)
    ax.text(
        0.5, 1.185, marker, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=FS_MIN, color=ID_GREY, clip_on=False,
    )
    ax.text(
        0.5, 1.285, marker_labels(platform).get(marker, gene), transform=ax.transAxes, ha="center", va="bottom",
        fontsize=FS_EMPH, fontweight="bold", clip_on=False,
    )


def _pct_and_star_texts(ax) -> list:
    return [
        t for t in ax.texts
        if t.get_text() and (t.get_text().endswith("%") or set(t.get_text()) == {"*"})
    ]


def _pad_labels(ax, renderer, *, top: bool = True) -> None:
    """Extend the y-range in 2% steps until every % label and star is inside the axes."""
    for _ in range(10):
        axes_bb = ax.get_window_extent(renderer)
        boxes = [t.get_window_extent(renderer) for t in _pct_and_star_texts(ax)]
        low = any(b.y0 < axes_bb.y0 for b in boxes)
        high = top and any(b.y1 > axes_bb.y1 for b in boxes)
        if not (low or high):
            return
        lo, hi = ax.get_ylim()
        step = 0.02 * (hi - lo)
        ax.set_ylim(lo - step * low, hi + step * high)
    raise RuntimeError("% labels do not fit inside the panel after 10 pad steps")


def _fit_panel(ax, renderer) -> None:
    """Grow the y-range until the labels fit and the 'Not covered' note (if any) touches no data.

    Padding a label side squeezes the data toward the other end, which can push it into the note
    or the labels there, so both checks repeat (each step only widens the range) until neither
    changes it.
    """
    note = _note_of(ax)
    for _ in range(50):
        before = ax.get_ylim()
        _pad_labels(ax, renderer)
        if note is not None and _note_hits(ax, note, renderer):
            lo, hi = ax.get_ylim()
            step = 0.02 * (hi - lo)
            if note.get_va() == "top":
                ax.set_ylim(lo, hi + step)
            else:
                ax.set_ylim(lo - step, hi)
        if ax.get_ylim() == before:
            return
    raise RuntimeError("labels and the note do not both fit inside the panel after 50 steps")


def _note_of(ax):
    notes = [t for t in ax.texts if t.get_text().startswith("Not covered")]
    return notes[0] if notes else None


def _set_corner(note, corner: str) -> None:
    x, ha, y, va = CORNERS[corner]
    note.set_position((x, y))
    note.set_ha(ha)
    note.set_va(va)


def _note_hits(ax, note, renderer, pad_pt: float = 1.5) -> list[str]:
    """Everything the note touches: lines, markers, error bars/caps, the 0 line, other text."""
    dpi = ax.figure.dpi
    box = note.get_window_extent(renderer).padded(pad_pt * dpi / 72)
    hits = []
    for line in ax.lines:
        path = line.get_transform().transform_path(line.get_path())
        if str(line.get_linestyle()).lower() not in ("none", "", " "):
            if path.intersects_bbox(box, filled=False) or any(
                box.contains(x, y) for x, y in path.vertices
            ):
                hits.append("line")
        if str(line.get_marker()).lower() not in ("none", "", " "):
            radius = line.get_markersize() * dpi / 72 / 2
            if any(box.padded(radius).contains(x, y) for x, y in path.vertices):
                hits.append("marker")
    for coll in ax.collections:
        transform = coll.get_transform()
        for p in coll.get_paths():
            tp = transform.transform_path(p)
            if tp.intersects_bbox(box, filled=False) or any(
                box.contains(x, y) for x, y in tp.vertices
            ):
                hits.append("errorbar")
    for t in ax.texts:
        if (
            t is not note
            and t.get_text().strip()
            and t.get_window_extent(renderer).overlaps(box)
        ):
            hits.append(repr(t.get_text()))
    return hits


def _min_extension(ax, note, corner: str, renderer, step=0.05, kmax=10) -> int | None:
    """Smallest k such that the note at ``corner``, with the y-range extended by k*step
    on the note's side, touches nothing; the y-range is restored."""
    lo0, hi0 = ax.get_ylim()
    span = hi0 - lo0
    _set_corner(note, corner)
    for k in range(kmax + 1):
        ext = k * step * span
        ax.set_ylim((lo0, hi0 + ext) if corner.startswith("top") else (lo0 - ext, hi0))
        if not _note_hits(ax, note, renderer):
            ax.set_ylim(lo0, hi0)
            return k
    ax.set_ylim(lo0, hi0)
    return None


def _apply_note(ax, note, corner: str, k: int, step=0.05) -> None:
    lo, hi = ax.get_ylim()
    ext = k * step * (hi - lo)
    _set_corner(note, corner)
    ax.set_ylim((lo, hi + ext) if corner.startswith("top") else (lo - ext, hi))


def place_notes(rows: dict[str, list[tuple[str, object]]], fig) -> None:
    """Place the "Not covered" notes of panel c: one corner per row where possible, the
    y-range extended only as far as needed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for platform, panels in rows.items():
        need = {
            cn: [_min_extension(ax, _note_of(ax), cn, renderer) for _, ax in panels]
            for cn in CORNERS
        }
        feasible = {cn: sum(v) for cn, v in need.items() if None not in v}
        row_corner = min(feasible, key=feasible.get)
        for i, (_gene, ax) in enumerate(panels):
            note = _note_of(ax)
            best_k, best_corner = min(
                (need[cn][i], cn) for cn in CORNERS if need[cn][i] is not None
            )
            if need[row_corner][i] <= best_k + 2:
                _apply_note(ax, note, row_corner, need[row_corner][i])
            else:
                _apply_note(ax, note, best_corner, best_k)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def make_figure():
    """Build Figure 1: class-count butterflies (a) and the configured genes' trajectories (b, c).

    Reads the trajectory parquets and the MMRM PCBL and between-treatment outputs, checks
    panel c's one-platform genes against ``uniprot_map.parquet`` and the vendor
    annotation (``PANEL_REF``), then pads labels and places the "Not covered" notes.

    Returns:
        The figure.

    Raises:
        AssertionError: If a panel c gene is on the other platform after all, a label leaves
            its panel, a note touches data or text falls outside the figure (``check_figure``).
    """
    init_figure_theme()
    width = NAT_W2
    panel_w = (width - LEFT_IN - RIGHT_IN) / (4 + 3 * WSPACE)
    panel_h = panel_w / TRAJ_PANEL_ASPECT
    gap_x = WSPACE * panel_w
    y_b = FA_TOP + FA_H + 0.42 + HEAD + 0.08
    y_c = y_b + 2 * panel_h + GY + 0.32 + HEAD
    y_strip = y_c + 2 * panel_h + GY + 0.22
    height = y_strip + 0.12
    fig = plt.figure(figsize=nature_figsize(width, height))

    def axes_at(x, y, w, h):
        return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

    def text_at(x, y, s, **kwargs):
        kwargs.setdefault("fontsize", FS_BODY)
        return fig.text(x / width, 1 - y / height, s, **kwargs)

    def letter(y, s):
        fig.text(
            LETTER_X / width, 1 - y / height, s, fontsize=FS_PANEL,
            fontweight="bold", ha="left", va="bottom",
        )

    # a
    letter(0.20, "a")
    draw_butterfly(axes_at(FA_LEFT, FA_TOP, FA_W, FA_H), "olink")
    draw_butterfly(axes_at(SOMA_X, FA_TOP, FA_W, FA_H), "soma", show_labels=False)

    # b, c
    label_x = tl.LINE_LABEL_X * width  # shared y-label gutter of the trajectory figures
    rows_by_block: dict[str, dict[str, list[tuple[str, object]]]] = {"b": {}, "c": {}}

    def block(y0, rows, label, one_platform):
        letter(y0 - HEAD - 0.06, label)
        for ri, (platform, genes) in enumerate(rows):
            y = y0 + ri * (panel_h + GY)
            for ci, gene in enumerate(genes):
                ax = axes_at(LEFT_IN + ci * (panel_w + gap_x), y, panel_w, panel_h)
                draw_trajectory_panel(ax, platform, gene, one_platform=one_platform)
                rows_by_block[label].setdefault(platform, []).append((gene, ax))
            text_at(
                TAG_X, y + panel_h / 2, PLAT[platform], fontsize=FS_EMPH,
                fontweight="bold", rotation=90, ha="center", va="center",
            )
        text_at(
            label_x, y0 + panel_h + GY / 2, YLABEL, fontsize=FS_EMPH, rotation=90,
            ha="center", va="center",
        )

    block(y_b, [("olink", REPLICATED), ("soma", REPLICATED)], "b", False)
    block(y_c, [(p, SINGLE[p]) for p in PLATFORMS], "c", True)

    # bottom strip: FDR key (left), Week (center), arm legend (right)
    x_left, x_right = LEFT_IN / width, 1 - RIGHT_IN / width
    y_fig = 1 - y_strip / height
    fig.text(
        x_left, y_fig, FDR_KEY, fontsize=FS_BODY, color=INK, ha="left", va="center",
        style="italic",
    )
    fig.text((x_left + x_right) / 2, y_fig, "Week", fontsize=FS_EMPH, ha="center", va="center")
    arms = [
        mlines.Line2D(
            [0], [0], color=ARM_COLORS[a], marker="o", markersize=3, linewidth=0.9,
            label=ARM_LABELS[a],
        )
        for a in tuple(ARMS)
    ]
    fig.legend(
        handles=arms, loc="center right", bbox_to_anchor=(x_right, y_fig),
        bbox_transform=fig.transFigure, ncol=2, fontsize=FS_EMPH, frameon=False,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
    )

    # keep % labels and stars inside their axes, then place the notes, then re-pad both
    # ends (a bottom extension squeezes the data up, so top labels can slip out)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_panels = [ax for rows in rows_by_block.values() for ps in rows.values() for _, ax in ps]
    for ax in all_panels:
        _pad_labels(ax, renderer)
    place_notes(rows_by_block["c"], fig)
    fig.canvas.draw()
    for ax in all_panels:
        _fit_panel(ax, renderer)
    fig.canvas.draw()
    check_figure(fig, all_panels, renderer)
    return fig


def check_figure(fig, panels, renderer) -> None:
    """Fail loudly on clipped or overlapping text."""
    for ax in panels:
        axes_bb = ax.get_window_extent(renderer)
        for t in _pct_and_star_texts(ax):
            bb = t.get_window_extent(renderer)
            assert axes_bb.y0 <= bb.y0 and bb.y1 <= axes_bb.y1, (
                f"label {t.get_text()!r} leaves its panel"
            )
        note = _note_of(ax)
        if note is not None:
            assert not _note_hits(ax, note, renderer), "a 'Not covered' note touches data"
    figure_bb = fig.bbox
    for t in fig.findobj(
        lambda o: hasattr(o, "get_text") and hasattr(o, "get_window_extent") and o.get_text().strip()
    ):
        bb = t.get_window_extent(renderer)
        assert (
            figure_bb.x0 - 1 <= bb.x0 and bb.x1 <= figure_bb.x1 + 1
            and figure_bb.y0 - 1 <= bb.y0 and bb.y1 <= figure_bb.y1 + 1
        ), f"text {t.get_text()[:40]!r} is outside the figure"


def render_full() -> None:
    """Build Figure 1 and save ``figures/fig1_trajectory.pdf`` at its designed page size.

    ``save_figure`` honors the ``FIGURE_OUTPUT_DIR`` / ``FIGURE_OUTPUT_FORMATS`` overrides of
    the regression harness.
    """
    fig = make_figure()
    out_path = str(THIS_DIR / "fig1_trajectory")
    # the theme crops to a tight bbox; the page must keep its designed size
    with mpl.rc_context({"savefig.bbox": "standard"}):
        save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf ({fig.get_figwidth() * 25.4:.1f} x {fig.get_figheight() * 25.4:.1f} mm)")
    plt.close(fig)


if __name__ == "__main__":
    render_full()
