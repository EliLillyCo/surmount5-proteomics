"""Per-gene trajectory line plots (paired with combined heatmap).

2x4 grid: row = platform (Olink top, SomaScan bottom); cols 1-4 = cross-platform
validated genes (same gene on both rows), cols 5-8 = platform-specific genes.
Each panel shows %CFB at Wk0/24/72 for TZP and SEMA with error bars and endpoint
labels. A colored banner at the top of each panel encodes the gene's TZP
trajectory class (matching the heatmap color scheme).
"""

from __future__ import annotations

from functools import lru_cache

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl

from ._common import (
    ANALYSIS_DIR,
    ARM_COLORS,
    ARM_LABELS,
    FS_BODY,
    FS_EMPH,
    FS_NARR,
    FS_PANEL,
    LILLY_COLORS,
    NAT_W2,
    PALETTE_VARIANT,
    TRAJECTORY_COLORS,
    apply_lilly_theme,
    init_figure_theme,
    load_marker_to_gene,
    save_figure,
)
from paths import COVAR, ensure_public_alias, public_mmrm_dir

c = LILLY_COLORS[PALETTE_VARIANT]
ARMS = {"TZP": "TZP15mgorMTD", "SEMA": "SEMA2.4mgorMTD"}
WEEKS = [0, 24, 72]

# Columns 0-1 = cross-platform (same gene on both rows); columns 2-3 =
# platform-specific. The remaining trajectory exemplars (MSTN, PNLIPRP1,
# HSD11B1, LPL, CRP, FABP3) live in supp_fig4_trajectory_lines.py.
PANELS: dict[str, list[str]] = {
    "olink": ["LEP", "IGFBP2", "COL26A1", "GAST"],
    "soma": ["LEP", "IGFBP2", "ADIPOQ", "INHBA"],
}
N_CROSS = 2  # first N_CROSS columns are the cross-platform pair


# ---------------------------------------------------------------------------
# Data loading (cached per platform)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_pcbl_csv(platform: str, arm_full: str, week: int) -> pl.DataFrame:
    path = ensure_public_alias(
        public_mmrm_dir(platform)
        / "finalRes"
        / "pcblRes"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_PCBLRes_{arm_full}@{week}_py.csv"
    )
    return pl.read_csv(path).select("marker", "PCBL", "SE_PCBL", "PCBL_lower_CI", "PCBL_upper_CI")


@lru_cache(maxsize=4)
def _load_actrt_csv(platform: str, week: int) -> pl.DataFrame:
    """TZP vs SEMA across-treatment contrast (FDR) for one platform/week."""
    contrast = "TZP15mgorMTDVSSEMA2.4mgorMTD"
    path = ensure_public_alias(
        public_mmrm_dir(platform)
        / "finalRes"
        / "acTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_{contrast}@{week}_py.csv"
    )
    return pl.read_csv(path).select("marker", "fdr")


def _fdr_stars_str(fdr: float | None) -> str:
    if fdr is None:
        return ""
    if fdr < 1e-3:
        return "***"
    if fdr < 1e-2:
        return "**"
    if fdr < 5e-2:
        return "*"
    return ""


@lru_cache(maxsize=4)
def _load_trajectory_table(platform: str) -> pl.DataFrame:
    return pl.read_parquet(ANALYSIS_DIR / "outputs" / f"trajectory_{platform}.parquet")


@lru_cache(maxsize=1)
def _genes_per_platform() -> dict[str, frozenset[str]]:
    """Set of gene symbols measured on each platform (from uniprot_map)."""
    m = pl.read_parquet(ANALYSIS_DIR / "outputs" / "uniprot_map.parquet")
    return {
        "olink": frozenset(m.filter(pl.col("present_olink"))["gene_symbol"].to_list()),
        "soma": frozenset(m.filter(pl.col("present_soma"))["gene_symbol"].to_list()),
    }


def _best_marker_for_gene(platform: str, gene: str) -> str | None:
    mtg = load_marker_to_gene(platform)
    markers = [m for m, g in mtg.items() if g == gene]
    if not markers:
        return None
    candidates = (
        _load_trajectory_table(platform)
        .filter(pl.col("marker").is_in(markers))
        .with_columns(
            pl.max_horizontal(
                pl.col("PCBL_TZP_24").abs(),
                pl.col("PCBL_TZP_72").abs(),
                pl.col("PCBL_SEMA_24").abs(),
                pl.col("PCBL_SEMA_72").abs(),
            ).alias("max_abs")
        )
        .sort("max_abs", descending=True)
    )
    return None if candidates.is_empty() else candidates["marker"][0]


def _gene_data(
    platform: str, gene: str
) -> tuple[str | None, dict[tuple[str, int], tuple[float, float, float, float]]]:
    marker = _best_marker_for_gene(platform, gene)
    if marker is None:
        return None, {}
    traj_row = _load_trajectory_table(platform).filter(pl.col("marker") == marker)
    if traj_row.is_empty():
        return None, {}
    trajectory = traj_row["trajectory_TZP"][0]

    data: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    for arm_short, arm_full in ARMS.items():
        for week in (24, 72):
            df = _load_pcbl_csv(platform, arm_full, week)
            row = df.filter(pl.col("marker") == marker)
            if row.is_empty():
                continue
            data[(arm_short, week)] = (
                row["PCBL"][0],
                row["SE_PCBL"][0],
                row["PCBL_lower_CI"][0],
                row["PCBL_upper_CI"][0],
            )
    return trajectory, data


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------
def _draw_panel(
    ax,
    platform: str,
    gene: str,
    show_x: bool,
    not_covered_by: str | None = None,
    show_fdr_stars: bool = False,
) -> str | None:
    """Render the trajectory line plot for one gene on one platform.

    show_fdr_stars: when True, overlay TZP-vs-SEMA across-treatment FDR
    significance stars near the timepoints they apply to. Star placement
    follows the trajectory direction — above the higher arm for ↑ classes,
    below the lower arm for ↓ classes (mirroring the supp_fig5_tzp_induced /
    supp_fig6_tzp_suppressed convention).
    """
    trajectory, data = _gene_data(platform, gene)
    if trajectory is None:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"{gene}\n(not measured)",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
            color=c["bold_grey"],
        )
        return None

    ax.axhline(0, color=c["bold_grey"], linewidth=0.4, zorder=1)

    for arm_short in ("SEMA", "TZP"):
        y = [0.0]
        yerr_lo = [0.0]
        yerr_hi = [0.0]
        for w in (24, 72):
            if (arm_short, w) in data:
                pcbl, _se, ci_lo, ci_hi = data[(arm_short, w)]
                y.append(pcbl)
                yerr_lo.append(pcbl - ci_lo)
                yerr_hi.append(ci_hi - pcbl)
            else:
                y.append(0.0)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
        color = ARM_COLORS[arm_short]
        ax.errorbar(
            WEEKS,
            y,
            yerr=[yerr_lo, yerr_hi],
            color=color,
            linewidth=0.9,
            marker="o",
            markersize=2.8,
            capsize=1.5,
            capthick=0.5,
            elinewidth=0.5,
            zorder=10 if arm_short == "TZP" else 9,
        )

    ax.set_xlim(-6, 80)
    # Adaptive y-padding so the % labels (and optional FDR stars) stay inside.
    _y_with_err: list[float] = [0.0]
    for _arm in ("TZP", "SEMA"):
        for _w in (24, 72):
            if (_arm, _w) in data:
                _v, _se, _ci_lo, _ci_hi = data[(_arm, _w)]
                _y_with_err.extend([_ci_lo, _ci_hi])
    _data_min = min(_y_with_err)
    _data_max = max(_y_with_err)
    _data_range = max(_data_max - _data_min, 1e-6)
    ymin = _data_min - 0.10 * _data_range
    ymax = _data_max + 0.10 * _data_range
    ax.set_ylim(ymin, ymax)
    y_range = max(ymax - ymin, 1e-6)
    axes_height_pts = ax.bbox.height / ax.figure.dpi * 72

    def _ci_half_to_pts(ci_half: float) -> float:
        return ci_half / y_range * axes_height_pts

    LABEL_FONT_PTS = 6.0
    PCT_OFFSET_PAD_PTS = 4.0
    STAR_FONT_PTS = 7.0
    STAR_GAP_PTS = 3.0
    STAR_MARGIN_PTS = 3.0

    # Determine star direction from trajectory class.
    stars_above = "↑" in trajectory

    # Across-treatment FDR per timepoint (only loaded when needed).
    star_marker: str | None = None
    fdr_per_week: dict[int, float | None] = {}
    if show_fdr_stars:
        star_marker = _best_marker_for_gene(platform, gene)
        if star_marker is not None:
            for _w in (24, 72):
                _df = _load_actrt_csv(platform, _w).filter(
                    pl.col("marker") == star_marker
                )
                fdr_per_week[_w] = _df["fdr"][0] if not _df.is_empty() else None

    def _higher_arm_at(w: int):
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val, _tzp_se, _tzp_ci_lo, tzp_ci_hi = data[("TZP", w)]
        sema_val, _sema_se, _sema_ci_lo, sema_ci_hi = data[("SEMA", w)]
        if tzp_val > sema_val:
            return (tzp_val, (tzp_ci_hi - tzp_val))
        return (sema_val, (sema_ci_hi - sema_val))

    def _lower_arm_at(w: int):
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val, _tzp_se, tzp_ci_lo, _tzp_ci_hi = data[("TZP", w)]
        sema_val, _sema_se, sema_ci_lo, _sema_ci_hi = data[("SEMA", w)]
        if tzp_val < sema_val:
            return (tzp_val, (tzp_val - tzp_ci_lo))
        return (sema_val, (sema_val - sema_ci_lo))

    # Adaptive padding: extend ymax / ymin to fit % labels (and stars).
    for _ in range(2):
        required_tops: list[float] = []
        required_bottoms: list[float] = []
        for _w in (24, 72):
            _higher = _higher_arm_at(_w)
            _lower = _lower_arm_at(_w)
            _has_star = show_fdr_stars and bool(_fdr_stars_str(fdr_per_week.get(_w)))
            if _higher is not None:
                _hval, _hci_half = _higher
                _above_pts = (
                    _ci_half_to_pts(_hci_half)
                    + PCT_OFFSET_PAD_PTS
                    + LABEL_FONT_PTS
                    + STAR_MARGIN_PTS
                )
                if _has_star and stars_above:
                    _above_pts += STAR_GAP_PTS + STAR_FONT_PTS
                required_tops.append(_hval + _above_pts / axes_height_pts * y_range)
            if _lower is not None:
                _lval, _lci_half = _lower
                _below_pts = (
                    _ci_half_to_pts(_lci_half)
                    + PCT_OFFSET_PAD_PTS
                    + LABEL_FONT_PTS
                    + STAR_MARGIN_PTS
                )
                if _has_star and not stars_above:
                    _below_pts += STAR_GAP_PTS + STAR_FONT_PTS
                required_bottoms.append(_lval - _below_pts / axes_height_pts * y_range)
        changed = False
        if required_tops:
            needed_ymax = max(required_tops)
            if needed_ymax > ymax:
                ymax = needed_ymax
                changed = True
        if required_bottoms:
            needed_ymin = min(required_bottoms)
            if needed_ymin < ymin:
                ymin = needed_ymin
                changed = True
        if not changed:
            break
        ax.set_ylim(ymin, ymax)
        y_range = max(ymax - ymin, 1e-6)

    for w in (24, 72):
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            continue
        tzp_val = data[("TZP", w)][0]
        sema_val = data[("SEMA", w)][0]
        tzp_above = tzp_val > sema_val
        for arm_short, val, place_above in (
            ("TZP", tzp_val, tzp_above),
            ("SEMA", sema_val, not tzp_above),
        ):
            _pcbl, _se, ci_lo, ci_hi = data[(arm_short, w)]
            ci_half = (ci_hi - ci_lo) / 2
            offset_pts = _ci_half_to_pts(ci_half) + PCT_OFFSET_PAD_PTS
            y_off = offset_pts if place_above else -offset_pts
            va = "bottom" if place_above else "top"
            ax.annotate(
                f"{val:+.0f}%",
                (w, val),
                xytext=(0, y_off),
                textcoords="offset points",
                fontsize=LABEL_FONT_PTS,
                color=ARM_COLORS[arm_short],
                fontweight="bold",
                ha="center",
                va=va,
                zorder=15,
            )

    # FDR stars (across-treatment) — direction follows the trajectory class.
    if show_fdr_stars and star_marker is not None:
        for w in (24, 72):
            stars = _fdr_stars_str(fdr_per_week.get(w))
            if not stars:
                continue
            anchor = _higher_arm_at(w) if stars_above else _lower_arm_at(w)
            if anchor is None:
                continue
            anchor_val, anchor_ci_half = anchor
            star_off = (
                _ci_half_to_pts(anchor_ci_half)
                + PCT_OFFSET_PAD_PTS
                + LABEL_FONT_PTS
                + STAR_GAP_PTS
            )
            y_off_pts = star_off if stars_above else -star_off
            va = "bottom" if stars_above else "top"
            ax.annotate(
                stars,
                xy=(w, anchor_val),
                xytext=(0, y_off_pts),
                textcoords="offset points",
                fontsize=STAR_FONT_PTS,
                color=c["black"],
                fontweight="bold",
                ha="center",
                va=va,
                zorder=16,
            )

    ax.set_xticks([0, 24, 72])

    if show_x:
        ax.set_xticklabels(["0", "24", "72"], fontsize=6)
        ax.tick_params(axis="x", length=2, width=0.4, pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not_covered_by is not None and data:
        from matplotlib.transforms import blended_transform_factory
        from matplotlib.patheffects import withStroke

        # Determine the best corner for the note using data context:
        #
        # Upward trajectories (non-Transient ↑): top-left — baseline is near
        #   the bottom, leaving the top-left corner clear.
        # Transient ↑: bottom-left — the Wk24 peak crowds the top-left.
        # Downward trajectories where data is already well below baseline by
        #   Wk24 (min_wk24_frac ≤ 0.25): top-right (x=72) — both arms have
        #   fallen to the bottom of the panel from Wk24 onward, so the only
        #   clear space at the left is above y=0, which is a thin sliver (
        #   y0_frac ≈ 0.90+); using x=72, top-right avoids both the label
        #   cluster and the y=0 line.
        # Other downward / reversal: bottom-left — data still near centre at
        #   Wk24, leaving the bottom-left corner open.
        vals_with_err = [0.0]
        for _k, (_v, _se, _ci_lo, _ci_hi) in data.items():
            vals_with_err.extend([_ci_lo, _ci_hi])
        _ymin_n = min(vals_with_err) - 0.10 * (max(vals_with_err) - min(vals_with_err))
        _ymax_n = max(vals_with_err) + 0.10 * (max(vals_with_err) - min(vals_with_err))
        _yr_n = max(_ymax_n - _ymin_n, 1e-6)
        wk24_vals = [data[_k][0] for _k in data if _k[1] == 24]
        min_wk24_frac = (
            min((_v - _ymin_n) / _yr_n for _v in wk24_vals) if wk24_vals else 0.5
        )

        is_up = trajectory is not None and "↑" in trajectory
        is_transient_up = is_up and "Transient" in trajectory
        is_down_steep = (
            trajectory is not None
            and "↑" not in trajectory
            and min_wk24_frac <= 0.25
        )

        if is_transient_up:
            x_data, ha, y_axes, va = 0, "left", 0.04, "bottom"
        elif is_up:
            x_data, ha, y_axes, va = 0, "left", 0.96, "top"
        elif is_down_steep:
            x_data, ha, y_axes, va = 80, "right", 0.96, "top"
        else:
            x_data, ha, y_axes, va = 0, "left", 0.04, "bottom"

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_data,
            y_axes,
            f"Not covered\nby {not_covered_by}",
            transform=trans,
            ha=ha,
            va=va,
            fontsize=5,
            color=c["black"],
            style="italic",
            linespacing=0.95,
            clip_on=False,
            path_effects=[
                withStroke(linewidth=1.5, foreground="white", alpha=0.45)
            ],
        )

    return trajectory


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
# Layout budget for the line section (inches). A single 2-row x 4-col grid:
# row 0 = Olink, row 1 = SomaScan; columns 0-1 = the cross-platform pair (gene
# named once per column), columns 2-3 = platform-specific (each panel named).
# Panels use the standard PCBL aspect; everything else is derived so the block
# fits whatever subfigure height the composite hands us.
LINE_ASPECT = 1.21  # panel width / height (matches TRAJ_PANEL_ASPECT)
LINE_TOP_MARGIN = 0.50  # above row 0 (block header + gene name + banner)
LINE_WITHIN = 0.55  # between rows (matches supp_fig5_tzp_induced hspace*panel_h)
LINE_BOTTOM_MARGIN = 0.46  # below row 1 (x-tick labels + Week + legend)
LINE_LEFT_IN = 0.46  # y-axis label + platform tag + tick numbers (tight gutter)
LINE_RIGHT_IN = 0.12
LINE_LABEL_X = 0.016  # shared far-left x for the y-label and platform tags
LINE_WSPACE_FRAC = 0.35  # within-block column gap (×panel_w), matches supp_fig5_tzp_induced
LINE_BLOCK_GAP_FRAC = 0.50  # extra gap between the two blocks (×panel_w)


def _draw_subplot_header(ax, gene, trajectory, *, show_name: bool) -> None:
    """Trajectory-class banner above a panel, with an optional gene name."""
    if trajectory is None:
        return
    banner_color = TRAJECTORY_COLORS.get(trajectory, "#888888")
    r_, g_, b_ = mpl.colors.to_rgb(banner_color)
    lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
    text_color = "white" if lum < 0.45 else c["black"]
    ax.add_patch(
        mpatches.Rectangle(
            (0, 1.04),
            1.0,
            0.12,
            facecolor=banner_color,
            edgecolor=c["black"],
            linewidth=0.4,
            transform=ax.transAxes,
            zorder=20,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        1.10,
        trajectory,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=FS_BODY,
        color=text_color,
        fontweight="bold",
        zorder=21,
        clip_on=False,
    )
    if show_name:
        ax.text(
            0.5,
            1.20,
            gene,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FS_EMPH,
            color=c["black"],
            fontweight="bold",
            clip_on=False,
        )


def render(
    parent_fig=None,
    save: bool = True,
    panel_label: str | None = None,
    panels: dict[str, list[str]] | None = None,
    out_name: str = "trajectory_lines",
) -> None:
    """Render the trajectory-line section as a single 2-row x 4-col grid.

    Row 0 = Olink, row 1 = SomaScan. Columns 0-1 ("Both platforms") show the
    cross-validated pair with the gene named once per column; columns 2-3
    ("Platform-specific") name each panel. A vertical divider separates the two
    blocks. Only the SomaScan row carries x-tick labels. Every panel uses the
    standard PCBL aspect (~1.21).

    `panels` overrides the default gene set (used by supp_fig4_trajectory_lines.py);
    it must have the same shape as PANELS (first N_CROSS columns cross-platform).
    """
    panels = panels or PANELS
    apply_lilly_theme()
    if parent_fig is None:
        fig = plt.figure(figsize=(NAT_W2, 3.2))
    else:
        fig = parent_fig

    fig.canvas.draw()
    H_in = fig.bbox.height / fig.dpi
    W_in = fig.bbox.width / fig.dpi

    rows = [
        ("olink", panels["olink"]),
        ("soma", panels["soma"]),
    ]
    _OTHER_KEY = {"olink": "soma", "soma": "olink"}
    _OTHER_LABEL = {"olink": "SomaScan", "soma": "Olink"}
    _PLATFORM_LABEL = {"olink": "Olink", "soma": "SomaScan"}
    coverage = _genes_per_platform()

    # --- Derive panel geometry from the WIDTH budget (so columns get the same
    #     ~0.35*panel_w gap as supp_fig5_tzp_induced, never a negative/crammed gap).
    #     Panel height follows from the aspect; the subfigure height is sized to
    #     match in render_full, so any leftover vertical space is small. ---------
    usable_w = W_in - LINE_LEFT_IN - LINE_RIGHT_IN
    # 4 panels, 2 within-block gaps + 1 (wider) block gap.
    panel_w = usable_w / (4 + 2 * LINE_WSPACE_FRAC + LINE_BLOCK_GAP_FRAC)
    panel_h = panel_w / LINE_ASPECT
    base_gap = LINE_WSPACE_FRAC * panel_w
    block_gap = LINE_BLOCK_GAP_FRAC * panel_w
    left = LINE_LEFT_IN / W_in
    right = 1.0 - LINE_RIGHT_IN / W_in

    def _yt(d_in: float) -> float:
        return 1.0 - d_in / H_in

    top = _yt(LINE_TOP_MARGIN)
    bot = _yt(LINE_TOP_MARGIN + 2 * panel_h + LINE_WITHIN)

    # Two side-by-side 2x2 gridspecs (cross block | specific block) so the gap
    # between them can be wider than the within-block column gap.
    block_w = 2 * panel_w + base_gap
    a_left = left
    a_right = left + block_w / W_in
    b_left = a_right + block_gap / W_in
    b_right = b_left + block_w / W_in
    gs_a = fig.add_gridspec(
        2,
        2,
        left=a_left,
        right=a_right,
        top=top,
        bottom=bot,
        wspace=LINE_WSPACE_FRAC,
        hspace=LINE_WITHIN / panel_h,
    )
    gs_b = fig.add_gridspec(
        2,
        2,
        left=b_left,
        right=b_right,
        top=top,
        bottom=bot,
        wspace=LINE_WSPACE_FRAC,
        hspace=LINE_WITHIN / panel_h,
    )

    # --- Populate panels ----------------------------------------------------
    row_records: list[tuple[str, list[tuple]]] = []
    for r, (platform, genes) in enumerate(rows):
        is_lower_row = r == 1
        other_genes = coverage[_OTHER_KEY[platform]]
        other_label = _OTHER_LABEL[platform]
        row_axes = []
        for ci, gene in enumerate(genes):
            is_specific = ci >= N_CROSS
            gs = gs_b if is_specific else gs_a
            sub_c = ci - N_CROSS if is_specific else ci
            ax = fig.add_subplot(gs[r, sub_c])
            not_covered = (
                other_label if is_specific and gene not in other_genes else None
            )
            # Every panel carries its own week x-ticks and gene-name title.
            trajectory = _draw_panel(
                ax,
                platform,
                gene,
                show_x=True,
                not_covered_by=not_covered,
            )
            _draw_subplot_header(ax, gene, trajectory, show_name=True)
            row_axes.append((ax, gene, trajectory))
        row_records.append((platform, row_axes))

    # --- Per-row platform tag (rotated, left gutter) ------------------------
    for platform, row_axes in row_records:
        ys = [a.get_position().y0 + a.get_position().height / 2 for a, *_ in row_axes]
        row_cy = sum(ys) / len(ys)
        fig.text(
            LINE_LABEL_X,
            row_cy,
            _PLATFORM_LABEL[platform],
            fontsize=FS_EMPH,
            fontweight="bold",
            color=c["black"],
            rotation=90,
            ha="center",
            va="center",
        )

    # --- Block headers + vertical divider -----------------------------------
    def _cols(ci_lo, ci_hi):
        return [row_records[0][1][ci][0] for ci in range(ci_lo, ci_hi)] + [
            row_records[1][1][ci][0] for ci in range(ci_lo, ci_hi)
        ]

    def _block_header(axes, text: str) -> float:
        xs0 = min(a.get_position().x0 for a in axes)
        xs1 = max(a.get_position().x1 for a in axes)
        y1 = max(a.get_position().y1 for a in axes)
        header_y = y1 + (0.26 * panel_h + 0.07) / H_in
        fig.text(
            (xs0 + xs1) / 2,
            header_y,
            text,
            ha="center",
            va="bottom",
            fontsize=FS_EMPH,
            fontweight="bold",
        )
        return header_y

    cross_axes = _cols(0, N_CROSS)
    spec_axes = _cols(N_CROSS, 4)
    head_y = _block_header(cross_axes, "Both platforms")
    _block_header(spec_axes, "Platform-specific")

    # Vertical divider in the block gap.
    div_x = (
        max(a.get_position().x1 for a in cross_axes)
        + min(a.get_position().x0 for a in spec_axes)
    ) / 2
    div_top = head_y - 0.005
    div_bot = min(a.get_position().y0 for a in cross_axes) - 0.01
    fig.add_artist(
        mlines.Line2D(
            [div_x, div_x],
            [div_bot, div_top],
            transform=getattr(fig, "transSubfigure", fig.transFigure),
            color=c["bold_grey"],
            linewidth=0.5,
            zorder=2,
            clip_on=False,
        )
    )

    if panel_label:
        fig.text(
            min(a.get_position().x0 for a in cross_axes) - 0.045,
            head_y,
            panel_label,
            ha="left",
            va="bottom",
            fontsize=FS_PANEL,
            fontweight="bold",
        )

    # --- Shared axis labels -------------------------------------------------
    fig.text(
        LINE_LABEL_X,
        0.5,
        "% Change from\nbaseline (95% CI)",
        fontsize=FS_EMPH,
        ha="center",
        va="center",
        rotation=90,
        linespacing=1.2,
    )
    fig.text(0.5, _yt(H_in - 0.10), "Week", fontsize=FS_EMPH, ha="center", va="center")

    legend_handles = [
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS["TZP"],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS["TZP"],
        ),
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS["SEMA"],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS["SEMA"],
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(right, _yt(H_in - 0.10)),
        ncol=2,
        fontsize=FS_NARR,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    if save:
        out_path = str(THIS_DIR / out_name)
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved -> {out_path}.pdf")
        plt.close()


if __name__ == "__main__":
    init_figure_theme()
    render()
