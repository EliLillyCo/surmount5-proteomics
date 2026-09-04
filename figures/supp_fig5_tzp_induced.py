"""Supplemental figure: TZP-induced biomarkers (proteins increased by both
treatments, more strongly with TZP than SEMA).

Per-marker %CFB line plots (Wk0/24/72) for TZP and SEMA. Shared proteins
(measured on both platforms) occupy identical grid positions so the reader
can compare Olink and SomaScan results at the same eye level; platform-
specific proteins (KLK4/CD276/CKB on Olink; LEPROT/MYF5 on Soma) fill the
remaining cells.

FDR stars are drawn above the higher arm's percentage label to follow the
upward direction of change (mirror of the suppressed figure, which places
stars below the lower arm).

Layout: 4 rows × 4 cols total — Olink left 2 cols, SomaScan right 2 cols.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from _internal._common import (  # noqa: E402
    ANALYSIS_DIR,
    ARM_COLORS,
    ARM_LABELS,
    BANNER_HEIGHT_IN,
    COVAR,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_NARR,
    FS_PANEL,
    c,
    TRAJECTORY_COLORS,
    apply_lilly_theme,
    init_figure_theme,
    save_figure,
)
from paths import ensure_public_alias, public_mmrm_dir  # noqa: E402

ARMS = {"TZP": "TZP15mgorMTD", "SEMA": "SEMA2.4mgorMTD"}
WEEKS = [0, 24, 72]

HEADER_MARKER_Y = 1.17
HEADER_GENE_Y = 1.275


# ---------------------------------------------------------------------------
# Marker layout — slot index → (marker, gene). Slots fill row-major within
# each 4×2 sub-grid: slot 0=(r0,c0), 1=(r0,c1), 2=(r1,c0), ..., 7=(r3,c1).
#
# Shared genes occupy slots 0–4 in the same positions on both panels (sorted
# by best min FDR across platforms). Platform-specific genes fill the rest
# (sorted by their own min FDR).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarkerSpec:
    marker: str
    gene: str


OLINK_LAYOUT: list[MarkerSpec | None] = [
    MarkerSpec("OID45019", "WFIKKN2"),  # shared
    MarkerSpec("OID44747", "LEPR"),  # shared
    MarkerSpec("OID45152", "IGFBP2"),  # shared
    MarkerSpec("OID45151", "IGFBP1"),  # shared
    MarkerSpec("OID45411", "SHBG"),  # shared
    MarkerSpec("OID43899", "KLK4"),  # Olink-only
    MarkerSpec("OID44508", "CD276"),  # Olink-only
    MarkerSpec("OID45073", "CKB"),  # Olink-only
]

SOMA_LAYOUT: list[MarkerSpec | None] = [
    MarkerSpec("3235-50", "WFIKKN2"),  # shared (best aptamer)
    MarkerSpec("5400-52", "LEPR"),  # shared
    MarkerSpec("8469-41", "IGFBP2"),  # shared (best of 4 aptamers)
    MarkerSpec("13741-36", "IGFBP1"),  # shared (best of 2 aptamers)
    MarkerSpec("4929-55", "SHBG"),  # shared (best of 2 aptamers)
    MarkerSpec("33726-54", "LEPROT"),  # Soma-only
    MarkerSpec("28044-13", "MYF5"),  # Soma-only
    None,  # empty slot
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_pcbl(platform: str, arm_full: str, week: int) -> pl.DataFrame:
    path = ensure_public_alias(
        public_mmrm_dir(platform)
        / "finalRes"
        / "pcblRes"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_PCBLRes_{arm_full}@{week}_py.csv"
    )
    return pl.read_csv(path).select("marker", "PCBL", "SE_PCBL", "PCBL_lower_CI", "PCBL_upper_CI", "fdr")


@lru_cache(maxsize=4)
def _load_actrt(platform: str, week: int) -> pl.DataFrame:
    path = ensure_public_alias(
        public_mmrm_dir(platform)
        / "finalRes"
        / "acTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_acrossTrts_resCmps_TZP15mgorMTDVSSEMA2.4mgorMTD@{week}_py.csv"
    )
    return pl.read_csv(path).select("marker", "FC", "fdr")


@lru_cache(maxsize=4)
def _load_trajectory(platform: str) -> pl.DataFrame:
    return pl.read_parquet(ANALYSIS_DIR / "outputs" / f"trajectory_{platform}.parquet")


def _marker_pcbl_data(
    platform: str, marker: str
) -> dict[tuple[str, int], tuple[float, float, float, float, float]]:
    """{(arm, week): (PCBL, SE, lower_CI, upper_CI, fdr)} for Wk24/72."""
    data: dict[tuple[str, int], tuple[float, float, float, float, float]] = {}
    for arm_short, arm_full in ARMS.items():
        for w in (24, 72):
            df = _load_pcbl(platform, arm_full, w)
            row = df.filter(pl.col("marker") == marker)
            if row.is_empty():
                continue
            data[(arm_short, w)] = (
                row["PCBL"][0],
                row["SE_PCBL"][0],
                row["PCBL_lower_CI"][0],
                row["PCBL_upper_CI"][0],
                row["fdr"][0],
            )
    return data


def _across_treatment_fdr(
    platform: str, marker: str
) -> tuple[float | None, float | None]:
    """Return (fdr_24, fdr_72) for the TZP vs SEMA across-treatment contrast."""
    out: list[float | None] = []
    for w in (24, 72):
        df = _load_actrt(platform, w)
        row = df.filter(pl.col("marker") == marker)
        out.append(row["fdr"][0] if not row.is_empty() else None)
    return tuple(out)  # type: ignore[return-value]


def _trajectory_for(platform: str, marker: str) -> str | None:
    df = _load_trajectory(platform)
    row = df.filter(pl.col("marker") == marker)
    if row.is_empty():
        return None
    return row["trajectory_TZP"][0]


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------
def _fdr_stars(fdr: float | None) -> str:
    if fdr is None:
        return ""
    if fdr < 1e-3:
        return "***"
    if fdr < 1e-2:
        return "**"
    if fdr < 5e-2:
        return "*"
    return ""


def _draw_panel(
    ax,
    platform: str,
    marker: str,
    fdr_24_between: float | None,
    fdr_72_between: float | None,
) -> None:
    data = _marker_pcbl_data(platform, marker)

    ax.axhline(0, color=c["bold_grey"], linewidth=0.4, zorder=1)

    for arm_short in ("SEMA", "TZP"):
        y = [0.0]
        yerr_lo = [0.0]
        yerr_hi = [0.0]
        for w in (24, 72):
            if (arm_short, w) in data:
                pcbl, _se, ci_lo, ci_hi, _ = data[(arm_short, w)]
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
    # Compute data extent including error bars and the y=0 baseline.
    _y_with_err: list[float] = [0.0]
    for _arm in ("TZP", "SEMA"):
        for _w in (24, 72):
            if (_arm, _w) in data:
                _v, _se, _ci_lo, _ci_hi, _ = data[(_arm, _w)]
                _y_with_err.extend([_ci_lo, _ci_hi])
    _data_min = min(_y_with_err)
    _data_max = max(_y_with_err)
    _data_range = max(_data_max - _data_min, 1e-6)
    # Tight default padding; the adaptive pass below extends ymax to fit FDR
    # stars (drawn above the higher arm) and the lower arm's % label.
    ymin = _data_min - 0.10 * _data_range
    ymax = _data_max + 0.10 * _data_range
    ax.set_ylim(ymin, ymax)
    y_range = max(ymax - ymin, 1e-6)
    axes_height_pts = ax.bbox.height / ax.figure.dpi * 72

    def _ci_half_to_pts(ci_half: float) -> float:
        return ci_half / y_range * axes_height_pts

    LABEL_FONT_PTS = FS_BODY
    PCT_OFFSET_PAD_PTS = 4.0
    STAR_FONT_PTS = FS_EMPH
    STAR_GAP_PTS = 3.0
    STAR_MARGIN_PTS = 3.0

    def _higher_arm_at(w: int) -> tuple[float, float] | None:
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val = data[("TZP", w)][0]
        sema_val = data[("SEMA", w)][0]
        if tzp_val > sema_val:
            return tzp_val, (data[("TZP", w)][3] - tzp_val)
        return sema_val, (data[("SEMA", w)][3] - sema_val)

    def _lower_arm_at(w: int) -> tuple[float, float] | None:
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val = data[("TZP", w)][0]
        sema_val = data[("SEMA", w)][0]
        if tzp_val < sema_val:
            return tzp_val, (tzp_val - data[("TZP", w)][2])
        return sema_val, (sema_val - data[("SEMA", w)][2])

    # Adaptive padding: extend ymax to fit FDR stars above the higher arm,
    # and extend ymin to fit the lower arm's % label.
    for _ in range(2):
        required_tops: list[float] = []
        required_bottoms: list[float] = []
        for _w in (24, 72):
            _higher = _higher_arm_at(_w)
            _lower = _lower_arm_at(_w)
            _fdr = fdr_24_between if _w == 24 else fdr_72_between
            _has_star = bool(_fdr_stars(_fdr))
            if _higher is not None:
                _hval, _hci_half = _higher
                _above_pts = (
                    _ci_half_to_pts(_hci_half)
                    + PCT_OFFSET_PAD_PTS
                    + LABEL_FONT_PTS
                    + STAR_MARGIN_PTS
                )
                if _has_star:
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
            ci_lo = data[(arm_short, w)][2]
            ci_hi = data[(arm_short, w)][3]
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

    # FDR stars sit immediately above the higher arm's percentage label —
    # data is trending upward, so the star follows the direction of change.
    for w in (24, 72):
        fdr = fdr_24_between if w == 24 else fdr_72_between
        stars = _fdr_stars(fdr)
        if not stars:
            continue
        higher = _higher_arm_at(w)
        if higher is None:
            continue
        higher_val, higher_ci_half = higher
        label_offset_pts = _ci_half_to_pts(higher_ci_half) + PCT_OFFSET_PAD_PTS
        star_offset_pts = label_offset_pts + LABEL_FONT_PTS + STAR_GAP_PTS
        ax.annotate(
            stars,
            xy=(w, higher_val),
            xytext=(0, star_offset_pts),
            textcoords="offset points",
            fontsize=STAR_FONT_PTS,
            color=c["black"],
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=16,
        )

    ax.set_xticks([0, 24, 72])
    ax.set_xticklabels(["0", "24", "72"], fontsize=FS_BODY)
    ax.tick_params(axis="x", length=2, width=0.4, pad=1)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_subplot_header(ax, gene: str, marker: str, trajectory: str | None) -> None:
    """Trajectory pill + marker ID + gene name above subplot (axes coords).
    Matches the supp_fig7_exocrine_pancreas / Fig 1 trajectory-line styling."""
    if trajectory and trajectory != "Null":
        banner_color = TRAJECTORY_COLORS.get(trajectory, "#888888")
        r_, g_, b_ = mpl.colors.to_rgb(banner_color)
        lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
        text_color = "white" if lum < 0.45 else c["black"]
        ax.add_patch(
            mpatches.Rectangle(
                (0, 1.03),
                1.0,
                0.11,
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
            1.085,
            trajectory,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_NARR,
            color=text_color,
            fontweight="bold",
            zorder=21,
            clip_on=False,
        )

    ax.text(
        0.5,
        HEADER_MARKER_Y,
        marker,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FS_AUX,
        color=c["bold_grey"],
        clip_on=False,
    )
    ax.text(
        0.5,
        HEADER_GENE_Y,
        gene,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FS_EMPH,
        color=c["black"],
        fontweight="bold",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def render() -> None:
    init_figure_theme()
    apply_lilly_theme()

    NROWS = 4
    PANEL_COLS = 2
    NCOLS = PANEL_COLS * 2  # Olink + Soma side by side

    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(
        NROWS,
        NCOLS,
        left=0.075,
        right=0.985,
        top=0.78,
        bottom=0.075,
        wspace=0.35,
        hspace=0.50,
    )

    def _populate(
        panel: list[MarkerSpec | None], platform: str, col_offset: int
    ) -> list:
        axes_list: list = []
        for slot, spec in enumerate(panel):
            r, sub_c = divmod(slot, PANEL_COLS)
            ax = fig.add_subplot(gs[r, col_offset + sub_c])
            if spec is None:
                ax.set_axis_off()
                continue
            fdr_24, fdr_72 = _across_treatment_fdr(platform, spec.marker)
            traj = _trajectory_for(platform, spec.marker)
            _draw_panel(ax, platform, spec.marker, fdr_24, fdr_72)
            _draw_subplot_header(ax, spec.gene, spec.marker, traj)
            axes_list.append(ax)
        return axes_list

    olink_axes = _populate(OLINK_LAYOUT, "olink", col_offset=0)
    soma_axes = _populate(SOMA_LAYOUT, "soma", col_offset=PANEL_COLS)

    # --- Platform banners + panel labels ---
    def _platform_banner(axes: list, label: str, panel_label: str) -> None:
        if not axes:
            return
        x0 = min(a.get_position().x0 for a in axes)
        x1 = max(a.get_position().x1 for a in axes)
        # Banner sits above the gene-name text of the topmost subplots.
        top_subs = [
            a
            for a in axes
            if a.get_position().y1 == max(b.get_position().y1 for b in axes)
        ]
        gene_baseline_y = max(
            a.get_position().y1 + (HEADER_GENE_Y - 1.0) * a.get_position().height
            for a in top_subs
        )
        gene_top_y = gene_baseline_y + FS_EMPH / 72.0 / fig.get_figheight()
        banner_height = BANNER_HEIGHT_IN / fig.get_figheight()
        banner_y = gene_top_y + 0.022
        banner = mpatches.FancyBboxPatch(
            (x0, banner_y),
            x1 - x0,
            banner_height,
            boxstyle="square,pad=0",
            facecolor="#E8E8E8",
            edgecolor="none",
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(banner)
        fig.text(
            (x0 + x1) / 2,
            banner_y + banner_height / 2,
            label,
            fontsize=FS_EMPH,
            fontweight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            x0 - 0.022,
            banner_y + banner_height + 0.003,
            panel_label,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    _platform_banner(olink_axes, "Olink", "a")
    _platform_banner(soma_axes, "SomaScan", "b")

    # --- Y-axis label, hugging the plot column ---
    fig.text(
        0.020,
        0.43,
        "% Change from baseline (95% CI)",
        fontsize=FS_EMPH,
        ha="center",
        va="center",
        rotation=90,
    )

    # --- Bottom strip: FDR legend (left), Week label (center), arm legend (right) ---
    STRIP_Y = 0.028
    plot_center_x = (0.075 + 0.985) / 2

    fig.text(
        0.075,
        STRIP_Y,
        "Across treatment FDR: * <0.05   ** <0.01   *** <0.001",
        fontsize=FS_BODY,
        color=c["black"],
        ha="left",
        va="center",
        style="italic",
    )
    fig.text(plot_center_x, STRIP_Y, "Week", fontsize=FS_EMPH, ha="center", va="center")

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
        bbox_to_anchor=(0.985, STRIP_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=FS_EMPH,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
    )

    out_path = str(THIS_DIR / "supp_fig5_tzp_induced")
    save_figure(fig, out_path, formats=("pdf",))
    print(f"Saved -> {out_path}.pdf")
    plt.close()


if __name__ == "__main__":
    render()
