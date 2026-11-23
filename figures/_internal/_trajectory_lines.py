"""Trajectory line panels, the TZP | SEMA class banner and the shared trajectory-figure parts.

Each panel shows %CFB at every proteomics week for TZP and SEMA with 95% CI error bars, the
arms' % labels and the between-treatment FDR stars (``draw_lines``). A colored banner above each
panel encodes the assay's trajectory class: one banner when TZP and SEMA share a class, split
TZP | SEMA halves when they differ. Fig 1 and Fig 5 draw a gene's panel (``draw_gene_panel``);
the supplementary trajectory figures draw an assay's panel with its header
(``draw_marker_panel``) and share the platform banner, the y-axis label and the bottom strip.
The panels read their data through ``_trajectory_data``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.transforms import blended_transform_factory

from analysis.design import ARMS, EARLY_WEEK, POST_BASELINE_WEEKS, WEEKS

from . import _trajectory_data as td
from ._common import (
    BETWEEN_TREATMENT_FDR_KEY,
    ARM_COLORS,
    ARM_LABELS,
    BANNER_HEIGHT_IN,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_MIN,
    FS_PANEL,
    TRAJ_PANEL_ASPECT,
    TRAJECTORY_COLORS,
    c,
    fdr_stars,
    nature_figsize,
)


# Panel text and spacing, in points.
LABEL_FONT_PTS = FS_BODY  # % labels
PCT_OFFSET_PAD_PTS = 4.0  # CI end -> % label
STAR_FONT_PTS = FS_EMPH
STAR_GAP_PTS = 3.0  # % label -> stars
STAR_MARGIN_PTS = 3.0  # stars or label -> axes edge
# Default headroom above the data when only the stars below are fitted (``fit_labels=False``).
STARS_BELOW_TOP_PAD = 0.18

YLABEL = "% Change from baseline (95% CI)"


@dataclass(frozen=True)
class MarkerSpec:
    """One panel: the assay ID (``marker``) and the gene symbol it is labeled with."""

    marker: str
    gene: str


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------
def draw_lines(
    ax,
    data: td.PanelData,
    fdr_by_week: dict[int, float | None],
    *,
    stars_above: bool,
    fit_labels: bool = True,
    gene: str | None = None,
    platform: str | None = None,
) -> None:
    """Draw one assay's %CFB lines, % labels and between-treatment FDR stars, and style the axes.

    The y-range starts at the data and CI extent (with the 0 baseline) padded by 10%, and is
    then extended until the labels fit.

    Args:
        ax: The panel axes.
        data: ``{(arm, week): (PCBL, SE, lower CI, upper CI)}`` (``_trajectory_data.marker_pcbl``).
        fdr_by_week: Between-treatment FDR per post-baseline week; a week below the threshold
            gets stars.
        stars_above: Stars above the higher arm's % label; False puts them below the lower arm's.
        fit_labels: Extend both ends of the y-range until every % label and star fits. False
            pads the top by ``STARS_BELOW_TOP_PAD`` and extends only the bottom, to fit the
            stars below the lower arm.
        gene: Gene symbol to identify missing results in the error message.
        platform: Platform to identify missing results in the error message.
    """
    for arm_short in ("SEMA", "TZP"):
        for week in POST_BASELINE_WEEKS:
            if (arm_short, week) not in data:
                raise ValueError(
                    f"Missing trajectory result for gene {gene}, platform {platform}, "
                    f"arm {arm_short}, week {week}"
                )

    ax.axhline(0, color=c["bold_grey"], linewidth=0.4, zorder=1)

    for arm_short in ("SEMA", "TZP"):
        y = [0.0]
        yerr_lo = [0.0]
        yerr_hi = [0.0]
        for w in POST_BASELINE_WEEKS:
            pcbl, _se, ci_lo, ci_hi = data[(arm_short, w)]
            y.append(pcbl)
            yerr_lo.append(pcbl - ci_lo)
            yerr_hi.append(ci_hi - pcbl)
        ax.errorbar(
            WEEKS,
            y,
            yerr=[yerr_lo, yerr_hi],
            color=ARM_COLORS[arm_short],
            linewidth=0.9,
            marker="o",
            markersize=2.8,
            capsize=1.5,
            capthick=0.5,
            elinewidth=0.5,
            zorder=10 if arm_short == "TZP" else 9,
        )

    ax.set_xlim(-6, 80)
    # Data extent including error bars and the y=0 baseline.
    y_with_err: list[float] = [0.0]
    for arm in tuple(ARMS):
        for w in POST_BASELINE_WEEKS:
            if (arm, w) in data:
                _v, _se, ci_lo, ci_hi = data[(arm, w)]
                y_with_err.extend([ci_lo, ci_hi])
    data_min = min(y_with_err)
    data_max = max(y_with_err)
    data_range = max(data_max - data_min, 1e-6)
    top_pad = 0.10 if fit_labels else STARS_BELOW_TOP_PAD
    ymin = data_min - 0.10 * data_range
    ymax = data_max + top_pad * data_range
    ax.set_ylim(ymin, ymax)
    axes_height_pts = ax.bbox.height / ax.figure.dpi * 72

    def higher_arm_at(w: int) -> tuple[float, float] | None:
        """(value, upper CI half-width) of the arm with the larger change at ``w``."""
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val, _tzp_se, _tzp_ci_lo, tzp_ci_hi = data[("TZP", w)]
        sema_val, _sema_se, _sema_ci_lo, sema_ci_hi = data[("SEMA", w)]
        if tzp_val > sema_val:
            return tzp_val, (tzp_ci_hi - tzp_val)
        return sema_val, (sema_ci_hi - sema_val)

    def lower_arm_at(w: int) -> tuple[float, float] | None:
        """(value, lower CI half-width) of the arm with the smaller change at ``w``."""
        if ("TZP", w) not in data or ("SEMA", w) not in data:
            return None
        tzp_val, _tzp_se, tzp_ci_lo, _tzp_ci_hi = data[("TZP", w)]
        sema_val, _sema_se, sema_ci_lo, _sema_ci_hi = data[("SEMA", w)]
        if tzp_val < sema_val:
            return tzp_val, (tzp_val - tzp_ci_lo)
        return sema_val, (sema_val - sema_ci_lo)

    if fit_labels:
        y_range = _fit_labels(
            ax, ymin, ymax, axes_height_pts, higher_arm_at, lower_arm_at, fdr_by_week, stars_above
        )
    else:
        y_range = _fit_stars_below(ax, ymin, ymax, axes_height_pts, lower_arm_at, fdr_by_week)

    def ci_half_to_pts(ci_half: float) -> float:
        return ci_half / y_range * axes_height_pts

    for w in POST_BASELINE_WEEKS:
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
            offset_pts = ci_half_to_pts(ci_half) + PCT_OFFSET_PAD_PTS
            y_off = offset_pts if place_above else -offset_pts
            va = "bottom" if place_above else "top"
            ax.annotate(
                f"{val:+.0f}%" if round(val) != 0 else "0%",
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

    for w in POST_BASELINE_WEEKS:
        stars = fdr_stars(fdr_by_week.get(w))
        if not stars:
            continue
        anchor = higher_arm_at(w) if stars_above else lower_arm_at(w)
        if anchor is None:
            continue
        anchor_val, anchor_ci_half = anchor
        star_off = (
            ci_half_to_pts(anchor_ci_half) + PCT_OFFSET_PAD_PTS + LABEL_FONT_PTS + STAR_GAP_PTS
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

    ax.set_xticks(WEEKS)
    ax.set_xticklabels([str(w) for w in WEEKS], fontsize=FS_BODY)
    ax.tick_params(axis="x", length=2, width=0.4, pad=1)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fit_labels(
    ax, ymin, ymax, axes_height_pts, higher_arm_at, lower_arm_at, fdr_by_week, stars_above
) -> float:
    """Extend ``ymax`` above the higher arm and ``ymin`` below the lower arm until the % labels
    and the stars fit, in at most two passes; returns the final y-range."""
    y_range = max(ymax - ymin, 1e-6)
    for _ in range(2):
        required_tops: list[float] = []
        required_bottoms: list[float] = []
        for w in POST_BASELINE_WEEKS:
            higher = higher_arm_at(w)
            lower = lower_arm_at(w)
            has_star = bool(fdr_stars(fdr_by_week.get(w)))
            if higher is not None:
                hval, hci_half = higher
                above_pts = (
                    hci_half / y_range * axes_height_pts
                    + PCT_OFFSET_PAD_PTS
                    + LABEL_FONT_PTS
                    + STAR_MARGIN_PTS
                )
                if has_star and stars_above:
                    above_pts += STAR_GAP_PTS + STAR_FONT_PTS
                required_tops.append(hval + above_pts / axes_height_pts * y_range)
            if lower is not None:
                lval, lci_half = lower
                below_pts = (
                    lci_half / y_range * axes_height_pts
                    + PCT_OFFSET_PAD_PTS
                    + LABEL_FONT_PTS
                    + STAR_MARGIN_PTS
                )
                if has_star and not stars_above:
                    below_pts += STAR_GAP_PTS + STAR_FONT_PTS
                required_bottoms.append(lval - below_pts / axes_height_pts * y_range)
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
    return y_range


def _fit_stars_below(ax, ymin, ymax, axes_height_pts, lower_arm_at, fdr_by_week) -> float:
    """Extend ``ymin`` until the stars below the lower arm fit, in at most two passes; returns
    the final y-range."""
    y_range = max(ymax - ymin, 1e-6)
    for _ in range(2):
        required_bottoms: list[float] = []
        for w in POST_BASELINE_WEEKS:
            if not fdr_stars(fdr_by_week.get(w)):
                continue
            lower = lower_arm_at(w)
            if lower is None:
                continue
            lval, lci_half = lower
            below_pts = (
                lci_half / y_range * axes_height_pts
                + PCT_OFFSET_PAD_PTS
                + LABEL_FONT_PTS
                + STAR_GAP_PTS
                + STAR_FONT_PTS
                + STAR_MARGIN_PTS
            )
            required_bottoms.append(lval - below_pts / axes_height_pts * y_range)
        if not required_bottoms:
            break
        needed_ymin = min(required_bottoms)
        if needed_ymin < ymin:
            ymin = needed_ymin
            ax.set_ylim(ymin, ymax)
            y_range = max(ymax - ymin, 1e-6)
        else:
            break
    return y_range


def draw_gene_panel(ax, platform: str, gene: str, not_covered_by: str | None = None
) -> tuple[str, str] | None:
    """Draw a gene's panel from its assay with the largest change (``best_marker_for_gene``).

    Stars sit above the higher arm for a class that rises (``↑`` in the TZP class) and below the
    lower arm otherwise. A gene without an assay in the trajectory table gets a "(not measured)"
    note and no axes.

    Args:
        ax: The panel axes.
        platform: ``"olink"`` or ``"soma"``.
        gene: Gene symbol.
        not_covered_by: The other platform's name, printed as a "Not covered by" note in the
            corner the data leave clear; None draws no note.

    Returns:
        The assay's (TZP, SEMA) trajectory classes, or None when the gene is not measured.
    """
    marker = td.best_marker_for_gene(platform, gene)
    if marker is None:
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
    classes = td.trajectory_classes(platform, marker)
    trajectory = classes[0]
    data = td.marker_pcbl(platform, marker)
    draw_lines(
        ax,
        data,
        td.across_treatment_fdr(platform, marker),
        stars_above="↑" in trajectory,
        gene=gene,
        platform=platform,
    )
    if not_covered_by is not None and data:
        _draw_not_covered_note(ax, data, trajectory, not_covered_by)
    return classes


def _draw_not_covered_note(ax, data: td.PanelData, trajectory: str, not_covered_by: str) -> None:
    """Place "Not covered by <platform>" in the corner the data leave clear.

    Rising classes other than Transient take the top-left corner; Transient ↑ the bottom-left;
    a falling class whose arms are in the bottom quarter of the panel by the early week the
    top-right; any other class the bottom-left.
    """
    vals_with_err = [0.0]
    for _k, (_v, _se, ci_lo, ci_hi) in data.items():
        vals_with_err.extend([ci_lo, ci_hi])
    ymin_n = min(vals_with_err) - 0.10 * (max(vals_with_err) - min(vals_with_err))
    ymax_n = max(vals_with_err) + 0.10 * (max(vals_with_err) - min(vals_with_err))
    yr_n = max(ymax_n - ymin_n, 1e-6)
    early_vals = [data[k][0] for k in data if k[1] == EARLY_WEEK]
    min_early_frac = min((v - ymin_n) / yr_n for v in early_vals) if early_vals else 0.5

    is_up = "↑" in trajectory
    is_transient_up = is_up and "Transient" in trajectory
    is_down_steep = "↑" not in trajectory and min_early_frac <= 0.25

    if is_transient_up:
        x_data, ha, y_axes, va = 0, "left", 0.04, "bottom"
    elif is_up:
        x_data, ha, y_axes, va = 0, "left", 0.96, "top"
    elif is_down_steep:
        x_data, ha, y_axes, va = 80, "right", 0.96, "top"
    else:
        x_data, ha, y_axes, va = 0, "left", 0.04, "bottom"

    ax.text(
        x_data,
        y_axes,
        f"Not covered\nby {not_covered_by}",
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        ha=ha,
        va=va,
        fontsize=5,
        color=c["black"],
        style="italic",
        linespacing=0.95,
        clip_on=False,
        path_effects=[withStroke(linewidth=1.5, foreground="white", alpha=0.45)],
    )


def draw_marker_panel(
    ax, platform: str, spec: MarkerSpec, *, stars_above: bool = True, fit_labels: bool = True
) -> None:
    """Draw one assay's panel with its class banner, assay ID and gene name above.

    Args:
        ax: The panel axes.
        platform: ``"olink"`` or ``"soma"``.
        spec: The assay and its label.
        stars_above: See ``draw_lines``.
        fit_labels: See ``draw_lines``.
    """
    fdr_by_week = td.across_treatment_fdr(platform, spec.marker)
    classes = td.trajectory_classes(platform, spec.marker)
    draw_lines(
        ax,
        td.marker_pcbl(platform, spec.marker),
        fdr_by_week,
        stars_above=stars_above,
        fit_labels=fit_labels,
        gene=spec.gene,
        platform=platform,
    )
    _draw_subplot_header(ax, spec.gene, *(classes or (None, None)), marker=spec.marker)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
# Shared far-left x for the y-label and platform tags of the trajectory figures.
LINE_LABEL_X = 0.016


# Marker-ID and gene-name rows above the banner (figures that print the assay ID).
HEADER_MARKER_Y = 1.17
HEADER_GENE_Y = 1.275
BANNER_Y0 = 1.04  # banner bottom, axes coordinates
BANNER_H = 0.12
SPLIT_GAP = 0.01  # gap between the TZP and SEMA halves (axes width fraction)
SPLIT_PAD_PT = 1.0  # minimum text clearance inside a half
_CLASS_SHORT = (
    ("Late-onset", "Late"),
    ("Sustained", "Sust."),
    ("Progressive", "Prog."),
    ("Transient", "Trans."),
    ("Reversal", "Rev."),
)
LUM_WHITE_TEXT = 0.45  # banner colors darker than this get white text


def _class_short(cls: str) -> str:
    if cls == "Null":
        return "ns"
    for long, short in _CLASS_SHORT:
        cls = cls.replace(long, short)
    return cls


def _banner_text_color(color: str) -> str:
    r_, g_, b_ = mpl.colors.to_rgb(color)
    lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
    return "white" if lum < LUM_WHITE_TEXT else c["black"]


def _banner_patch(ax, x0, width, color, y0, height):
    return ax.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=color,
            edgecolor=c["black"],
            linewidth=0.4,
            transform=ax.transAxes,
            zorder=20,
            clip_on=False,
        )
    )


def _check_banner_fit(ax, text, x0: float, width: float) -> None:
    """Raise when a banner label spills outside its patch (axes-width fractions)."""
    ax_bb = ax.get_window_extent(ax.figure.canvas.get_renderer())
    txt_bb = text.get_window_extent(ax.figure.canvas.get_renderer())
    pad = SPLIT_PAD_PT * ax.figure.dpi / 72
    lo = ax_bb.x0 + x0 * ax_bb.width + pad
    hi = ax_bb.x0 + (x0 + width) * ax_bb.width - pad
    if txt_bb.x0 < lo or txt_bb.x1 > hi:
        raise ValueError(
            f"Banner text {text.get_text()!r} ({txt_bb.width:.1f}px) overflows its "
            f"half ({hi - lo:.1f}px) at {text.get_fontsize()} pt"
        )


def draw_class_banner(ax, tzp: str, sema: str) -> None:
    """Trajectory-class banner above a panel.

    Same class in both arms: one full-width banner with the full class name.
    Different classes: two halves, "TZP: <short>" | "SEMA: <short>", each colored
    by TRAJECTORY_COLORS (a Null half is blank, labeled "ns"). Halves use FS_MIN.
    Null in both arms draws no banner.
    """
    if tzp == sema == "Null":
        return
    if tzp == sema:
        color = TRAJECTORY_COLORS[tzp]
        _banner_patch(ax, 0.0, 1.0, color, BANNER_Y0, BANNER_H)
        text = ax.text(
            0.5,
            BANNER_Y0 + BANNER_H / 2,
            tzp,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_BODY,
            color=_banner_text_color(color),
            fontweight="bold",
            zorder=21,
            clip_on=False,
        )
        _check_banner_fit(ax, text, 0.0, 1.0)
        return
    half_w = (1.0 - SPLIT_GAP) / 2
    for x0, arm, cls in ((0.0, "TZP", tzp), (half_w + SPLIT_GAP, "SEMA", sema)):
        color = c["white"] if cls == "Null" else TRAJECTORY_COLORS[cls]
        _banner_patch(ax, x0, half_w, color, BANNER_Y0, BANNER_H)
        text = ax.text(
            x0 + half_w / 2,
            BANNER_Y0 + BANNER_H / 2,
            f"{arm}: {_class_short(cls)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_MIN,
            color=_banner_text_color(color),
            fontweight="bold",
            zorder=21,
            clip_on=False,
        )
        _check_banner_fit(ax, text, x0, half_w)


def _draw_subplot_header(
    ax, gene, tzp, sema, *, show_name: bool = True, marker: str | None = None
) -> None:
    """Class banner above a panel, with the gene name (and the assay ID if given).

    tzp=None (assay missing from the trajectory table) skips the banner only.
    """
    if tzp is not None:
        draw_class_banner(ax, tzp, sema)
    if marker is not None:
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
    if show_name:
        ax.text(
            0.5,
            HEADER_GENE_Y if marker is not None else 1.20,
            gene,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FS_EMPH,
            color=c["black"],
            fontweight="bold",
            clip_on=False,
        )


def draw_platform_banner(
    fig,
    axes: list,
    label: str,
    letter: str | None,
    *,
    x0: float,
    x1: float,
    gap: float = 0.022,
    letter_dx: float = 0.022,
    letter_dy: float = 0.004,
) -> None:
    """Gray platform banner above the gene names of a block's top row, with its panel letter.

    Args:
        fig: The figure.
        axes: The block's panel axes.
        label: Platform name printed in the banner.
        letter: Panel letter at the banner's top-left, or None for no letter.
        x0: Banner left edge (figure fraction).
        x1: Banner right edge (figure fraction).
        gap: Gene-name top -> banner bottom (figure fraction).
        letter_dx: Letter offset left of ``x0`` (figure fraction).
        letter_dy: Letter offset above the banner (figure fraction).
    """
    top_y = max(a.get_position().y1 for a in axes)
    top_axes = [a for a in axes if a.get_position().y1 == top_y]
    gene_baseline_y = max(
        a.get_position().y1 + (HEADER_GENE_Y - 1.0) * a.get_position().height for a in top_axes
    )
    gene_top_y = gene_baseline_y + FS_EMPH / 72.0 / fig.get_figheight()
    banner_height = BANNER_HEIGHT_IN / fig.get_figheight()
    banner_y = gene_top_y + gap
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (x0, banner_y),
            x1 - x0,
            banner_height,
            boxstyle="square,pad=0",
            facecolor="#E8E8E8",
            edgecolor="none",
            transform=fig.transFigure,
            clip_on=False,
        )
    )
    fig.text(
        (x0 + x1) / 2,
        banner_y + banner_height / 2,
        label,
        fontsize=FS_EMPH,
        fontweight="bold",
        ha="center",
        va="center",
    )
    if letter is not None:
        fig.text(
            x0 - letter_dx,
            banner_y + banner_height + letter_dy,
            letter,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def draw_ylabel(fig, x: float, y: float) -> None:
    """The rotated "% Change from baseline (95% CI)" label centered at ``(x, y)`` (figure fraction)."""
    fig.text(x, y, YLABEL, fontsize=FS_EMPH, ha="center", va="center", rotation=90)


def draw_bottom_strip(fig, x_left: float, x_right: float, y: float) -> None:
    """Bottom strip at height ``y``: the FDR key at ``x_left``, "Week" centered, the arm legend
    ending at ``x_right`` (figure fractions)."""
    fig.text(
        x_left,
        y,
        BETWEEN_TREATMENT_FDR_KEY,
        fontsize=FS_BODY,
        color=c["black"],
        ha="left",
        va="center",
        style="italic",
    )
    fig.text((x_left + x_right) / 2, y, "Week", fontsize=FS_EMPH, ha="center", va="center")
    handles = [
        mlines.Line2D(
            [0],
            [0],
            color=ARM_COLORS[arm],
            marker="o",
            markersize=3,
            linewidth=0.9,
            label=ARM_LABELS[arm],
        )
        for arm in tuple(ARMS)
    ]
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(x_right, y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=FS_EMPH,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
    )


# Stacked-platform layout of the supplementary trajectory figures: Olink block on top, SomaScan
# below, ``STACK_COLS`` panels across, every panel the same size.
STACK_COLS = 4
STACK_LEFT, STACK_RIGHT = 0.075, 0.985
STACK_WSPACE = 0.35
STACK_HSPACE = 0.50  # row gap as a fraction of panel height
STACK_TOP_IN = 0.70  # Olink banner + gene headers
STACK_GAP_IN = 1.20  # SomaScan banner + headers + Olink x-labels + extra clearance
STACK_BOTTOM_IN = 0.62  # x-axis labels + bottom legend strip


def stacked_panel_height_in(fig_w: float, cols: int = STACK_COLS) -> float:
    """Panel height (inches) that gives a ``cols``-across panel the ``TRAJ_PANEL_ASPECT`` ratio."""
    cell_w = (STACK_RIGHT - STACK_LEFT) * fig_w / (cols + (cols - 1) * STACK_WSPACE)
    return cell_w / TRAJ_PANEL_ASPECT


def stacked_platform_figure(
    olink: list[MarkerSpec], soma: list[MarkerSpec], *, stars_above: bool = True, fit_labels: bool = True
):
    """Olink panels above SomaScan panels, with platform banners, the y-label and the bottom strip.

    Rows derive from each platform's panel count; the block heights are budgeted in inches so
    every panel in both blocks has the same height. Unfilled slots of a block's last row are
    blank axes.

    Args:
        olink: Olink panels in drawing order (row-major).
        soma: SomaScan panels in drawing order.
        stars_above: See ``draw_lines``.
        fit_labels: See ``draw_lines``.

    Returns:
        The figure.
    """
    olink_rows = -(-len(olink) // STACK_COLS)
    soma_rows = -(-len(soma) // STACK_COLS)

    fig_w = nature_figsize("2col", 1.0)[0]
    PANEL_H_IN = stacked_panel_height_in(fig_w)

    def _block_in(rows: int) -> float:
        return PANEL_H_IN * (rows + (rows - 1) * STACK_HSPACE)

    olink_in, soma_in = _block_in(olink_rows), _block_in(soma_rows)
    height = STACK_TOP_IN + olink_in + STACK_GAP_IN + soma_in + STACK_BOTTOM_IN
    soma_bottom = STACK_BOTTOM_IN / height
    soma_top = (STACK_BOTTOM_IN + soma_in) / height
    olink_bottom = (STACK_BOTTOM_IN + soma_in + STACK_GAP_IN) / height
    olink_top = (STACK_BOTTOM_IN + soma_in + STACK_GAP_IN + olink_in) / height

    fig = plt.figure(figsize=nature_figsize("2col", height))
    gs_olink = fig.add_gridspec(
        olink_rows,
        STACK_COLS,
        left=STACK_LEFT,
        right=STACK_RIGHT,
        top=olink_top,
        bottom=olink_bottom,
        wspace=STACK_WSPACE,
        hspace=STACK_HSPACE,
    )
    gs_soma = fig.add_gridspec(
        soma_rows,
        STACK_COLS,
        left=STACK_LEFT,
        right=STACK_RIGHT,
        top=soma_top,
        bottom=soma_bottom,
        wspace=STACK_WSPACE,
        hspace=STACK_HSPACE,
    )

    def _populate(layout, gs, rows, platform):
        axes = []
        for slot in range(rows * STACK_COLS):
            r, col = divmod(slot, STACK_COLS)
            ax = fig.add_subplot(gs[r, col])
            if slot >= len(layout):
                ax.set_axis_off()
                continue
            draw_marker_panel(ax, platform, layout[slot], stars_above=stars_above, fit_labels=fit_labels)
            axes.append(ax)
        return axes

    olink_axes = _populate(olink, gs_olink, olink_rows, "olink")
    soma_axes = _populate(soma, gs_soma, soma_rows, "soma")

    # The banners span the full plot width even when a block has fewer filled columns.
    for axes, label, letter in ((olink_axes, "Olink", "a"), (soma_axes, "SomaScan", "b")):
        if axes:
            draw_platform_banner(fig, axes, label, letter, x0=STACK_LEFT, x1=STACK_RIGHT)

    draw_ylabel(fig, 0.018, (olink_top + soma_bottom) / 2)
    draw_bottom_strip(fig, STACK_LEFT, STACK_RIGHT, soma_bottom * 0.32)
    return fig
