"""Extended Data figure: SURMOUNT-5 study design and dual-platform quality-control flow.

CONSORT-style overview: framed
cards (slim rounded header bands over white bodies, orthogonal connectors, subtle drop
shadows, red ✗ / green ✓ badges) and exclusion callouts listing each QC gate's
per-reason counts. One trial card branches into two **platform** columns (Olink Explore
HT, SomaScan 11K) in two columns; each runs down through its own
technical, sample-identity and longitudinal gates, tees exclusions to side callouts, and
the two spines converge on one analysis-ready card with participants and visit-level
samples by platform and arm, each platform closed by a bold total row.

Every count is read from ``analysis/outputs/qc_exclusion_breakdown.parquet`` (written by
``analysis/qc_breakdown.py``) through ``analysis/qc_flow.py``; nothing is
hardcoded. Each excluded sample is charged to a single reason (the first gate it fails in
QC's own precedence), so a callout's removed counts sum to its total, and the analysis-ready
tier is the QC analysis cohort, the MMRM cohort. QC flows in samples; the cohort and final
tiers also show participants. Regenerate the parquet, not this file, if QC changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

from figures._internal._common import FS_AUX, FS_BODY, FS_EMPH, FS_MIN, FS_NARR, c, init_figure_theme, save_figure
from analysis.qc_flow import load_breakdown, load_platform_qc
from figures._internal._qc_breakdown import (
    ARM_DISPLAY,
    GRID_TOTAL,
    MINUS,
    PLATFORMS,
    grid_columns,
    grid_rows,
    identity_lines,
    randomized,
    removed_lines,
)
from figures._internal._theme import (
    CALLOUT_INK,
    ED_MAX_W_IN,
    GRAY_LIGHT,
    GRAY_MID,
    GRAY_MUTED,
    GRAY_RULE,
    GRAY_THRESHOLD,
    LW_DIAGRAM_ARROW,
    LW_DIAGRAM_ARROW_MAIN,
    LW_DIAGRAM_BOX,
    LW_DIAGRAM_BRANCH,
    LW_DIAGRAM_CALLOUT,
    LW_DIAGRAM_CARD,
    LW_DIAGRAM_LINE,
    LW_DIAGRAM_RULE,
    LW_DIAGRAM_SPINE,
    LW_DIAGRAM_SYMBOL_EDGE,
    PLATFORM_COLORS,
    PLATFORM_LABELS,
    PLATFORM_SHORT,
    fade_hex,
)

from analysis.design import ARM_FULL_NAMES as ARM_FULL

_HERE = Path(__file__).resolve().parent

# Trial-design facts not derivable from the QC data (registration; the arm names are
# analysis.design.ARM_FULL_NAMES).
TRIAL_NAME = "SURMOUNT-5"
TRIAL_REGISTRATION = "NCT05822830"

TRIAL_EC = c["bold_grey"]
FRAME_EC = c["black"]
TXT = c["black"]
ARR = c["bold_grey"]
EXCL = c["black"]
EXCL_FC = fade_hex(c["vibrant_coral"], 0.80)
EXCL_EC = fade_hex(c["red"], 0.25)
CARD_FC = c["white"]
CARD_EDGE = GRAY_LIGHT
HEADER_TXT = c["white"]
GREEN_BADGE = c["bold_green"]
RED_BADGE = c["red"]
MUTED = GRAY_MUTED

F1, F3, F4, F5, F6 = FS_EMPH, FS_NARR, FS_BODY, FS_AUX, FS_MIN
CARD_PAD = 0.011

FIG_W = ED_MAX_W_IN
FIG_H = 9.0
GAP = 0.019
STACK_GAP = 0.008


def _t(ax, x, y, s, fs=F4, col=TXT, w="normal", ha="center", va="center", z=3):
    return ax.text(
        x, y, s, ha=ha, va=va, fontsize=fs, fontweight=w, color=col,
        transform=ax.transAxes, zorder=z, clip_on=False,
    )


def _arr(ax, xy0, xy1, col=ARR, lw=LW_DIAGRAM_ARROW):
    ax.add_patch(
        FancyArrowPatch(
            xy0, xy1, arrowstyle="->", color=col, lw=lw, transform=ax.transAxes,
            zorder=1, mutation_scale=6, shrinkA=0, shrinkB=0, clip_on=False,
        )
    )


def _line(ax, x0, y0, x1, y1, col=GRAY_THRESHOLD, lw=LW_DIAGRAM_LINE, z=0):
    ax.plot(
        [x0, x1], [y0, y1], color=col, lw=lw, transform=ax.transAxes, zorder=z,
        clip_on=False, solid_capstyle="round",
    )


def _box(ax, cx, cy, w, h, fc, ec, lw=LW_DIAGRAM_BOX, pad=CARD_PAD):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad={pad}", fc=fc, ec=ec,
            lw=lw, transform=ax.transAxes, zorder=2, clip_on=False,
        )
    )


def _card(ax, cx, cy, w, h, header_fc, title, body_lines, *, title_fs=F3, header_frac=0.30):
    """Card: drop shadow, white body, slim rounded header band clipped to the card."""
    pad = CARD_PAD
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2 + 0.0025, cy - h / 2 - 0.004), w, h, boxstyle=f"round,pad={pad}",
            fc=(0, 0, 0, 0.09), ec="none", transform=ax.transAxes, zorder=1, clip_on=False,
        )
    )
    card = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad={pad}", fc=CARD_FC,
        ec=CARD_EDGE, lw=LW_DIAGRAM_CARD, transform=ax.transAxes, zorder=2, clip_on=False,
    )
    ax.add_patch(card)
    header_h = h * header_frac
    band = Rectangle(
        (cx - w / 2 - pad, cy + h / 2 - header_h), w + 2 * pad, header_h + pad * 2,
        fc=header_fc, ec="none", transform=ax.transAxes, zorder=2.5, clip_on=True,
    )
    ax.add_patch(band)
    band.set_clip_path(card)
    _t(ax, cx, cy + h / 2 - (header_h - pad) / 2, title, fs=title_fs, col=HEADER_TXT, w="bold")
    body_top, body_bot = cy + h / 2 - header_h, cy - h / 2
    n = len(body_lines)
    for i, (text, fs, col) in enumerate(body_lines):
        _t(ax, cx, body_top - (i + 0.5) * ((body_top - body_bot) / n) - pad / 2, text, fs=fs, col=col)


_C_TOPMARGIN = 0.013
_C_TITLEGAP = 0.013
_C_BULLETSP = 0.0108
_C_BOTMARGIN = 0.012


def _callout_h(k: int) -> float:
    """Visual height of an exclusion callout with ``k`` reason lines."""
    return _C_TOPMARGIN + _C_TITLEGAP + (k - 1) * _C_BULLETSP + _C_BOTMARGIN


def _excl_callout(ax, cx, cy, w, step, total_n, lines):
    """Exclusion callout, sized to its content; ``cy`` is its visual center."""
    vis_h = _callout_h(len(lines))
    _box(ax, cx, cy, w, vis_h - 2 * CARD_PAD, EXCL_FC, EXCL_EC, lw=LW_DIAGRAM_CALLOUT)
    tx = cx - w / 2 + 0.014
    title_y = cy + vis_h / 2 - _C_TOPMARGIN
    _t(ax, tx, title_y, f"{MINUS}{total_n:,} samples · {step}", fs=F5, col=EXCL, w="bold", ha="left")
    by = title_y - _C_TITLEGAP
    for label, count in lines:
        _t(ax, tx, by, f"• {label}  ({count})", fs=F6, col=CALLOUT_INK, ha="left")
        by -= _C_BULLETSP


def _badge(ax, x, y, kind, ms=7.0):
    col = RED_BADGE if kind == "x" else GREEN_BADGE
    ax.plot(
        x, y, "o", color=col, ms=ms, markeredgecolor="white",
        markeredgewidth=LW_DIAGRAM_SYMBOL_EDGE, zorder=5, transform=ax.transAxes, clip_on=False,
    )
    _t(ax, x, y + 0.0004, "×" if kind == "x" else "✓", fs=FS_MIN, col="white", w="bold", z=6)


def _tier(ax, y, label):
    _line(ax, 0.0, y, 1.0, y, GRAY_RULE, LW_DIAGRAM_RULE)
    _t(ax, 0.01, y - 0.005, label, fs=FS_BODY, col=GRAY_MID, w="bold", ha="left", va="top")


GRID_PAD_X = 0.030
GRID_ARM_X = 0.085
GRID_FIRST_VALUE_X = 0.205
GRID_ROW = 0.0125
GRID_TOP_GAP = 0.007
GRID_TOTAL_GAP = 0.004
FRAME_HEADER_H = 0.0165


def _grid_h(rows) -> float:
    n_gaps = sum(r.total for r in rows[:-1])
    return GRID_TOP_GAP + (len(rows) + 1) * GRID_ROW + n_gaps * GRID_TOTAL_GAP


def _final_grid(ax, left, right, top, columns, rows):
    """Column headers over a rule, then right-aligned counts per platform and arm."""
    first = left + GRID_FIRST_VALUE_X
    pitch = (right - first) / (len(columns) - 1)
    xs = [first + k * pitch for k in range(len(columns))]
    y = top - GRID_TOP_GAP - GRID_ROW / 2
    for x, name in zip(xs, columns, strict=True):
        _t(ax, x, y, name, fs=F5, col=MUTED, ha="right")
    _line(ax, left, y - GRID_ROW / 2, right, y - GRID_ROW / 2, GRAY_LIGHT, LW_DIAGRAM_RULE, z=3)
    previous = None
    for row in rows:
        y -= GRID_ROW
        if row.total:
            rule_y = y + GRID_ROW / 2
            _line(ax, left + GRID_ARM_X, rule_y, right, rule_y, GRAY_LIGHT, LW_DIAGRAM_RULE, z=3)
        weight = "bold" if row.total else "normal"
        if row.platform != previous:
            _t(ax, left, y, PLATFORM_SHORT[row.platform], fs=F5, ha="left")
            previous = row.platform
        _t(ax, left + GRID_ARM_X, y, GRID_TOTAL if row.total else ARM_DISPLAY[row.arm], fs=F5, w=weight, ha="left")
        for x, n in zip(xs, row.counts, strict=True):
            _t(ax, x, y, f"{n:,}", fs=F5, w=weight, ha="right")
        if row.total:
            y -= GRID_TOTAL_GAP


def make_figure():
    """Assemble the figure; returns it and the lower edge of the used area (figure fraction)."""
    init_figure_theme()
    df = load_breakdown()
    qcs = {platform: load_platform_qc(platform, df) for platform in PLATFORMS}
    n_rand, rand_arm = randomized(df)
    columns = grid_columns(df)
    weeks = " / ".join(label.removeprefix("Week ") for label in columns[1:])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    P = CARD_PAD

    def _vb(cy, h):
        return cy - h / 2 - P

    def _vt(cy, h):
        return cy + h / 2 + P

    X = {"olink": 0.28, "soma": 0.72}
    CX = 0.50
    SIDE = {"olink": "left", "soma": "right"}
    arrow_col = {p: fade_hex(PLATFORM_COLORS[p], 0.5) for p in PLATFORMS}

    # Tier 1: trial design
    t1 = 0.985
    _tier(ax, t1, "Trial design")
    trial_h = 0.044
    trial_cy = t1 - GAP - (trial_h / 2 + P)
    arms_line = " · ".join(f"{ARM_FULL[arm]}: {rand_arm[arm]:,}" for arm in ARM_DISPLAY)
    _card(
        ax, CX, trial_cy, 0.46, trial_h, TRIAL_EC, f"{TRIAL_NAME} · {TRIAL_REGISTRATION}",
        [(f"{n_rand:,} randomized", F5, TXT), (arms_line, F6, MUTED)],
        title_fs=F1, header_frac=0.36,
    )

    # Tier 2: one proteomics card per platform, branching from the trial card
    t2 = _vb(trial_cy, trial_h) - STACK_GAP
    _tier(ax, t2, "Proteomics sub-study")
    coh_h = 0.044
    coh_cy = (t2 - GAP) - (coh_h / 2 + P)
    branch_y = t2 - GAP / 2
    _line(ax, CX, _vb(trial_cy, trial_h), CX, branch_y, ARR, LW_DIAGRAM_BRANCH)
    _line(ax, X["olink"], branch_y, X["soma"], branch_y, ARR, LW_DIAGRAM_BRANCH)
    for platform, qc in qcs.items():
        _arr(ax, (X[platform], branch_y), (X[platform], _vt(coh_cy, coh_h)), col=arrow_col[platform])
        arm_text = " · ".join(f"{qc.cohort_arm[arm]} {label}" for arm, label in ARM_DISPLAY.items())
        _card(
            ax, X[platform], coh_cy, 0.38, coh_h, PLATFORM_COLORS[platform],
            f"{PLATFORM_LABELS[platform]} · {qc.cohort_subjects} participants",
            [(arm_text, F5, TXT), (f"{qc.input_samples:,} samples · Weeks {weeks}", F6, MUTED)],
        )

    # Tier 3: quality control, one spine per platform with callouts to the outside
    coh_vis_bot = _vb(coh_cy, coh_h)
    t3 = coh_vis_bot - STACK_GAP
    _tier(ax, t3, "Quality control  (samples)")
    tiers = [
        ("Technical QC", "technical", lambda qc: removed_lines(qc.technical)),
        ("Sample identity", "identity", identity_lines),
        ("Longitudinal QC", "longitudinal", lambda qc: removed_lines(qc.longitudinal)),
    ]
    heights = [_callout_h(max(len(lines(qc)) for qc in qcs.values())) for *_, lines in tiers]
    tier_cy = []
    top = t3 - GAP
    for h in heights:
        tier_cy.append(top - h / 2)
        top -= h + STACK_GAP
    spine_bot = top
    excl_w = 0.235

    for platform, qc in qcs.items():
        cx = X[platform]
        if SIDE[platform] == "left":
            excl_cx, excl_inner = 0.135, 0.135 + excl_w / 2 + P
            count_x, count_ha = cx + 0.020, "left"
        else:
            excl_cx, excl_inner = 0.865, 0.865 - excl_w / 2 - P
            count_x, count_ha = cx - 0.020, "right"
        _line(ax, cx, coh_vis_bot, cx, spine_bot, arrow_col[platform], LW_DIAGRAM_SPINE)
        remaining = qc.input_samples
        notes = []
        for gy, (step, attr, lines) in zip(tier_cy, tiers, strict=True):
            total = getattr(qc, f"{attr}_n")
            _line(ax, cx, gy, excl_inner, gy, EXCL, LW_DIAGRAM_LINE)
            _badge(ax, excl_inner, gy, "x", ms=6.5)
            _excl_callout(ax, excl_cx, gy, excl_w, step, total, lines(qc))
            remaining -= total
            notes.append(remaining)
        identity_note = [f"{notes[1]:,} samples", "after identity checks"]
        if qc.reassigned > 0:
            identity_note.append(f"{qc.reassigned:,} reassigned")
        for y, text in (
            ((tier_cy[0] + tier_cy[1]) / 2, f"{notes[0]:,} samples\nafter technical QC"),
            ((tier_cy[1] + tier_cy[2]) / 2, "\n".join(identity_note)),
        ):
            ax.text(
                count_x, y, text, ha=count_ha, va="center", fontsize=F6, color=MUTED,
                linespacing=1.1, transform=ax.transAxes, zorder=4, clip_on=False,
            )

    # Tier 4: the two spines converge on the analysis-ready card
    t4 = spine_bot - STACK_GAP
    _tier(ax, t4, "Analysis  (participants / samples)")
    rows = grid_rows(df)
    frame_w = 0.56
    frame_h = FRAME_HEADER_H + _grid_h(rows)
    frame_cy = (t4 - GAP) - (frame_h / 2 + P)
    merge_y = t4 - GAP / 2
    for platform in PLATFORMS:
        _line(ax, X[platform], spine_bot, X[platform], merge_y, arrow_col[platform], LW_DIAGRAM_SPINE)
    _line(ax, X["olink"], merge_y, X["soma"], merge_y, ARR, LW_DIAGRAM_BRANCH)
    _arr(ax, (CX, merge_y), (CX, _vt(frame_cy, frame_h)), col=ARR, lw=LW_DIAGRAM_ARROW_MAIN)
    _card(
        ax, CX, frame_cy, frame_w, frame_h, FRAME_EC, "Analysis-ready cohort", [],
        title_fs=F1, header_frac=FRAME_HEADER_H / frame_h,
    )
    _badge(ax, CX - frame_w / 2 + 0.014, frame_cy + frame_h / 2 - (FRAME_HEADER_H - P) / 2, "check", ms=6.8)
    _final_grid(
        ax,
        CX - frame_w / 2 + GRID_PAD_X,
        CX + frame_w / 2 - GRID_PAD_X,
        frame_cy + frame_h / 2 - FRAME_HEADER_H,
        columns,
        rows,
    )
    return fig, frame_cy - frame_h / 2 - P - STACK_GAP


def build():
    """The figure and the ``savefig`` kwargs that crop the unused space below the flow."""
    fig, crop_bottom = make_figure()
    return fig, {"bbox_inches": Bbox([[0, crop_bottom * FIG_H], [FIG_W, FIG_H]])}


if __name__ == "__main__":
    fig, kwargs = build()
    save_figure(fig, str(_HERE / "supp_fig_study_design"), formats=("pdf",), **kwargs)
