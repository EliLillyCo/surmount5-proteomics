"""Supplementary figure 1: study design and quality-control flow.

CONSORT-style overview of the SURMOUNT-5 proteomics sub-study, styled as a clean
set of framed cards (slim rounded brand-color header bands over white bodies,
orthogonal connectors, subtle drop shadows, red ✗ / green ✓ status badges, and
TRIUMPH-style exclusion callouts that list each QC gate's per-reason counts).

The whole vertical layout is derived top-down from a single ``GAP`` so every
divider→box / box→box / box→divider step shares one rhythm.

Per-reason QC counts are derived from the committed sample-QC summaries
(``<qc_root>/surmount5_{olink,soma}/*.sample_qc_summary.tsv``) by the
technical-then-longitudinal staging that reproduces the net step totals:

    nonbridge    = ~FAIL_Bridge_Sample                         # 1824 / 1806
    technical    = nonbridge & any(FAIL_PCA_Outlier, _Median_IQR_Outlier,
                   _Sex_Concordance, _Age_Concordance, _Missing_Rate[OL] /
                   _RowCheck_and_RFU_Outlier[SM])              # −25 / −31
    longitudinal = (nonbridge & ~technical)
                   & any(FAIL_Discontinued, _No_Baseline, _Only_Baseline)
                                                               # −111 / −124
    final        = remainder                                   # 1688/598, 1651/591

Each excluded sample is attributed to a SINGLE reason — the first criterion it
fails in a fixed priority order (Olink technical: missing → PCA → median/IQR →
sex/age; SomaScan technical: RFU/row → PCA → median/IQR → sex/age; longitudinal:
discontinuation → no-baseline → only-baseline) — so per-reason counts sum
exactly to the net step total.

These counts are not hardcoded here: they are read from the committed artifact
``analysis/outputs/qc_exclusion_breakdown.parquet``, generated from the QC
summaries by ``analysis/scripts/qc_exclusion_breakdown.py`` (which holds the
staging/attribution logic). Regenerate that parquet — not this file — if the QC
inputs change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _internal._common import (
    ANALYSIS_DIR,
    ARM_COLORS,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_NARR,
    c,
    init_figure_theme,
    save_figure,
)

_HERE = Path(__file__).resolve().parent


# ─── Helpers ─────────────────────────────────────────────────────────


def _fade_hex(hex_color: str, frac: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"#{int(r + (255 - r) * frac):02X}{int(g + (255 - g) * frac):02X}{int(b + (255 - b) * frac):02X}"


# ─── Palette (Lilly brand) ───────────────────────────────────────────

TZP_EC = ARM_COLORS["TZP"]
SEMA_EC = ARM_COLORS["SEMA"]
OL_EC = c["bold_green"]
SM_EC = c["bold_brown"]
NEU_EC = c["bold_grey"]
TXT = c["black"]
ARR = c["bold_grey"]
EXCL = c["black"]
EXCL_FC = _fade_hex(c["vibrant_coral"], 0.80)
EXCL_EC = _fade_hex(c["red"], 0.25)
BULLET = "#5C4A48"

OL_ARR = _fade_hex(c["bold_green"], 0.5)
SM_ARR = _fade_hex(c["bold_brown"], 0.5)

ANALYSIS_FC = _fade_hex(c["neutral_cream"], 0.6)
ANALYSIS_EC = c["bold_grey"]

CARD_FC = "#FFFFFF"
CARD_EDGE = "#D9D9D9"
HEADER_TXT = "#FFFFFF"
GREEN_BADGE = c["bold_green"]
RED_BADGE = c["red"]
MUTED = "#8A8A8A"

# Font hierarchy ────────────────────────────────────────────────────
F1, F3, F4, F5 = FS_EMPH, FS_NARR, FS_BODY, FS_AUX
F6 = 5.0  # sub-detail (bullets, caveats) — Nature 5 pt floor
SEC_FS = FS_BODY
SEC_COL = "#666666"

CARD_PAD = 0.011  # rounded-corner radius for cards / callouts

# ── QC exclusion flow counts (read from the committed artifact) ──────────────
# All study-design / QC-flow counts come from
# analysis/outputs/qc_exclusion_breakdown.parquet (see module docstring); they
# are not hardcoded here. Each excluded sample is attributed to a SINGLE reason,
# so per-reason counts sum exactly to the net step total.
_QC_BREAKDOWN_PATH = ANALYSIS_DIR / "outputs" / "qc_exclusion_breakdown.parquet"


@dataclass(frozen=True)
class _PlatformQC:
    """Per-platform QC-flow counts unpacked from the breakdown artifact."""

    input_samples: int          # non-bridge samples entering technical QC
    technical: list[tuple[str, int]]    # (label, count) bullets, priority order
    longitudinal: list[tuple[str, int]]
    final_samples: int
    final_participants: int
    final_arm: str              # e.g. "301 TZP · 297 SEMA"

    @property
    def technical_n(self) -> int:
        return sum(n for _, n in self.technical)

    @property
    def longitudinal_n(self) -> int:
        return sum(n for _, n in self.longitudinal)


def _load_platform_qc(platform: str) -> _PlatformQC:
    """Build a :class:`_PlatformQC` for ``olink`` / ``soma`` from the artifact."""
    df = pl.read_parquet(_QC_BREAKDOWN_PATH).filter(pl.col("platform") == platform)

    def _bullets(stage: str) -> list[tuple[str, int]]:
        rows = df.filter(pl.col("stage") == stage).sort("order")
        return list(zip(rows["label"], rows["n_samples"]))

    inp = df.filter(pl.col("stage") == "input")
    final = df.filter((pl.col("stage") == "final") & pl.col("arm").is_null())
    arms = df.filter((pl.col("stage") == "final") & pl.col("arm").is_not_null())
    arm_n = dict(zip(arms["arm"], arms["n_participants"]))
    return _PlatformQC(
        input_samples=inp["n_samples"].item(),
        technical=_bullets("technical"),
        longitudinal=_bullets("longitudinal"),
        final_samples=final["n_samples"].item(),
        final_participants=final["n_participants"].item(),
        final_arm=f"{arm_n['TZP']} TZP · {arm_n['SEMA']} SEMA",
    )


# ─── Primitives ───────────────────────────────────────────────────────


def _t(ax, x, y, s, fs=F4, col=TXT, w="normal", ha="center", va="center", z=3):
    return ax.text(
        x, y, s, ha=ha, va=va, fontsize=fs, fontweight=w, color=col,
        transform=ax.transAxes, zorder=z, clip_on=False,
    )


def _arr(ax, xy0, xy1, col=ARR, lw=0.6, rad=0.0):
    # shrinkA/shrinkB=0 so the arrow touches both endpoints exactly (no gap
    # where it meets a connecting line or box edge).
    ax.add_patch(
        FancyArrowPatch(
            xy0, xy1, arrowstyle="->", connectionstyle=f"arc3,rad={rad}",
            color=col, lw=lw, transform=ax.transAxes, zorder=1,
            mutation_scale=6, shrinkA=0, shrinkB=0, clip_on=False,
        )
    )


def _line(ax, x0, y0, x1, y1, col="#CCCCCC", lw=0.5, z=0):
    ax.plot(
        [x0, x1], [y0, y1], color=col, lw=lw, transform=ax.transAxes,
        zorder=z, clip_on=False, solid_capstyle="round",
    )


def _box(ax, cx, cy, w, h, fc, ec, lw=0.5, pad=CARD_PAD):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad={pad}",
            fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2, clip_on=False,
        )
    )


def _card(
    ax, cx, cy, w, h, header_fc, title, body_lines, *,
    title_fs=F3, pad=CARD_PAD, header_frac=0.30, edge=CARD_EDGE, lw=0.6,
):
    """TRIUMPH-style card: drop shadow → white body → slim rounded header band.

    ``body_lines`` is a list of either plain strings (F5/TXT) or
    ``(text, fontsize, color)`` tuples.
    """
    # 1) drop shadow (soft, offset down-right)
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2 + 0.0025, cy - h / 2 - 0.004), w, h,
            boxstyle=f"round,pad={pad}", fc=(0, 0, 0, 0.09), ec="none",
            transform=ax.transAxes, zorder=1, clip_on=False,
        )
    )

    # 2) white card body with hairline border
    card = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad={pad}",
        fc=CARD_FC, ec=edge, lw=lw, transform=ax.transAxes, zorder=2, clip_on=False,
    )
    ax.add_patch(card)

    # 3) slim colored header band — a rectangle clipped to the rounded card so
    #    its TOP corners follow the card outline and its bottom edge is flat.
    #    NOTE: clip_on MUST be True or set_clip_path is ignored (square corners).
    header_h = h * header_frac
    band = Rectangle(
        (cx - w / 2 - pad, cy + h / 2 - header_h), w + 2 * pad, header_h + pad * 2,
        fc=header_fc, ec="none", transform=ax.transAxes, zorder=2.5, clip_on=True,
    )
    ax.add_patch(band)
    band.set_clip_path(card)

    # 4) title vertically centered in the VISIBLE band, which spans the divider
    #    (cy+h/2-header_h) up to the card's visual top (cy+h/2+pad)
    title_y = cy + h / 2 - (header_h - pad) / 2
    _t(ax, cx, title_y, title, fs=title_fs, col=HEADER_TXT, w="bold", z=3)

    # 5) body lines, vertically distributed in the body region
    body_top = cy + h / 2 - header_h
    body_bot = cy - h / 2
    n = len(body_lines)
    region = body_top - body_bot
    for i, line in enumerate(body_lines):
        ly = body_top - (i + 0.5) * (region / n)
        text, fs, col = line if isinstance(line, tuple) else (line, F5, TXT)
        _t(ax, cx, ly, text, fs=fs, col=col, z=3)
    return card


# Callout content metrics (visual = drawn box, incl. rounded-corner pad)
_C_TOPMARGIN = 0.013  # visual top → title
_C_TITLEGAP = 0.013   # title → first bullet
_C_BULLETSP = 0.0108  # bullet → bullet
_C_BOTMARGIN = 0.012  # last bullet → visual bottom


def _callout_h(k):
    """Visual height of an exclusion callout with ``k`` reason bullets."""
    return _C_TOPMARGIN + _C_TITLEGAP + (k - 1) * _C_BULLETSP + _C_BOTMARGIN


def _excl_callout(ax, cx, cy, w, step, total_n, reasons):
    """TRIUMPH-style exclusion callout, auto-sized tight to its content.

    Box height tracks the number of reasons so there is no empty strip above
    the title; ``cy`` is the visual centre.
    """
    vis_h = _callout_h(len(reasons))
    _box(ax, cx, cy, w, vis_h - 2 * CARD_PAD, EXCL_FC, EXCL_EC, lw=0.4, pad=CARD_PAD)
    tx = cx - w / 2 + 0.014
    vis_top = cy + vis_h / 2  # h_nom/2 + pad == vis_h/2
    title_y = vis_top - _C_TOPMARGIN
    _t(ax, tx, title_y, f"−{total_n} samples · {step}",
       fs=F5, col=EXCL, w="bold", ha="left")
    by = title_y - _C_TITLEGAP
    for label, n in reasons:
        _t(ax, tx, by, f"• {label}  ({n})", fs=F6, col=BULLET, ha="left", va="center")
        by -= _C_BULLETSP


def _badge(ax, x, y, kind, ms=7.0):
    """Small circular status badge — 'x' (red, exclusion) or 'check' (green)."""
    col = RED_BADGE if kind == "x" else GREEN_BADGE
    sym = "×" if kind == "x" else "✓"  # × / ✓ (both present in Inter)
    ax.plot(
        x, y, "o", color=col, ms=ms, markeredgecolor="white", markeredgewidth=0.6,
        zorder=5, transform=ax.transAxes, clip_on=False,
    )
    _t(ax, x, y + 0.0004, sym, fs=4.6, col="white", w="bold", z=6)


def _tier(ax, y, label):
    _line(ax, 0.0, y, 1.0, y, "#E0E0E0", 0.3)
    _t(ax, 0.01, y - 0.005, label, fs=SEC_FS, col=SEC_COL, w="bold", ha="left", va="top")


# ─── Figure ───────────────────────────────────────────────────────────


def make_figure():
    init_figure_theme()
    ol_qc = _load_platform_qc("olink")
    sm_qc = _load_platform_qc("soma")
    fig = plt.figure(figsize=(7.2, 8.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # One uniform vertical gap governs the whole figure: every divider→box,
    # box→box and box→divider step is GAP, so the flow reads with a single
    # consistent rhythm top to bottom. Positions are derived sequentially from
    # the top divider; nothing below is hand-placed.
    GAP = 0.021
    P = CARD_PAD

    def _vb(cy, h):  # visual bottom edge of a card (incl. rounded-corner pad)
        return cy - h / 2 - P

    def _vt(cy, h):  # visual top edge
        return cy + h / 2 + P

    # ═══ TIER 1 — Trial design ═══════════════════════════════════════════
    t1 = 0.985
    _tier(ax, t1, "Trial design")

    sur_h = 0.042
    sur_cy = t1 - GAP - (sur_h / 2 + P)
    _card(ax, 0.50, sur_cy, 0.30, sur_h, NEU_EC, "SURMOUNT-5",
          [("NCT05822830", F5, TXT), ("751 randomized", F5, TXT)],
          title_fs=F1, header_frac=0.36)

    arm_h = 0.042
    arm_cy = _vb(sur_cy, sur_h) - GAP - (arm_h / 2 + P)
    _card(ax, 0.25, arm_cy, 0.22, arm_h, TZP_EC, "Tirzepatide",
          [("15 mg (or maximum tolerated dose)", F5, TXT), ("n = 375", F5, TXT)])
    _card(ax, 0.75, arm_cy, 0.22, arm_h, SEMA_EC, "Semaglutide",
          [("2.4 mg (or maximum tolerated dose)", F5, TXT), ("n = 376", F5, TXT)])

    branch_y = (_vb(sur_cy, sur_h) + _vt(arm_cy, arm_h)) / 2
    _line(ax, 0.50, _vb(sur_cy, sur_h), 0.50, branch_y, ARR, 0.6)
    _line(ax, 0.25, branch_y, 0.75, branch_y, ARR, 0.6)
    _line(ax, 0.25, branch_y, 0.25, _vt(arm_cy, arm_h), TZP_EC, 0.6)
    _line(ax, 0.75, branch_y, 0.75, _vt(arm_cy, arm_h), SEMA_EC, 0.6)

    # ═══ TIER 2 — Sample collection ══════════════════════════════════════
    t2 = _vb(arm_cy, arm_h) - GAP
    _tier(ax, t2, "Sample collection")

    cohort_h = 0.042
    cohort_cy = (t2 - GAP) - (cohort_h / 2 + P)
    _card(ax, 0.50, cohort_cy, 0.34, cohort_h, NEU_EC, "Proteomics cohort (n = 641)",
          [("322 TZP  ·  319 SEMA", F5, TXT),
           ("110 of 751 not profiled", F6, "#999999")])

    merge_y = (_vb(arm_cy, arm_h) + _vt(cohort_cy, cohort_h)) / 2
    _line(ax, 0.25, _vb(arm_cy, arm_h), 0.25, merge_y, TZP_EC, 0.6)
    _line(ax, 0.75, _vb(arm_cy, arm_h), 0.75, merge_y, SEMA_EC, 0.6)
    _line(ax, 0.25, merge_y, 0.75, merge_y, ARR, 0.6)
    _arr(ax, (0.50, merge_y), (0.50, _vt(cohort_cy, cohort_h)), col=ARR)

    # Timeline (line + week labels above, sample splits below)
    tl_y = _vb(cohort_cy, cohort_h) - GAP - 0.020
    _arr(ax, (0.50, _vb(cohort_cy, cohort_h)), (0.50, tl_y + 0.018), col=ARR)
    nodes_x = [0.22, 0.50, 0.78]
    _line(ax, nodes_x[0], tl_y, nodes_x[-1], tl_y, "#CCCCCC", 1.0)
    visits = [
        ("Week 0", "634 participants", "319 TZP · 315 SEMA"),
        ("Week 24", "624 participants", "314 TZP · 310 SEMA"),
        ("Week 72", "564 participants", "280 TZP · 284 SEMA"),
    ]
    for nx, (vn, ns, nd) in zip(nodes_x, visits):
        ax.plot(nx, tl_y, "o", color="#666666", ms=5, zorder=3,
                transform=ax.transAxes, clip_on=False)
        _t(ax, nx, tl_y + 0.015, vn, fs=F4, w="bold")
        _t(ax, nx, tl_y - 0.013, ns, fs=F5)
        _t(ax, nx, tl_y - 0.027, nd, fs=F5, col="#999999")
    _t(ax, nodes_x[-1] + 0.065, tl_y, "3 visits", fs=F5, col=MUTED, ha="left")
    tl_bot = tl_y - 0.034  # below the sample-split sub-labels

    # ═══ TIER 3 — Dual-platform profiling ════════════════════════════════
    t3 = tl_bot - GAP
    _tier(ax, t3, "Dual-platform profiling")

    plat_h = 0.044
    plat_cy = (t3 - GAP) - (plat_h / 2 + P)
    plat_top = _vt(plat_cy, plat_h)

    split_y = (tl_bot + plat_top) / 2
    _line(ax, 0.50, tl_bot, 0.50, split_y, ARR, 0.5)
    _line(ax, 0.30, split_y, 0.70, split_y, ARR, 0.5)
    _line(ax, 0.30, split_y, 0.30, plat_top, ARR, 0.5)
    _line(ax, 0.70, split_y, 0.70, plat_top, ARR, 0.5)

    _card(ax, 0.30, plat_cy, 0.28, plat_h, OL_EC, "Olink Explore HT",
          [("Antibody-based · 5,416 assays", F5, TXT),
           (f"{ol_qc.input_samples:,} samples", F5, TXT)])
    _card(ax, 0.70, plat_cy, 0.28, plat_h, SM_EC, "SomaScan 11K",
          [("Aptamer-based · 10,771 aptamers", F5, TXT),
           (f"{sm_qc.input_samples:,} samples", F5, TXT)])

    # ═══ TIER 4 — Quality control ════════════════════════════════════════
    # Callouts are auto-sized and centred on evenly-gapped slots derived from
    # the platform's visual bottom; the same GAP carries through.
    QC_GAP = GAP
    plat_vis_bot = plat_cy - plat_h / 2 - CARD_PAD
    tier_qc = plat_vis_bot - QC_GAP
    _tier(ax, tier_qc, "Quality control")

    ol_x, sm_x = 0.30, 0.70
    ht, hl = _callout_h(4), _callout_h(3)
    tech_cy = (tier_qc - QC_GAP) - ht / 2
    long_cy = tech_cy - ht / 2 - QC_GAP - hl / 2
    final_h = 0.046
    final_vis_top = (long_cy - hl / 2) - QC_GAP
    final_cy = final_vis_top - final_h / 2 - CARD_PAD
    excl_w = 0.235

    def _qc_column(cx, side, tech_n, long_n, tech_reasons, long_reasons,
                   final_n_part, final_n_samp, final_arm, plat_in, plat_ec, plat_arr):
        if side == "left":
            excl_cx = 0.135
            excl_inner = excl_cx + excl_w / 2 + CARD_PAD  # visual right edge
            count_x, count_ha = cx + 0.020, "left"
        else:
            excl_cx = 0.865
            excl_inner = excl_cx - excl_w / 2 - CARD_PAD  # visual left edge
            count_x, count_ha = cx - 0.020, "right"

        # Continuous spine: platform → final analysis-ready card (one arrowhead)
        _arr(ax, (cx, plat_vis_bot), (cx, final_vis_top), col=plat_arr)

        # Exclusion gates tee off the spine to the side callouts
        for gy, step, n, reasons in [
            (tech_cy, "Technical QC", tech_n, tech_reasons),
            (long_cy, "Longitudinal QC", long_n, long_reasons),
        ]:
            _line(ax, cx, gy, excl_inner, gy, EXCL, 0.5)
            _badge(ax, excl_inner, gy, "x", ms=6.5)
            _excl_callout(ax, excl_cx, gy, excl_w, step, n, reasons)

        # Running retained count, beside the spine between the two gates
        ax.text(count_x, (tech_cy + long_cy) / 2,
                f"{plat_in - tech_n:,} samples\nafter technical QC",
                ha=count_ha, va="center", fontsize=F6, color=MUTED, linespacing=1.1,
                transform=ax.transAxes, zorder=4, clip_on=False)

        # Final analysis-ready card; ✓ centered in the header band (matches title)
        _card(ax, cx, final_cy, 0.26, final_h, plat_ec, f"{final_n_part} participants",
              [(f"{final_n_samp:,} samples", F5, TXT), (final_arm, F5, TXT)])
        _badge(ax, cx - 0.26 / 2 + 0.011,
               final_cy + final_h / 2 - (final_h * 0.30 - CARD_PAD) / 2, "check", ms=6.8)

    _qc_column(ol_x, "left", ol_qc.technical_n, ol_qc.longitudinal_n,
               ol_qc.technical, ol_qc.longitudinal, ol_qc.final_participants,
               ol_qc.final_samples, ol_qc.final_arm, ol_qc.input_samples, OL_EC, OL_ARR)
    _qc_column(sm_x, "right", sm_qc.technical_n, sm_qc.longitudinal_n,
               sm_qc.technical, sm_qc.longitudinal, sm_qc.final_participants,
               sm_qc.final_samples, sm_qc.final_arm, sm_qc.input_samples, SM_EC, SM_ARR)

    # ═══ TIER 5 — Analyses ═══════════════════════════════════════════════
    final_vis_bot = final_cy - final_h / 2 - CARD_PAD
    tier_an = final_vis_bot - QC_GAP
    _tier(ax, tier_an, "Analyses")

    dist_y = tier_an - 0.018
    _arr(ax, (ol_x, final_vis_bot), (ol_x, dist_y + 0.002), col=OL_ARR, lw=0.5)
    _arr(ax, (sm_x, final_vis_bot), (sm_x, dist_y + 0.002), col=SM_ARR, lw=0.5)

    analyses = [
        ("MMRM", "Figs 1–4"), ("Trajectories", "Fig 1"), ("Differential", "Fig 2"),
        ("Pathways", "Fig 2"), ("Mediation", "Fig 3"), ("Pharmacogenomics", "Fig 4"),
    ]
    n_a = len(analyses)
    bw = 0.11
    gap = (0.84 - n_a * bw) / (n_a - 1)
    x_start = 0.08
    first_cx = x_start + bw / 2
    last_cx = x_start + (n_a - 1) * (bw + gap) + bw / 2
    _line(ax, first_cx, dist_y, last_cx, dist_y, "#CCCCCC", 0.6)

    box_h = 0.035
    box_cy = dist_y - QC_GAP - box_h / 2 - 0.008  # comb drop + box half (+pad)
    for i, (name, fig_lbl) in enumerate(analyses):
        cx = x_start + bw / 2 + i * (bw + gap)
        _line(ax, cx, dist_y, cx, box_cy + box_h / 2 + 0.008 + 0.002, "#CCCCCC", 0.6)
        _box(ax, cx, box_cy, bw, box_h, ANALYSIS_FC, ANALYSIS_EC, lw=0.3, pad=0.008)
        _t(ax, cx, box_cy + 0.006, name, fs=F5, w="bold")
        _t(ax, cx, box_cy - 0.007, fig_lbl, fs=F6, col="#888888")

    # Per-reason QC counts are now single-attribution (sum to net), so no
    # overlap caveat is needed. Crop a uniform margin below the analysis boxes.
    crop_bottom = (box_cy - box_h / 2 - 0.008) - GAP

    return fig, crop_bottom


if __name__ == "__main__":
    from matplotlib.transforms import Bbox

    fig, crop_bottom = make_figure()
    crop = Bbox([[0, crop_bottom * 8.0], [7.2, 8.0]])
    save_figure(
        fig, str(_HERE / "supp_fig1_study_design"),
        formats=("pdf",), bbox_inches=crop,
    )
