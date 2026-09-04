"""Pancreatic protein trajectories with external validation + pharmacogenomics.

Two figures produced from one parameterized renderer:
  - PPY  → main figure          (fig5_ppy.pdf)
  - REG4 → supplementary figure (supp_fig15_reg4.pdf)

Layout: two rows. The top row sets up the GIP→PPY mechanism and the
SURMOUNT-5 / STEP responses; the bottom row is the pharmacogenomic test.
  Panel a (top-left): predicted GIP-receiving cell types (replotted Hormone
                      Cell Atlas, Fei et al. 2026) — the pancreatic γ/PP cell
                      (the PPY source) tops the ranking, motivating the PPY
                      response to GIP-containing TZP.
  Panel b (top-middle): SURMOUNT-5 %CFB trajectories — Olink + SomaScan,
                      TZP vs SEMA across Wk0/24/72.
  Panel c (top-right): STEP 1 + STEP 2 SomaScan forest — SEMA vs PBO at the
                       trial's single post-baseline timepoint, primary
                       (top, filled) vs weight + HbA1c-adjusted (bottom, open).
                       Same mediation idiom as ``fig4_mediation.py``.
  Panel d (bottom): GIPR rs1800437 (E354Q) × treatment-effect boxplots +
                    swarm of residualized change-from-baseline across
                    Olink × SomaScan × {Wk24, Wk72}, 1×4 grid grouped by
                    platform. MMRM TZP-SEMA contrasts per genotype × visit
                    are overlaid as the per-genotype β with significance
                    stars. Each (gene × platform) is fit as its own MMRM
                    so Olink and SomaScan contrasts reflect each platform
                    independently.

External STEP data: Maretty et al. Nat Med 31, 1–11 (2025),
    Supplementary Table 2 — prepared by ``analysis/scripts/maretty_step_data.py``.
Genotype + MMRM data: produced by ``analysis/pharmacogenomics/run.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
FIGURES_DIR = THIS_DIR.parent

from ._common import (
    ARM_COLORS,
    ARM_LABELS,
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FS_AUX,
    FS_BODY,
    FS_EMPH,
    FS_NARR,
    FS_PANEL,
    NAT_W2,
    c,
    TRAJECTORY_COLORS,
    TRAJ_PANEL_ASPECT,
    init_figure_theme,
    save_figure,
)
from ._trajectory_lines import _best_marker_for_gene, _draw_panel
from ._gip_panel import (
    ATLAS_DATA,
    draw_gip_panel,
    load_hormone_strengths,
)
from ._maretty_step import load_step_effects as _load_step_effects

PLATFORM_LABELS = {"olink": "Olink", "soma": "SomaScan"}

CONNECTOR_GREY = "#666666"

# Which adjusted-p column from Maretty's supp tables drives the significance
# threshold for the forest plot. "pvalue_adj" is the paper's Bonferroni column
# (very conservative — for PPY/REG4, all cells fail at 0.05). "qvalue" is the
# paper's Storey q-value (more powerful — PPY/REG4 STEP 1 primary survives).
# The legend below the banner reads back this column name verbatim so the
# significance criterion is self-documenting.
SIG_COLUMN = "pvalue_adj"
SIG_THRESHOLD = 0.05

# Banner sizing kept absolute (inches) — ~2× the 7pt bold font height — so it
# stays the same physical size regardless of figure height (mediation forest
# convention). Convert to figure-fraction at module load.
#
# Two-row landscape layout:
#   Row 1 (top): a GIP-receiving cells · b PPY trajectory (Olink+SomaScan) ·
#                c STEP forest — all sharing one plot-top / plot-bottom.
#   Row 2 (bottom): d GIPR-E354Q genotype boxplots (1×4 grouped by platform).
# Vertical extents are budgeted in inches (top → bottom) so the figure is
# exactly as tall as its content; the trajectory aspect (TRAJ_PANEL_ASPECT)
# sets the row-1 plot height and every row-1 panel shares it, so their
# tops/bottoms align.
FIG_W = NAT_W2
BANNER_HEIGHT_IN = 0.20  # canonical banner height
_R1_BANNER_GAP_IN = 0.20  # row-1 plot-top → banner (clears traj class-chip)
_R2_BANNER_GAP_IN = 0.16  # row-2 plot-top → banner
_R1_PLOT_H_IN = 1.34  # trajectory aspect drives this (w = h · 1.21)
_TRAJ_W_IN = _R1_PLOT_H_IN * TRAJ_PANEL_ASPECT  # ≈1.62 in per trajectory panel
_R2_PLOT_H_IN = 1.30
_TOP_MARGIN_IN = 0.12
_BOT_MARGIN_IN = 0.12
_R1_BELOW_IN = 0.52  # row-1 x-ticks + axis-label + legend stripe
_INTER_ROW_IN = 0.22  # row-1 stripe bottom → row-2 banner top (breathing)
_R2_BELOW_IN = 0.50  # row-2 x-ticks + n-labels + axis label

FIG_H = (
    _TOP_MARGIN_IN
    + BANNER_HEIGHT_IN
    + _R1_BANNER_GAP_IN
    + _R1_PLOT_H_IN
    + _R1_BELOW_IN
    + _INTER_ROW_IN
    + BANNER_HEIGHT_IN
    + _R2_BANNER_GAP_IN
    + _R2_PLOT_H_IN
    + _R2_BELOW_IN
    + _BOT_MARGIN_IN
)

BANNER_HEIGHT = BANNER_HEIGHT_IN / FIG_H
BANNER_GAP = _R1_BANNER_GAP_IN / FIG_H  # default gap (row 1)
PANEL_C_BANNER_GAP = _R2_BANNER_GAP_IN / FIG_H


def _yt(dist_from_top_in: float) -> float:
    """Figure-fraction y (from bottom) of a point ``dist_from_top_in`` below top."""
    return 1.0 - dist_from_top_in / FIG_H


def _xl(inch: float) -> float:
    return inch / FIG_W


# --- Row 1 vertical (shared by GIP, trajectory, forest) ---
_R1_TOP_DIST = _TOP_MARGIN_IN + BANNER_HEIGHT_IN + _R1_BANNER_GAP_IN
PLOT_TOP = _yt(_R1_TOP_DIST)
PLOT_BOT = _yt(_R1_TOP_DIST + _R1_PLOT_H_IN)

# --- Row 1 horizontal anchors (inches from fig-left) ---
_LEFT_LABEL_IN = 0.05  # fig-left → GIP cell-type labels (thin margin)
_GIP_GUTTER_IN = 0.62  # just fits the (truncated) 2-line cell labels
_GIP_BARS_IN = 0.88  # GIP bar-axis width
_GIP_TRAJ_GAP_IN = 0.53  # holds the trajectory y-axis label + ticks (breathing)
_TRAJ_WSPACE_IN = 0.24
_TRAJ_FOREST_GAP_IN = 0.53  # holds the forest STEP y-labels (breathing)
_FOREST_W_IN = 1.09  # widened: matches "Data from Maretty et al., 2025" text width

_gip_axis_left = _LEFT_LABEL_IN + _GIP_GUTTER_IN
GIP_LEFT = _xl(_gip_axis_left)
GIP_RIGHT = _xl(_gip_axis_left + _GIP_BARS_IN)
# Panel a has no banner strip, so its plot rises into the (otherwise empty)
# banner-gap band — taller than b/c (which start at PLOT_TOP) — giving the 8
# bars more room. A small gap is left below its title (which stays aligned with
# the b/c banner band). Bottoms still align at PLOT_BOT, so the x-axes are level.
_GIP_TITLE_GAP_IN = 0.05
GIP_TOP = PLOT_TOP + (_R1_BANNER_GAP_IN - _GIP_TITLE_GAP_IN) / FIG_H
GIP_BOT = PLOT_BOT
GIP_TOP_N = 8
GIP_TITLE_X = _xl(0.20)  # left edge of the panel-a title (right of the 'a' letter)

# Panel b (trajectory): 1×2 Olink + SomaScan, aspect locked via _TRAJ_W_IN.
_traj_left = _gip_axis_left + _GIP_BARS_IN + _GIP_TRAJ_GAP_IN
_soma_left = _traj_left + _TRAJ_W_IN + _TRAJ_WSPACE_IN
PANEL_A_LEFT = _xl(_traj_left)
PANEL_A_RIGHT = _xl(_soma_left + _TRAJ_W_IN)
PANEL_A_WSPACE = _TRAJ_WSPACE_IN / _TRAJ_W_IN

# Panel c (forest): single narrow axis; its STEP y-labels live in the gap.
_forest_left = _soma_left + _TRAJ_W_IN + _TRAJ_FOREST_GAP_IN
PANEL_B_LEFT = _xl(_forest_left)
PANEL_B_RIGHT = _xl(_forest_left + _FOREST_W_IN)

# Two consistent y-coords for the row-1 axis labels and legends. The legend
# row sits just below the axis-label row; all three legends share LEGEND_Y so
# they stay mutually aligned.
AXIS_LABEL_Y = _yt(_R1_TOP_DIST + _R1_PLOT_H_IN + 0.30)
LEGEND_Y = _yt(_R1_TOP_DIST + _R1_PLOT_H_IN + 0.385)
BOTTOM_LABEL_FONTSIZE = 6.5

# --- Row 2: panel d (genotype boxplots, 1×4 grouped by platform) ---
_R2_TOP_DIST = (
    _R1_TOP_DIST
    + _R1_PLOT_H_IN
    + _R1_BELOW_IN
    + _INTER_ROW_IN
    + BANNER_HEIGHT_IN
    + _R2_BANNER_GAP_IN
)
PLOT_C_TOP = _yt(_R2_TOP_DIST)
PLOT_C_BOT = _yt(_R2_TOP_DIST + _R2_PLOT_H_IN)
PANEL_C_LEFT, PANEL_C_RIGHT = 0.080, 0.990
# Tighter wspace within each platform pair (cols 1↔2, 3↔4); a wider
# PLATFORM_GAP separates the Olink pair from the SomaScan pair.
PANEL_C_WSPACE_TIGHT = 0.30
PANEL_C_PLATFORM_GAP = 0.07  # fig-fraction inserted between cols 2 and 3

# Where the bottom-row axis label sits (below panel d plot).
PANEL_C_AXIS_LABEL_Y = _yt(_R2_TOP_DIST + _R2_PLOT_H_IN + 0.38)


# ---------------------------------------------------------------------------
# Panel a: trajectory
# ---------------------------------------------------------------------------
def _render_trajectory_panel(
    ax,
    platform: str,
    gene: str,
    show_x: bool = True,
    show_ylabel: bool = False,
) -> None:
    """Line plot + in-panel header (trajectory-class chip + marker ID).

    TZP-vs-SEMA across-treatment FDR stars are shown above/below the peak/
    trough arm following the trajectory direction (mirrors the supp PCBL
    figures' convention).
    """
    trajectory = _draw_panel(ax, platform, gene, show_x=show_x, show_fdr_stars=True)
    marker = _best_marker_for_gene(platform, gene)

    if show_ylabel:
        ax.set_ylabel("% Change from baseline (95% CI)", fontsize=FS_EMPH, labelpad=2)

    if trajectory is None:
        return

    chip_color = TRAJECTORY_COLORS.get(trajectory, "#888888")
    r_, g_, b_ = mpl.colors.to_rgb(chip_color)
    lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
    text_color = "white" if lum < 0.45 else c["black"]

    ax.add_patch(
        mpatches.Rectangle(
            (0, 1.02),
            1.0,
            0.09,
            facecolor=chip_color,
            edgecolor=c["black"],
            linewidth=0.4,
            transform=ax.transAxes,
            zorder=20,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        1.065,
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
    if marker is not None:
        ax.text(
            0.02,
            0.97,
            marker,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_AUX,
            color=c["bold_grey"],
            zorder=22,
            clip_on=False,
        )


# ---------------------------------------------------------------------------
# Panel b: STEP forest
# ---------------------------------------------------------------------------
STEP_TRIALS = [
    # Listed in display order top→bottom. matplotlib y=0 is bottom, so the
    # later entry renders at the top — reverse the natural order so STEP 1
    # sits above STEP 2 in the panel.
    ("step2", "STEP 2", "obesity\n+ T2D"),
    ("step1", "STEP 1", "obesity"),
]

def _draw_step_forest(ax, gene: str) -> str | None:
    """Forest plot: STEP 1 + STEP 2, primary (filled, y+0.18) vs weight/HbA1c-
    adjusted (open, y-0.18) per trial. Returns the SeqId used (for the
    top-left corner annotation).
    """
    effects = _load_step_effects(gene)
    Y_PRIMARY, Y_BLOCKED = 0.18, -0.18
    seqid = None

    ax.axvline(0, color=c["bold_grey"], linewidth=0.5, alpha=0.7, zorder=1)

    for i, (trial_key, _, _) in enumerate(STEP_TRIALS):
        y = i
        primary = effects.get((trial_key, "primary"))
        blocked = effects.get((trial_key, "blocked"))
        if primary is None:
            continue
        if seqid is None:
            seqid = primary["SeqId"]

        primary_sig = (
            primary[SIG_COLUMN] is not None and primary[SIG_COLUMN] < SIG_THRESHOLD
        )
        direction_color = COLOR_UP if primary["effect_size"] > 0 else COLOR_DOWN
        primary_color = direction_color if primary_sig else COLOR_NS

        # Primary effect bar (filled circle, y+0.18)
        p_lo = primary["effect_size"] - 1.96 * primary["std_error"]
        p_hi = primary["effect_size"] + 1.96 * primary["std_error"]
        ax.plot(
            [p_lo, p_hi],
            [y + Y_PRIMARY, y + Y_PRIMARY],
            color=primary_color,
            linewidth=0.9,
            zorder=2,
        )
        ax.scatter(
            [primary["effect_size"]],
            [y + Y_PRIMARY],
            s=18,
            c=primary_color,
            edgecolors=c["black"],
            linewidths=0.3,
            zorder=3,
        )

        # Blocked / adjusted effect bar (open circle, y-0.18)
        if blocked is not None:
            blocked_sig = (
                blocked[SIG_COLUMN] is not None and blocked[SIG_COLUMN] < SIG_THRESHOLD
            )
            blocked_color = (
                direction_color if (primary_sig and blocked_sig) else COLOR_NS
            )
            b_lo = blocked["effect_size"] - 1.96 * blocked["std_error"]
            b_hi = blocked["effect_size"] + 1.96 * blocked["std_error"]
            ax.plot(
                [b_lo, b_hi],
                [y + Y_BLOCKED, y + Y_BLOCKED],
                color=blocked_color,
                linewidth=0.9,
                zorder=2,
            )
            ax.scatter(
                [blocked["effect_size"]],
                [y + Y_BLOCKED],
                s=18,
                c="white",
                edgecolors=blocked_color,
                linewidths=0.9,
                zorder=3,
            )

            if primary_sig:
                ax.plot(
                    [primary["effect_size"], blocked["effect_size"]],
                    [y + Y_PRIMARY, y + Y_BLOCKED],
                    color=CONNECTOR_GREY,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=1,
                )

    # Two-line y-tick labels: trial name (bold) + short context (grey).
    n = len(STEP_TRIALS)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([""] * n)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 1 + 0.6)

    from matplotlib import transforms

    base = ax.get_yaxis_transform()
    # Trial name anchored just ABOVE the row centre (va="bottom") and the grey
    # context just BELOW it (va="top"), so a multi-line context (e.g. STEP 2's
    # "obesity / + T2D") grows downward without ever colliding with the name.
    LINE_OFFSET_PT = 1.5
    trans_up = transforms.offset_copy(
        base, fig=ax.figure, x=0, y=LINE_OFFSET_PT, units="points"
    )
    trans_dn = transforms.offset_copy(
        base, fig=ax.figure, x=0, y=-LINE_OFFSET_PT, units="points"
    )
    for i, (_, trial_label, ctx_label) in enumerate(STEP_TRIALS):
        ax.text(
            -0.04,
            i,
            trial_label,
            transform=trans_up,
            ha="right",
            va="bottom",
            fontsize=FS_NARR,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            -0.04,
            i,
            ctx_label,
            transform=trans_dn,
            ha="right",
            va="top",
            fontsize=FS_AUX,
            color=c["bold_grey"],
            clip_on=False,
            linespacing=1.3,
        )

    # X-axis label is rendered at the figure level via fig.text so its y
    # aligns with panel a's "Week" label across panels — see render().
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.4, pad=1)
    # Y-ticks are at row centres (y=0, y=1), which sit vertically midway
    # between each row's trial-name (offset up) and subtitle (offset down)
    # text lines. Suppress matplotlib's auto-generated minor ticks so only
    # the two major ticks at the row centres remain visible.
    ax.tick_params(axis="y", length=2, width=0.4, pad=1)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    # Full box: keep all four spines visible (top + right not hidden) so the
    # forest reads as a complete framed panel.
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # SeqId in top-left corner (matches trajectory plots' marker-ID convention).
    if seqid is not None:
        ax.text(
            0.02,
            0.97,
            seqid,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_AUX,
            color=c["bold_grey"],
            zorder=22,
            clip_on=False,
        )

    return seqid


# ---------------------------------------------------------------------------
# Panel c: GIPR rs1800437 × genotype boxplots + swarm of residualized Δ
# ---------------------------------------------------------------------------
GENO_LABELS = {0: "GG", 1: "GC", 2: "CC"}
PANEL_C_CELLS = [
    ("olink", "Week24", "Wk24"),
    ("olink", "Week72", "Wk72"),
    ("soma", "Week24", "Wk24"),
    ("soma", "Week72", "Wk72"),
]
RESID_COVS = ["age", "sex", "PC1", "PC2", "PC3", "base"]


def _load_e354q_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the focused frame + per-genotype MMRM contrasts produced by
    analysis/pharmacogenomics/run.py. Raises a clear error if missing.
    """
    frame_path = ROOT / "analysis" / "outputs" / "genotype_E354Q_frame.parquet"
    contrasts_path = ROOT / "analysis" / "outputs" / "genotype_E354Q_contrasts.parquet"
    if not (frame_path.exists() and contrasts_path.exists()):
        raise FileNotFoundError(
            "Missing genotype outputs. Run: "
            "uv run python analysis/pharmacogenomics/run.py"
        )
    return pl.read_parquet(frame_path), pl.read_parquet(contrasts_path)


def _residualize_chg(frame: pl.DataFrame, gene: str) -> pl.DataFrame:
    """Per-cell OLS residualization of Δ on age + sex + PC1-3 + base
    (and plate dummies for Olink cells only, since plate is sample-level
    Olink-specific batch info).

    Pools both arms within each (platform, visit) cell so the treatment
    effect is preserved in the residuals.
    """
    import statsmodels.api as sm
    import numpy as np

    sub = frame.filter(pl.col("gene") == gene).with_columns(
        (pl.col("value") - pl.col("base")).alias("chg"),
    )
    out_parts = []
    for (platform, visit), cell in sub.group_by(["platform", "visit"]):
        cell2 = cell.drop_nulls(subset=["chg"] + RESID_COVS)
        if cell2.height < len(RESID_COVS) + 5:
            continue
        y = cell2["chg"].to_numpy()
        X_num = cell2.select(RESID_COVS).to_numpy()
        if platform == "olink" and "plate" in cell2.columns:
            # Drop-first dummy encoding for the Olink plate factor — drops
            # one column to avoid collinearity with the constant. Singletons
            # (plate × cell with n=1) get dropped via the rank check.
            plate_dummies = (
                cell2.select("plate")
                .to_pandas()
                .assign(plate=lambda d: d["plate"].astype("category"))
                .pipe(
                    lambda d: __import__("pandas").get_dummies(
                        d["plate"], drop_first=True, dtype=float
                    )
                )
                .to_numpy()
            )
            if plate_dummies.shape[1] > 0:
                X_num = np.hstack([X_num, plate_dummies])
        X = sm.add_constant(X_num, has_constant="add")
        # Use pinv-based OLS so rank-deficient designs (e.g. plate dummies
        # collinear with sparse cells) don't blow up — coefficients are
        # minimum-norm; residuals are the same as the full-rank fit.
        resid = y - sm.OLS(y, X).fit().predict(X)
        out_parts.append(cell2.with_columns(pl.Series("chg_resid", resid)))
    return pl.concat(out_parts, how="diagonal_relaxed")


def _draw_genotype_cell(
    ax,
    cell: pl.DataFrame,
    gene: str,
    platform: str,
    visit: str,
    contrasts: pl.DataFrame,
    show_ylabel: bool,
) -> None:
    """Boxplot + jittered swarm of residualized Δ by genotype × arm."""
    import numpy as np

    visit_short = "W" + visit[len("Week") :]  # Week24 → W24
    # Genotype positions on x: 0/1/2 are the row centers; within each row,
    # TZP shifts right (+0.18) and SEMA shifts left (-0.18) — same offset
    # idiom as the forest plot.
    ARM_OFFSET = {"TZP": +0.18, "SEMA": -0.18}
    BOX_WIDTH = 0.30

    ax.axhline(0, color=c["bold_grey"], linewidth=0.4, alpha=0.6, zorder=1)

    rng = np.random.default_rng(0)

    n_per_geno: dict[int, dict[str, int]] = {0: {}, 1: {}, 2: {}}

    for dosage in (0, 1, 2):
        for arm_short in ("SEMA", "TZP"):
            sub = cell.filter(
                (pl.col("dosage") == float(dosage))
                & (
                    pl.col("arm").str.starts_with(
                        "TZP" if arm_short == "TZP" else "SEMA"
                    )
                )
            )
            n_per_geno[dosage][arm_short] = sub.height
            if sub.is_empty():
                continue
            ys = sub["chg_resid"].to_numpy()
            x_center = dosage + ARM_OFFSET[arm_short]
            color = ARM_COLORS[arm_short]

            # Points first, narrow jitter so they cluster centrally and don't
            # spill outside the box edges. Slightly translucent so the box's
            # colored outline (drawn on top) reads as the dominant element.
            jx = rng.uniform(-BOX_WIDTH * 0.18, BOX_WIDTH * 0.18, size=len(ys))
            ax.scatter(
                np.full_like(ys, x_center) + jx,
                ys,
                s=3,
                c=color,
                alpha=0.40,
                edgecolors="none",
                zorder=2,
            )
            # Box on top of points: light translucent fill so points still
            # show through, strong colored outline + bold median for the
            # box to read clearly regardless of point density.
            ax.boxplot(
                ys,
                positions=[x_center],
                widths=BOX_WIDTH,
                showfliers=False,
                patch_artist=True,
                medianprops=dict(color=c["black"], linewidth=1.2),
                boxprops=dict(
                    facecolor=color, alpha=0.18, edgecolor=color, linewidth=0.9
                ),
                whiskerprops=dict(color=color, linewidth=0.7),
                capprops=dict(color=color, linewidth=0.7),
                zorder=10,
            )

    # X-tick labels with embedded n; the n_TZP/n_SEMA is a small grey second
    # line below each genotype label to make sample size legible without a
    # separate annotation row.
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([""] * 3)
    ax.set_xlim(-0.6, 2.6)

    from matplotlib import transforms

    base_trans = ax.get_xaxis_transform()  # x = data, y = axes-fraction
    LINE_OFFSET_PT = 4.5
    trans_up = transforms.offset_copy(
        base_trans,
        fig=ax.figure,
        x=0,
        y=-LINE_OFFSET_PT,
        units="points",
    )
    trans_dn = transforms.offset_copy(
        base_trans,
        fig=ax.figure,
        x=0,
        y=-LINE_OFFSET_PT * 3.0,
        units="points",
    )
    # Genotype label (bold) on the top line; per-arm n on the bottom line
    # in grey, slash-joined in box-display order (SEMA/TZP, matching the
    # left-to-right offset layout above the labels).
    for dosage in (0, 1, 2):
        ax.text(
            dosage,
            0,
            GENO_LABELS[dosage],
            transform=trans_up,
            ha="center",
            va="top",
            fontsize=FS_BODY,
            fontweight="bold",
            clip_on=False,
        )
        ns = n_per_geno[dosage]
        ax.text(
            dosage,
            0,
            f"n={ns.get('SEMA', 0)}/{ns.get('TZP', 0)}",
            transform=trans_dn,
            ha="center",
            va="top",
            fontsize=FS_AUX,
            color=c["bold_grey"],
            clip_on=False,
        )

    if show_ylabel:
        ax.set_ylabel(f"Δ {gene} (residuals)", fontsize=FS_EMPH, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, length=2, width=0.4, pad=1)
    # Only show the 3 major ticks at genotype centres (GG/GC/CC). Suppress
    # the auto-generated minor ticks that matplotlib draws between them.
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="x", which="major", length=2, width=0.4, pad=1)
    ax.tick_params(axis="x", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Pad y-axis upper limit just enough to clear the contrast labels above
    # the highest data point — no wasted empty space, since the Week pill
    # now sits ABOVE the top spine rather than inside the plot.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.12)

    # Week pill ABOVE the plot, bottom edge anchored to the top of the
    # y-axis (axes_y=1.0). Volcano "Week N" rounded-box styling.
    ax.text(
        0.5,
        1.0,
        f"Week {visit_short[1:]}",
        transform=ax.transAxes,
        fontsize=FS_BODY,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=c["black"],
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#F0F0F0",
            edgecolor="#CCCCCC",
            linewidth=0.4,
        ),
        zorder=11,
        clip_on=False,
    )

    # MMRM contrast strip — platform-specific (each subplot looks up its
    # own (gene × visit × geno × platform) row from the per-platform MMRM).
    # Sits INSIDE the plot just below the top spine, above the data.
    ctx = contrasts.filter(
        (pl.col("gene") == gene)
        & (pl.col("visit") == visit_short)
        & (pl.col("platform") == platform)
    )
    if ctx.is_empty():
        return
    # GC nudged right so it doesn't visually touch GG. GG is kept centered
    # to avoid colliding with the y-axis / its tick labels on the leftmost
    # subplots. CC's short "ns" stays centered.
    ANNO_X_OFFSET = {0: 0.0, 1: +0.10, 2: 0.0}
    ctx_lookup = {r["geno"]: r for r in ctx.iter_rows(named=True)}
    for dosage, geno_label in GENO_LABELS.items():
        r = ctx_lookup.get(geno_label)
        if r is None or r["p"] is None:
            continue
        # Numeric p-value (no "p=" prefix). All significant p-values share
        # the same scientific-notation format and are bold (via mathtext
        # \mathbf, since matplotlib's `fontweight=bold` does not bolden
        # mathtext glyphs). "ns" (regular weight, grey) for p≥0.05.
        p = r["p"]
        if p >= 0.05:
            label, sig_color, weight = "ns", c["bold_grey"], "normal"
        else:
            mantissa, exp_str = f"{p:.1e}".split("e")
            if mantissa.endswith(".0"):
                mantissa = mantissa[:-2]
            label = f"$\\mathbf{{{mantissa}{{\\times}}10^{{{int(exp_str)}}}}}$"
            sig_color, weight = c["black"], "bold"
        ax.text(
            dosage + ANNO_X_OFFSET[dosage],
            0.96,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FS_AUX,
            fontweight=weight,
            color=sig_color,
            zorder=11,
        )


def _draw_genotype_panel(fig, gene: str) -> tuple[list, list]:
    """Render the 1×4 genotype boxplot panel for `gene`. Uses two side-by-
    side gridspecs (Olink columns / SomaScan columns) so the inter-platform
    gap is wider than the inter-visit gap, making the grouping legible.
    Returns the list of axes (4) and the (left, right) bounding x for the
    panel-c banner span.
    """
    frame, contrasts = _load_e354q_data()
    resid_frame = _residualize_chg(frame, gene)

    # Split the panel width into two equal halves (Olink | SomaScan) with a
    # platform gap between them. Each half is a 1×2 gridspec.
    half_w = (PANEL_C_RIGHT - PANEL_C_LEFT - PANEL_C_PLATFORM_GAP) / 2
    olink_left = PANEL_C_LEFT
    olink_right = PANEL_C_LEFT + half_w
    soma_left = olink_right + PANEL_C_PLATFORM_GAP
    soma_right = PANEL_C_RIGHT

    gs_olink = fig.add_gridspec(
        1,
        2,
        left=olink_left,
        right=olink_right,
        top=PLOT_C_TOP,
        bottom=PLOT_C_BOT,
        wspace=PANEL_C_WSPACE_TIGHT,
    )
    gs_soma = fig.add_gridspec(
        1,
        2,
        left=soma_left,
        right=soma_right,
        top=PLOT_C_TOP,
        bottom=PLOT_C_BOT,
        wspace=PANEL_C_WSPACE_TIGHT,
    )

    axes = []
    for ci, (platform, visit, _vs) in enumerate(PANEL_C_CELLS):
        gs = gs_olink if platform == "olink" else gs_soma
        col_idx = ci % 2  # 0 or 1 within each platform pair
        ax = fig.add_subplot(gs[0, col_idx])
        cell = resid_frame.filter(
            (pl.col("platform") == platform) & (pl.col("visit") == visit)
        )
        # Y-label only on the very leftmost subplot (Olink-W24). SomaScan
        # keeps its own tick labels (different scale) but no axis title —
        # "Δ GENE (residuals)" reads identically across platforms.
        show_ylabel = ci == 0
        _draw_genotype_cell(
            ax,
            cell,
            gene,
            platform,
            visit,
            contrasts,
            show_ylabel,
        )
        axes.append(ax)
    return axes, (olink_left, olink_right, soma_left, soma_right)


# ---------------------------------------------------------------------------
# Shared banner helper (mediation pattern)
# ---------------------------------------------------------------------------
def _draw_shared_banner(
    fig,
    ax_left,
    ax_right,
    *,
    bold_text: str,
    tail_text: str,
    left_edge_text: str = "",
    right_edge_text: str = "",
    gap: float | None = None,
    span_x: tuple[float, float] | None = None,
) -> float:
    """Optional ``gap`` overrides the default BANNER_GAP between the
    referenced axes' top and the banner's bottom. Used by panel d, which
    needs extra room above the plot for the Week pill + contrast strip.

    Optional ``span_x`` (figure-fraction (x0, x1)) overrides the banner's
    horizontal extent — used by the narrow GIP and forest panels, whose short
    banner text needs to span the empty label gutter beside the plot to fit.
    Vertical placement still tracks ``ax_left``'s top edge.
    """
    bb_left = ax_left.get_position()
    bb_right = ax_right.get_position()
    x0 = span_x[0] if span_x is not None else bb_left.x0
    x1 = span_x[1] if span_x is not None else bb_right.x1
    banner_y = bb_left.y1 + (gap if gap is not None else BANNER_GAP)
    patch = mpatches.FancyBboxPatch(
        (x0, banner_y),
        x1 - x0,
        BANNER_HEIGHT,
        boxstyle="square,pad=0",
        facecolor="#E8E8E8",
        edgecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(patch)
    mid_x = (x0 + x1) / 2
    mid_y = banner_y + BANNER_HEIGHT / 2

    t_bold = fig.text(
        mid_x,
        mid_y,
        bold_text,
        fontsize=FS_EMPH,
        fontweight="bold",
        color=c["black"],
        ha="left",
        va="center",
    )
    t_tail = fig.text(
        mid_x,
        mid_y,
        tail_text,
        fontsize=FS_AUX,
        color="#666666",
        ha="left",
        va="center",
    )
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    w_bold = t_bold.get_window_extent().transformed(inv).width
    w_tail = t_tail.get_window_extent().transformed(inv).width
    new_bold_x = mid_x - (w_bold + w_tail) / 2
    t_bold.set_x(new_bold_x)
    t_tail.set_x(new_bold_x + w_bold)

    if left_edge_text:
        fig.text(
            x0 + 0.006,
            mid_y,
            left_edge_text,
            fontsize=FS_AUX,
            fontweight="bold",
            color="#444",
            ha="left",
            va="center",
        )
    if right_edge_text:
        fig.text(
            x1 - 0.006,
            mid_y,
            right_edge_text,
            fontsize=FS_AUX,
            fontweight="bold",
            color="#444",
            ha="right",
            va="center",
        )
    return mid_y


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def _truncate_celltype(disp: str, max_chars: int = 16) -> str:
    """Shorten a cell-type display label so the 2-line tissue/cell labels fit
    the narrow main-figure gutter (the full label is the one produced by
    _gip_panel.load_hormone_strengths).

    Two steps: (1) drop the trailing parenthetical qualifier
    ("ciliated columnar cell (tracheobronchial)" → "ciliated columnar cell");
    (2) if still too long and ≥3 words, collapse interior words to keep only
    the first + head noun ("ciliated columnar cell" → "ciliated cell")."""
    s = re.sub(r"\s*\(.*\)\s*$", "", disp).strip()
    words = s.split()
    if len(s) > max_chars and len(words) >= 3:
        s = f"{words[0]} {words[-1]}"
    return s


def render(
    gene: str,
    output_name: str,
    save: bool = True,
    fdr_legend_text: str = "FDR: * <0.05   ** <0.01   *** <0.001",
) -> None:
    """Render one figure for `gene`: panel a trajectory + panel b STEP forest."""
    init_figure_theme()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Panel a: GIP-receiving cell types (replotted Hormone Cell Atlas). Sets up
    # the mechanism — the pancreatic γ/PP cell (the PPY source) is the top
    # GIP-receiving cell type, motivating the PPY response to GIP-containing TZP.
    gs_gip = fig.add_gridspec(
        1,
        1,
        left=GIP_LEFT,
        right=GIP_RIGHT,
        top=GIP_TOP,
        bottom=GIP_BOT,
    )
    ax_gip = fig.add_subplot(gs_gip[0, 0])
    _gip_data = load_hormone_strengths(hormone="gip", top_n=GIP_TOP_N, path=ATLAS_DATA)
    # Truncate cell-type labels (drop parentheticals) for the narrow panel.
    _gip_data = _gip_data.with_columns(
        pl.col("celltype_display")
        .map_elements(_truncate_celltype, return_dtype=pl.String)
        .alias("celltype_display")
    )
    draw_gip_panel(
        ax_gip, _gip_data, hormone="gip", subtitle=False, show_colorbar=False
    )

    # Panel b: trajectory (Olink + SomaScan, 1×2 grid)
    gs_a = fig.add_gridspec(
        1,
        2,
        left=PANEL_A_LEFT,
        right=PANEL_A_RIGHT,
        top=PLOT_TOP,
        bottom=PLOT_BOT,
        wspace=PANEL_A_WSPACE,
    )
    ax_o = fig.add_subplot(gs_a[0, 0])
    ax_s = fig.add_subplot(gs_a[0, 1])
    _render_trajectory_panel(ax_o, "olink", gene, show_x=True, show_ylabel=True)
    _render_trajectory_panel(ax_s, "soma", gene, show_x=True, show_ylabel=False)

    # Panel b: STEP forest (single axis)
    gs_b = fig.add_gridspec(
        1,
        1,
        left=PANEL_B_LEFT,
        right=PANEL_B_RIGHT,
        top=PLOT_TOP,
        bottom=PLOT_BOT,
    )
    ax_b = fig.add_subplot(gs_b[0, 0])
    _draw_step_forest(ax_b, gene)

    fig.canvas.draw()

    # Panel a banner — gene · contrast · covariates, with platform edge labels.
    _draw_shared_banner(
        fig,
        ax_o,
        ax_s,
        bold_text=gene,
        tail_text=" · TZP vs SEMA · adj. age, sex, baseline protein",
        left_edge_text="Olink",
        right_edge_text="SomaScan",
    )

    # Panel c banner (forest) — same structure as panel b. SeqId in top-left of
    # the plot carries platform context; tail is short so it fits (Maretty et al.
    # 2025 attribution goes in the figure caption). The banner spans leftward
    # over the (empty) STEP-label gutter so the text isn't cramped.
    # Banner spans only the forest plot box (no extension over the empty STEP
    # gutter); the title centers within it. The wider forest fits the short text.
    _draw_shared_banner(
        fig,
        ax_b,
        ax_b,
        bold_text=gene,
        tail_text=" · SEMA vs PBO",
    )

    # Panel a title — plain text, NO gray banner. The other panels' banners
    # encode an analysis contrast + covariates; panel a is replotted external
    # reference data (Hormone Cell Atlas), so a bold title + inline grey
    # attribution reads cleaner than a banner strip. Centered over the bars.
    bb_gip = ax_gip.get_position()
    # Title band aligned with the b/c banner band (panel a's plot rises above
    # this, so use the shared band rather than the raised plot top).
    _gip_band0 = PLOT_TOP + BANNER_GAP
    _gip_cx = (GIP_LEFT + GIP_RIGHT) / 2
    fig.text(
        _gip_cx,
        _gip_band0 + BANNER_HEIGHT * 0.78,
        "GIP-receiving cells",
        fontsize=FS_EMPH,
        fontweight="bold",
        color=c["black"],
        ha="center",
        va="center",
    )
    fig.text(
        _gip_cx,
        _gip_band0 + BANNER_HEIGHT * 0.22,
        "Data from Fei et al., 2026",
        fontsize=FS_AUX,
        color="#666666",
        ha="center",
        va="center",
    )

    # Attribution note in the banner gap above the forest plot.
    bb_b = ax_b.get_position()
    ns_y = bb_b.y1 + BANNER_GAP / 2
    fig.text(
        (bb_b.x0 + bb_b.x1) / 2,
        ns_y,
        "Data from Maretty et al., 2025",
        fontsize=FS_AUX,
        color="#666666",
        ha="center",
        va="center",
    )

    # Panel labels — bottom of label aligned with top of grey banner. Both at
    # the same y so they line up across the row.
    bb_o = ax_o.get_position()
    bb_s = ax_s.get_position()
    bb_b = ax_b.get_position()
    banner_top = bb_o.y1 + BANNER_GAP + BANNER_HEIGHT
    # b = trajectory, c = forest (a is the GIP panel to the left; d is genotype
    # below). Letters sit in the gutter just left of each panel — a small offset
    # keeps b/c out of the neighbouring panel to their left.
    for ax, label in ((ax_o, "b"), (ax_b, "c")):
        bb = ax.get_position()
        fig.text(
            bb.x0 - 0.045,
            banner_top,
            label,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    # Panel a letter — at the far left, aligned with the b/c letters (banner_top).
    fig.text(
        0.006,
        banner_top,
        "a",
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ---- Bottom rows (axis labels + legends) -------------------------------
    # Two horizontal stripes shared by both panels: axis labels above,
    # legends below. Same y across panels for visual alignment; same fontsize
    # within each row so the hierarchy reads consistently.

    # Row 1: axis labels. Pin b's "Week" and c's "log₂ FC" to panel a's own
    # x-axis-label baseline so all three row-1 x-labels sit at the same height.
    fig.canvas.draw()
    _gxl = ax_gip.xaxis.label.get_window_extent()
    gip_xlabel_y = fig.transFigure.inverted().transform((0, (_gxl.y0 + _gxl.y1) / 2))[1]
    fig.text(
        (bb_o.x0 + bb_s.x1) / 2,
        gip_xlabel_y,
        "Week",
        fontsize=BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )
    fig.text(
        (bb_b.x0 + bb_b.x1) / 2,
        gip_xlabel_y,
        "log₂ FC (Week 68)",
        fontsize=BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )

    # Row 2: legends.
    # Across-treatment FDR star key — left-aligned under panel b. The level(s)
    # documented are caller-controlled: PPY's stars are all *** so it passes the
    # short form, while REG4 (which has a * star) keeps the full key.
    fig.text(
        bb_o.x0,
        LEGEND_Y,
        fdr_legend_text,
        fontsize=BOTTOM_LABEL_FONTSIZE - 0.5,
        color=c["black"],
        ha="left",
        va="center",
        style="italic",
    )

    legend_handles_a = [
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
    # Arm legend right-aligned to the trajectory pair's right edge.
    fig.legend(
        handles=legend_handles_a,
        loc="center right",
        bbox_to_anchor=(bb_s.x1, LEGEND_Y),
        ncol=2,
        fontsize=BOTTOM_LABEL_FONTSIZE,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.9,
    )

    legend_handles_b = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linewidth=0,
            markerfacecolor=COLOR_NS,
            markeredgecolor=c["black"],
            markeredgewidth=0.3,
            markersize=4,
            label="Primary",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            linewidth=0,
            markerfacecolor="white",
            markeredgecolor=COLOR_NS,
            markeredgewidth=0.9,
            markersize=4,
            label="+∆Wgt+HbA1c",
        ),
    ]
    # Single-line (ncol=2) forest key, right-aligned at the forest edge; the
    # arm legend sits far enough left to clear it.
    fig.legend(
        handles=legend_handles_b,
        loc="center",
        bbox_to_anchor=((bb_b.x0 + bb_b.x1) / 2, LEGEND_Y),
        ncol=2,
        fontsize=BOTTOM_LABEL_FONTSIZE,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.3,
        columnspacing=0.6,
    )

    # ---- Panel c: GIPR E354Q genotype boxplots ----------------------------
    axes_c, (olink_l, olink_r, soma_l, soma_r) = _draw_genotype_panel(fig, gene)
    fig.canvas.draw()

    # Panel c banner — gene · contrast · covariates, with platform group
    # labels at the inner edges of each platform pair (similar to panel a).
    ax_c_first, ax_c_last = axes_c[0], axes_c[-1]
    _draw_shared_banner(
        fig,
        ax_c_first,
        ax_c_last,
        bold_text=gene,
        tail_text=" · Δ from baseline · adj. age, sex, genetic PC 1-3, baseline protein",
        gap=PANEL_C_BANNER_GAP,
    )
    bb_c1 = axes_c[0].get_position()
    bb_c4 = axes_c[-1].get_position()
    banner_c_mid_y = bb_c1.y1 + PANEL_C_BANNER_GAP + BANNER_HEIGHT / 2
    # Platform group labels at the inner-left of each platform pair (matches
    # the Olink/SomaScan edge-label convention from panel a).
    fig.text(
        olink_l + 0.006,
        banner_c_mid_y,
        "Olink",
        fontsize=FS_AUX,
        fontweight="bold",
        color="#444",
        ha="left",
        va="center",
    )
    fig.text(
        soma_r - 0.006,
        banner_c_mid_y,
        "SomaScan",
        fontsize=FS_AUX,
        fontweight="bold",
        color="#444",
        ha="right",
        va="center",
    )

    # Panel d label — genotype boxplots (a=GIP, b=trajectory, c=forest above).
    banner_c_top = bb_c1.y1 + PANEL_C_BANNER_GAP + BANNER_HEIGHT
    fig.text(
        bb_c1.x0 - 0.075,
        banner_c_top,
        "d",
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Vertical divider through the platform gap — visually separates the
    # Olink pair (cols 1-2) from the SomaScan pair (cols 3-4). Centered
    # between the right edge of the Olink-W72 plot (olink_r) and the
    # leftmost tip of the SomaScan-W24 y-tick labels (measured from the
    # rendered figure to handle tick-label width robustly).
    fig.canvas.draw()
    soma_w24_ax = axes_c[2]
    inv = fig.transFigure.inverted()
    tick_left_edges = [
        inv.transform((lbl.get_window_extent().x0, 0))[0]
        for lbl in soma_w24_ax.get_yticklabels()
        if lbl.get_text()
    ]
    soma_tick_left = min(tick_left_edges) if tick_left_edges else soma_l
    div_x = (olink_r + soma_tick_left) / 2
    fig.add_artist(
        mlines.Line2D(
            [div_x, div_x],
            [PLOT_C_BOT - 0.045, PLOT_C_TOP],
            transform=fig.transFigure,
            color=c["bold_grey"],
            linewidth=0.8,
            zorder=20,
            clip_on=False,
        )
    )

    # X-axis label below panel c (single shared label across the 4 subplots).
    fig.text(
        (bb_c1.x0 + bb_c4.x1) / 2,
        PANEL_C_AXIS_LABEL_Y,
        "rs1800437 (GIPR E354Q) genotype",
        fontsize=BOTTOM_LABEL_FONTSIZE,
        ha="center",
        va="center",
    )

    if save:
        out_path = str(FIGURES_DIR / output_name)
        save_figure(fig, out_path, formats=("pdf",))
        print(f"Saved → {out_path}.pdf")
        plt.close()


if __name__ == "__main__":
    # PPY's across-treatment stars are all *** → short FDR key. The REG4
    # counterpart is now its own script, supp_fig15_reg4.py (no GIP panel).
    render("PPY", "fig5_ppy", fdr_legend_text="FDR: *** <0.001")
