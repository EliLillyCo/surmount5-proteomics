"""Supplementary Fig. 1: pooled proteome PCA by week (a) and mixed-model variance attribution (b).

a, PC1 vs PC2 per platform, one panel per week on shared axes, colored by arm.
b, share of the variance of each of PCs 1–6 attributed to each model term, one column block per
platform, rows grouped by term type.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from scipy.stats import t

from analysis.design import ARMS, RAW_CONTROL_VALUE, RAW_TREAT_VALUE, WEEKS
from analysis.output_layout import PCA_PARTITION, PCA_SCORES, PCA_VARIANCE
from analysis.statistics import SEED
from figures._internal._common import (
    ARM_COLORS,
    FS_AXLABEL,
    FS_BODY,
    FS_MIN,
    FS_PANEL,
    FS_TICK,
    c,
    init_figure_theme,
    save_figure,
)

PLATFORMS = (("olink", "Olink Explore HT"), ("soma", "SomaScan 11K"))
ARM_KEY = {RAW_TREAT_VALUE: "TZP", RAW_CONTROL_VALUE: "SEMA"}
ARM_NAME = {"TZP": "Tirzepatide", "SEMA": "Semaglutide"}
PCS = ("PC1", "PC2", "PC3", "PC4", "PC5", "PC6")
ZERO_LINE = {"color": "0.85", "lw": 0.4, "zorder": 0}
POINT = {"s": 4, "alpha": 0.5, "linewidths": 0, "rasterized": True}
# (partition component, display label, group); None = an ungrouped single row.
ROWS = (
    ("Participant", "Participant (random intercept)", None),
    ("WEEK", "Timepoint", "Study design"),
    ("ARM", "Treatment arm", "Study design"),
    ("WEEK:ARM", "Arm × timepoint", "Study design"),
    ("PLATE", "Assay plate", "Sample and plate"),
    ("QC_METRIC", "Median assay signal¹", "Sample and plate"),
    ("SEX", "Sex", "Clinical"),
    ("AGE", "Age", "Clinical"),
    ("BMI", "BMI", "Clinical"),
    ("PCTCHG_WGT", "Weight change (%)", "Clinical"),
    ("HBA1C", "HbA1c", "Clinical"),
    ("SBP", "Systolic BP", "Clinical"),
    ("LDLC", "LDL-C", "Clinical"),
    ("TG", "Triglycerides", "Clinical"),
    ("EGFR", "eGFR", "Clinical"),
    ("LOG_ALT", "ALT (log)", "Clinical"),
    ("Residual", "Residual (unexplained)", None),
)
FOOTNOTE = ("¹ Olink: detection-weighted, standardized median of each sample's PCNormalizedNPX across assays "
            "(Z_Median); SomaScan: the vendor hybridization control scale factor (HybControlNormScale).")
GROUP_GAP = 0.35  # white gap between row groups, in row heights
SHARE_CMAP = LinearSegmentedColormap.from_list("white_tzp", (c["white"], ARM_COLORS["TZP"]))

# Layout, inches.
FIG_W, FIG_H = 7.8, 4.65
SCATTER_X, SCATTER_SIDE, SCATTER_GAP = 0.45, 1.0, 0.08
PLATFORM_STEP = 0.85  # vertical space between the two platform rows beyond the panel height
A_B_GAP = 0.42  # panel a's right edge to panel b's group strip
ROW_H, COL_W, BLOCK_GAP = 0.165, 0.184, 0.1


def _load() -> tuple[pl.DataFrame, dict[str, dict[str, float]], dict[tuple[str, str, str], float]]:
    """Shuffled sample scores, per-platform variance explained and percent shares."""
    scores = pl.read_parquet(PCA_SCORES).with_columns(pl.col("TRT01A").replace_strict(ARM_KEY).alias("ARM"))
    # Random draw order so neither arm paints over the other.
    scores = scores[np.random.default_rng(SEED).permutation(scores.height)]
    variance = pl.read_parquet(PCA_VARIANCE)
    fractions = {p: dict(variance.filter(pl.col("platform") == p).select("PC", "variance_explained").iter_rows())
                 for p, _ in PLATFORMS}
    shares = pl.read_parquet(PCA_PARTITION).filter(pl.col("PC").is_in(PCS))
    pct = {(r["platform"], r["PC"], r["component"]): 100 * r["share"] for r in shares.iter_rows(named=True)}
    return scores, fractions, pct


def _axes(fig, x: float, y: float, w: float, h: float):
    """Axes at inch coordinates (x, y = bottom-left)."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, y / H, w / W, h / H))


def _text_in(fig, text: str, size: float, rotation: float = 0) -> tuple[float, float]:
    """Rendered (width, height) of ``text`` in inches."""
    t = fig.text(0, 0, text, fontsize=size, rotation=rotation)
    box = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def _wrap(fig, text: str, size: float, width: float) -> str:
    """Greedy word wrap of ``text`` to ``width`` inches as rendered."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and _text_in(fig, trial, size)[0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return "\n".join([*lines, line])


def _row_positions(rows) -> list[float]:
    """Top of each row in row units, with a gap wherever the group changes."""
    ys, y = [], 0.0
    for i, row in enumerate(rows):
        if i and (row[2] is None or row[2] != rows[i - 1][2]):
            y += GROUP_GAP
        ys.append(y)
        y += 1
    return ys


def _panel_a(fig, scores: pl.DataFrame, fractions, top: float) -> float:
    """Week-faceted PC1 vs PC2 scatters; returns the bottom of the legend row."""
    W, H = fig.get_size_inches()
    weeks = sorted(WEEKS)
    width = len(weeks) * SCATTER_SIDE + (len(weeks) - 1) * SCATTER_GAP
    for i, (platform, name) in enumerate(PLATFORMS):
        d = scores.filter(pl.col("platform") == platform)
        span = {pc: (d[pc].min(), d[pc].max()) for pc in ("PC1", "PC2")}
        lims = {pc: (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)) for pc, (lo, hi) in span.items()}
        row_top = top - i * (SCATTER_SIDE + PLATFORM_STEP)
        fig.text((SCATTER_X + width / 2) / W, (row_top + 0.2) / H, name, fontsize=FS_BODY, ha="center", va="bottom")
        for j, week in enumerate(weeks):
            ax = _axes(fig, SCATTER_X + j * (SCATTER_SIDE + SCATTER_GAP), row_top - SCATTER_SIDE,
                       SCATTER_SIDE, SCATTER_SIDE)
            w = d.filter(pl.col("WEEK") == week)
            ax.scatter(w["PC1"].to_numpy(), w["PC2"].to_numpy(), c=[ARM_COLORS[a] for a in w["ARM"]], **POINT)
            for arm in ARM_NAME:
                z = w.filter(pl.col("ARM") == arm)
                mean = [z[pc].mean() for pc in ("PC1", "PC2")]
                ci = [t.ppf(0.975, z.height - 1) * z[pc].std() / np.sqrt(z.height)
                      for pc in ("PC1", "PC2")]
                ax.errorbar(*mean, xerr=ci[0], yerr=ci[1], fmt="D", markersize=4,
                            markerfacecolor="white", markeredgecolor=ARM_COLORS[arm],
                            markeredgewidth=1.1, ecolor=ARM_COLORS[arm], elinewidth=1,
                            capsize=2, zorder=5)
            ax.axhline(0, **ZERO_LINE)
            ax.axvline(0, **ZERO_LINE)
            ax.set_xlim(lims["PC1"])
            ax.set_ylim(lims["PC2"])
            ax.set_xlabel(f"PC1 ({fractions[platform]['PC1']:.1%})", fontsize=FS_AXLABEL)
            if j == 0:
                ax.set_ylabel(f"PC2 ({fractions[platform]['PC2']:.1%})", fontsize=FS_AXLABEL)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_title(f"Week {week}", fontsize=FS_TICK, pad=2)
    legend_y = top - 2 * SCATTER_SIDE - PLATFORM_STEP - 0.36
    handles = [Patch(facecolor=ARM_COLORS[a], edgecolor="none", label=ARM_NAME[a]) for a in tuple(ARMS)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.2 / W, legend_y / H), ncol=len(handles),
               frameon=False, fontsize=FS_TICK, handlelength=1.0, handleheight=0.8, handletextpad=0.35,
               columnspacing=0.9, borderaxespad=0, borderpad=0)
    return SCATTER_X + width


def _panel_b(fig, pct, x: float, top: float) -> None:
    """Grouped-row share heatmap: one PC1–PC6 block per platform, shared row labels and colorbar."""
    W, H = fig.get_size_inches()
    ys = _row_positions(ROWS)
    extent = ys[-1] + 1
    height = extent * ROW_H
    bottom = top - height
    values = {p: np.array([[pct[(p, pc, comp)] for pc in PCS] for comp, _, _ in ROWS]) for p, _ in PLATFORMS}
    vmax = float(np.ceil(max(v.max() for v in values.values()) / 10) * 10)
    norm = Normalize(0, vmax)

    group_w = max(_text_in(fig, group, FS_TICK, rotation=90)[0]
                  for _, _, group in ROWS if group) + 0.06
    label_w = max(_text_in(fig, label, FS_TICK)[0] for _, label, _ in ROWS)
    x_heat = x + group_w + label_w + 0.06
    block_w = COL_W * len(PCS)
    right = x_heat + 2 * block_w + BLOCK_GAP
    fig.text(x / W, (top + 0.33) / H, "b", fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")

    group_ax = _axes(fig, x, bottom, group_w, height)
    group_ax.set_xlim(0, 1)
    group_ax.set_ylim(extent, 0)
    group_ax.axis("off")
    for group in dict.fromkeys(g for _, _, g in ROWS if g):
        span = [y for y, r in zip(ys, ROWS, strict=True) if r[2] == group]
        strip_label = "Sample\nand plate" if group == "Sample and plate" else group
        group_ax.text(0.4, (span[0] + span[-1] + 1) / 2, strip_label, rotation=90, ha="center", va="center",
                      fontsize=FS_TICK)
        group_ax.plot([0.95, 0.95], [span[0] + 0.1, span[-1] + 0.9], color="0.6", lw=0.5, solid_capstyle="butt")

    for i, (platform, name) in enumerate(PLATFORMS):
        ax = _axes(fig, x_heat + i * (block_w + BLOCK_GAP), bottom, block_w, height)
        ax.set_xlim(-0.5, len(PCS) - 0.5)
        ax.set_ylim(extent, 0)
        for r, y in enumerate(ys):
            for k in range(len(PCS)):
                v = values[platform][r, k]
                ax.add_patch(Rectangle((k - 0.5, y), 1, 1, facecolor=SHARE_CMAP(norm(v)), edgecolor=c["white"], lw=0.6))
                ax.text(k, y + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=FS_TICK,
                        color=c["white"] if v > 0.5 * vmax else c["black"])
        ax.set_xticks(range(len(PCS)), PCS, fontsize=FS_TICK)
        ax.set_yticks([y + 0.5 for y in ys], [label for _, label, _ in ROWS] if i == 0 else [], fontsize=FS_TICK)
        ax.tick_params(length=0, pad=2, which="both")
        ax.minorticks_off()
        ax.spines[:].set_visible(False)
        ax.set_title(name, fontsize=FS_BODY, pad=3)

    cb_y = bottom - 0.34
    cb_ax = _axes(fig, x_heat, cb_y, right - x_heat, 0.07)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SHARE_CMAP), cax=cb_ax, orientation="horizontal")
    cb.set_label("Share of PC variance (%)", fontsize=FS_AXLABEL, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, which="major")
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4)
    # The footnote wraps within panel b, group strip to heatmap edge.
    fig.text(x / W, (cb_y - 0.33) / H, _wrap(fig, FOOTNOTE, FS_MIN, right - x), fontsize=FS_MIN,
             ha="left", va="top", linespacing=1.3)


def make_figure() -> plt.Figure:
    init_figure_theme()
    plt.rcParams["axes.grid"] = False
    scores, fractions, pct = _load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    top = FIG_H - 0.45
    fig.text(0.02 / FIG_W, (FIG_H - 0.03) / FIG_H, "a", fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")
    a_right = _panel_a(fig, scores, fractions, top)
    _panel_b(fig, pct, a_right + A_B_GAP, top + 0.1)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    save_figure(figure, Path(__file__).resolve().parent / "supp_fig_proteome_pca", formats=("pdf",))
    plt.close(figure)
