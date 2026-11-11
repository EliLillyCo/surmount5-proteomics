"""Figure 3b — disease-signature reversal skyline (HPA pan-disease blood atlas, Olink week 72).

One bar per adult HPA disease, grouped by disease class: the reversal score −ρ
(``analysis/disease_reversal.py``). A reversal bar (−ρ > 0) is navy when FDR < ``FDR_THRESHOLD``
and a lighter navy hatched in navy when only nominally significant (P < ``FDR_THRESHOLD``, FDR at
or above it); a bar toward disease (−ρ ≤ 0) uses dark gray and a lighter gray hatched in dark gray
for the same two levels; other bars are light gray. The legend keys the toward-disease levels
only when such bars are drawn.
Bar width is fixed by the layout template; slot pitch fills the panel.
Alternate classes sit on shaded bands that reach halfway into the inter-class gap, so the
short class names fit inside them (checked; ``ValueError`` otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from analysis.design import LATE_WEEK
from analysis.disease_reversal import output_path
from analysis.statistics import FDR_LABEL, FDR_THRESHOLD

from ._common import (
    COLOR_DOWN,
    COLOR_NS,
    COLOR_UP,
    FS_AUX,
    FS_BODY,
    FS_NARR,
    NAT_W2,
    init_figure_theme,
    panel_figure,
    panel_grid,
)
from ._theme import GRAY_BAND, GRAY_BASELINE, GRAY_LEGEND_INK, INK, MM_PER_INCH, fade_hex

PANEL_HEIGHT_MM = 85.0
# HPA adult disease classes in display order; ``bar_rows`` raises when the table's classes differ.
CLASS_ORDER = ("Metabolic", "Cardiovascular", "Psychiatric", "Cancer", "Autoimmune", "Infection")
CLASS_SHORT = {"Psychiatric": "Psych.", "Cardiovascular": "Cardiovasc."}
# Printed names of the diseases whose HPA names are too long for the x labels.
# ``check_short_names`` raises when a key is not among the plotted diseases (an HPA rename).
DISEASE_SHORT = {
    "Breast ductal carcinoma in situ": "Breast DCIS",
    "Small intestine neuroendocrine tumor": "Small-intestine NET",
    "Pituitary neuroendocrine tumor": "Pituitary NET",
    "Diffuse large B-cell lymphoma": "DLBCL",
    "Chronic lymphocytic leukemia": "CLL",
    "Acute myeloid leukemia": "AML",
    "Coronary artery calcification": "Coronary artery calc.",
    "Previous venous thromboembolism": "Previous VTE",
    "Acute venous thromboembolism": "Acute VTE",
    "Staphylococcus aureus bacteremia": "S. aureus bacteremia",
    "Streptococcal soft tissue infection": "Strep. soft-tissue inf.",
    "Viral hepatitis related cirrhosis": "Viral-hepatitis cirrhosis",
    "Alcohol-related liver disease": "Alcohol-related liver",
    "Systemic lupus erythematosus": "Lupus (SLE)",
    "Sjögrens syndrome": "Sjögren's syndrome",
    "E.coli pyelonephritis": "E. coli pyelonephritis",
}
YLABEL = "Disease-signature reversal (−ρ)"

COLOR_SIG = COLOR_UP  # FDR below the threshold, reversal
COLOR_TOWARD = COLOR_DOWN  # FDR below the threshold, toward disease
# P below the threshold, FDR not: the direction's color faded toward white and hatched in the full color, so it
# reads between the solid and the gray bars in print and in grayscale.
COLOR_NOMINAL = fade_hex(COLOR_SIG, 0.55)
COLOR_TOWARD_NOMINAL = fade_hex(COLOR_TOWARD, 0.55)
NOMINAL_HATCH = "//////"
HATCH_COLOR = COLOR_SIG
TOWARD_HATCH_COLOR = COLOR_TOWARD
HATCH_LW = 0.5

# Layout (inches, data units of the x slot grid).
LEFT_IN, RIGHT_IN, TOP_IN, BOTTOM_IN = 0.40, 0.04, 0.06, 1.30
CLASS_GAP = 0.9  # extra x units between classes
BAR_FRAC = 0.72  # bar width as a fraction of the template slot
# ``TEMPLATE_N`` slots in ``TEMPLATE_GAPS + 1`` classes fix the physical bar width; the slot
# pitch widens to fill the panel.
TEMPLATE_N, TEMPLATE_GAPS = 59, 6
LEGEND_Y_IN = 0.10
LEGEND_GAP = 0.008  # figure fraction between legend items
SWATCH_W_PT, SWATCH_H_PT = 5.5, 4.5
PT_PER_INCH = 72.0


@dataclass(frozen=True)
class Bar:
    """One bar of the skyline: a disease, its class, the reversal score, the fill color and the hatch."""

    Disease: str
    Class: str
    value: float
    color: str
    hatched: bool = False
    hatch_color: str = HATCH_COLOR


@dataclass(frozen=True)
class LegendText:
    """A legend label."""

    text: str
    bold: bool = False


@dataclass(frozen=True)
class LegendSwatch:
    """A ``SWATCH_W_PT`` x ``SWATCH_H_PT`` legend swatch, hatched like a nominal bar when ``hatched``."""

    color: str
    hatched: bool = False
    hatch_color: str = HATCH_COLOR


def load_reversal(week: int = LATE_WEEK) -> pl.DataFrame:
    """The reversal table of ``analysis/pipelines/run_disease_reversal.py`` at ``week``."""
    return pl.read_parquet(output_path(week))


def bar_rows(table: pl.DataFrame) -> list[Bar]:
    """Bars in class order, by decreasing −ρ within a class, with their colors.

    FDR < ``FDR_THRESHOLD``: solid ``COLOR_SIG`` (reversal, −ρ > 0) or ``COLOR_TOWARD`` (toward
    disease, −ρ ≤ 0); P < ``FDR_THRESHOLD`` with FDR at or above it: ``COLOR_NOMINAL`` or
    ``COLOR_TOWARD_NOMINAL``, hatched in the direction's solid color; otherwise gray.

    Args:
        table: Reversal table with ``Disease, Class, neg_rho, p, fdr``.

    Returns:
        One ``Bar`` per row of ``table``.

    Raises:
        ValueError: When the classes differ from ``CLASS_ORDER``.
    """
    if set(table["Class"]) != set(CLASS_ORDER):
        raise ValueError(f"disease classes {sorted(set(table['Class']))} differ from {CLASS_ORDER}")
    class_rank = {k: i for i, k in enumerate(CLASS_ORDER)}
    bars = []
    for r in table.iter_rows(named=True):
        significant = r["fdr"] < FDR_THRESHOLD
        nominal = not significant and r["p"] < FDR_THRESHOLD
        reversal = r["neg_rho"] > 0
        if significant:
            color = COLOR_SIG if reversal else COLOR_TOWARD
            bars.append(Bar(r["Disease"], r["Class"], r["neg_rho"], color))
        elif nominal:
            color, hatch = (COLOR_NOMINAL, HATCH_COLOR) if reversal else (COLOR_TOWARD_NOMINAL, TOWARD_HATCH_COLOR)
            bars.append(Bar(r["Disease"], r["Class"], r["neg_rho"], color, hatched=True, hatch_color=hatch))
        else:
            bars.append(Bar(r["Disease"], r["Class"], r["neg_rho"], COLOR_NS))
    bars.sort(key=lambda b: (class_rank[b.Class], -b.value))
    return bars


def legend_items(bars: list[Bar]) -> list[LegendText | LegendSwatch]:
    """The skyline legend: the reversal levels and ns, then the toward-disease levels drawn.

    The "Toward disease:" group appears only when a significant or nominal toward-disease bar is
    drawn, and lists only the levels present.
    """
    items: list[LegendText | LegendSwatch] = [
        LegendText("Sig:", bold=True),
        LegendSwatch(COLOR_SIG),
        LegendText(f"FDR < {FDR_LABEL},"),
        LegendSwatch(COLOR_NOMINAL, hatched=True),
        LegendText(f"P < {FDR_LABEL}, FDR ≥ {FDR_LABEL},"),
        LegendSwatch(COLOR_NS),
        LegendText("ns"),
    ]
    drawn = {b.color for b in bars}
    toward = [
        (LegendSwatch(COLOR_TOWARD), LegendText(f"FDR < {FDR_LABEL}")),
        (
            LegendSwatch(COLOR_TOWARD_NOMINAL, hatched=True, hatch_color=TOWARD_HATCH_COLOR),
            LegendText(f"P < {FDR_LABEL}, FDR ≥ {FDR_LABEL}"),
        ),
    ]
    toward = [(swatch, text) for swatch, text in toward if swatch.color in drawn]
    if toward:
        items[-1] = LegendText("ns;")
        items.append(LegendText("Toward disease:", bold=True))
        for i, (swatch, text) in enumerate(toward):
            last = i == len(toward) - 1
            items += [swatch, text if last else LegendText(f"{text.text},")]
    return items


def check_short_names(bars: list[Bar], short_names: dict[str, str]) -> None:
    """Raise when a key of ``short_names`` is not among the plotted diseases.

    Raises:
        ValueError: Naming every absent key (an HPA rename would otherwise leave a long label).
    """
    absent = sorted(set(short_names) - {b.Disease for b in bars})
    if absent:
        raise ValueError(f"short names for diseases that are not plotted: {absent}")


def legend_row(
    fig, x: float, y: float, items: list[LegendText | LegendSwatch], fs: float = FS_BODY
) -> float:
    """Lay out legend items left to right from (x, y), by their rendered text extents.

    Args:
        fig: The figure to draw on.
        x: Figure-fraction x of the first item.
        y: Figure-fraction y of the row's center.
        items: ``LegendText`` labels and ``LegendSwatch`` swatches.
        fs: Label font size in points.

    Returns:
        The figure-fraction x after the last item.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px = fig.bbox.width
    width_in, height_in = fig.get_size_inches()
    for item in items:
        if isinstance(item, LegendText):
            text = fig.text(
                x,
                y,
                item.text,
                fontsize=fs,
                va="center",
                ha="left",
                color=GRAY_LEGEND_INK,
                fontweight="bold" if item.bold else "normal",
            )
            x += text.get_window_extent(renderer).width / width_px + LEGEND_GAP
        else:
            swatch_h = (SWATCH_H_PT / PT_PER_INCH) / height_in
            swatch_w = (SWATCH_W_PT / PT_PER_INCH) / width_in
            fig.add_artist(
                mpatches.Rectangle(
                    (x, y - swatch_h / 2),
                    swatch_w,
                    swatch_h,
                    facecolor=item.color,
                    edgecolor=item.hatch_color if item.hatched else "none",
                    hatch=NOMINAL_HATCH if item.hatched else None,
                    hatch_linewidth=HATCH_LW,
                    lw=0,
                    transform=fig.transFigure,
                )
            )
            x += swatch_w + LEGEND_GAP * 0.6
    return x


def make_reversal_figure(
    table: pl.DataFrame | None = None,
    height_mm: float = PANEL_HEIGHT_MM,
    short_names: dict[str, str] = DISEASE_SHORT,
) -> Figure:
    """Fig 3b as its own 183 mm wide figure.

    Args:
        table: Reversal table; ``load_reversal()`` when None.
        height_mm: Figure height.
        short_names: Printed names of the diseases with long HPA names.

    Returns:
        The figure, not yet saved.

    Raises:
        ValueError: From ``bar_rows`` and ``check_short_names``, or when a class name does not
            fit inside its shaded band.
    """
    init_figure_theme()
    bars = bar_rows(load_reversal() if table is None else table)
    check_short_names(bars, short_names)
    height_in = height_mm / MM_PER_INCH
    fig = panel_figure(NAT_W2, height_in)
    axes_w_in = NAT_W2 - LEFT_IN - RIGHT_IN
    (ax,) = panel_grid(fig, [[1]], left=LEFT_IN, right=RIGHT_IN, top=TOP_IN, bottom=BOTTOM_IN)

    x_pos, cursor, class_span, prev_class = [], 0.0, {}, None
    for bar in bars:
        if prev_class is not None and bar.Class != prev_class:
            cursor += CLASS_GAP
        x_pos.append(cursor)
        class_span.setdefault(bar.Class, [cursor, cursor])[1] = cursor
        cursor += 1.0
        prev_class = bar.Class
    in_per_unit = axes_w_in / (cursor - 0.2 + 0.8)
    template_in_per_unit = axes_w_in / (TEMPLATE_N + TEMPLATE_GAPS * CLASS_GAP + 0.6)
    bar_width = BAR_FRAC * template_in_per_unit / in_per_unit
    for x, bar in zip(x_pos, bars):
        ax.add_patch(
            mpatches.Rectangle(
                (x - bar_width / 2, min(0, bar.value)),
                bar_width,
                abs(bar.value),
                facecolor=bar.color,
                edgecolor=bar.hatch_color if bar.hatched else "none",
                hatch=NOMINAL_HATCH if bar.hatched else None,
                hatch_linewidth=HATCH_LW,
                lw=0,
                zorder=3,
            )
        )
    values = [bar.value for bar in bars]
    ymax = float(np.ceil((max(values) + 0.12) * 10) / 10)
    ymin = float(min(0.0, np.floor((min(values) - 0.02) * 10) / 10))
    ticks = [t for t in np.arange(-1, 1.01, 0.2).round(1) if ymin <= t <= ymax - 0.08]
    ax.axhline(0, color=GRAY_BASELINE, lw=0.5, zorder=4)
    ax.set_xlim(-0.8, cursor - 0.2)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(YLABEL)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{(0.0 if abs(t) < 1e-9 else t):.1f}".replace("-", "−") for t in ticks])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="minor", left=False)

    for i, (name, (first, last)) in enumerate(class_span.items()):
        band = (first - 0.5 - CLASS_GAP / 2, last + 0.5 + CLASS_GAP / 2)
        if i % 2 == 0:
            ax.axvspan(*band, color=GRAY_BAND, lw=0, zorder=0)
        class_text = ax.text(
            (first + last) / 2,
            ymax - 0.017 * (ymax - ymin),
            CLASS_SHORT.get(name, name),
            fontsize=FS_NARR,
            fontweight="bold",
            ha="center",
            va="top",
            color=INK,
            zorder=6,
        )
        if i % 2 == 0:
            fig.canvas.draw()
            x0, x1 = ax.transData.inverted().transform(class_text.get_window_extent())[:, 0]
            if not (band[0] <= x0 and x1 <= band[1]):
                raise ValueError(
                    f"class label {name!r} spans x {x0:.2f}-{x1:.2f}, outside its band {band}"
                )
    label_transform = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for x, bar in zip(x_pos, bars):
        ax.text(
            x,
            -0.012,
            short_names.get(bar.Disease, bar.Disease),
            rotation=90,
            ha="center",
            va="top",
            fontsize=FS_AUX,
            color=INK,
            zorder=6,
            transform=label_transform,
        )
    # Same title and wording as the panel a "Sig:" legend.
    legend_row(fig, LEFT_IN / NAT_W2, LEGEND_Y_IN / height_in, legend_items(bars))
    return fig
