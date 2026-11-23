"""Theme and style constants shared across the figure scripts.

Holds the Lilly web palette, the matplotlib rcParams (``FIGURE_RC``), the
default tick labels of linear and log axes, and the PDF writer. Figures are plain matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.scale as mscale
import numpy as np
from matplotlib import cycler
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, ScalarFormatter

from analysis.design import ARM_LABELS as ARM_LABELS
from analysis.statistics import FDR_LABEL
from analysis.statistics import FDR_THRESHOLD as FDR_THRESHOLD

# Lilly brand colors, web (RGB) values; every figure takes its palette from this one dict.
c: dict[str, str] = {
    "red": "#D31710",
    "pink": "#FFDAD4",
    "white": "#FFFFFF",
    "black": "#191919",
    "bold_brown": "#501009",
    "vibrant_coral": "#FD9485",
    "vibrant_gold": "#F4C003",
    "bold_blue": "#003A6C",
    "vibrant_azure": "#86B3F2",
    "bold_green": "#00422C",
    "bold_grey": "#818C94",
    "neutral_rose": "#F9EEED",
    "neutral_orange": "#FFDCC6",
    "neutral_cream": "#F9F0E0",
    "neutral_sage": "#BEEBE6",
    "neutral_stone": "#D8E4EC",
}

# CVD-safe categorical ordering of the default color cycle: blue first, coral and gold
# next, red and green as far apart as possible.
CATEGORICAL_ORDER: tuple[str, ...] = (
    "bold_blue",
    "vibrant_coral",
    "vibrant_gold",
    "bold_grey",
    "red",
    "vibrant_azure",
    "bold_brown",
    "bold_green",
)

# Leader lines from repelled gene labels to their points (volcano, cross-study scatters).
LEADER_GRAY = "#929591"

# ---------------------------------------------------------------------------
# Font ladder (5 pt min, 7 pt max body text, 8 pt bold panel letters)
# ---------------------------------------------------------------------------
# Tiers in half-point steps so they read distinctly. Figures are authored at
# print size, so these are literal printed point sizes. Nothing may go below
# FS_MIN (5 pt) or above FS_EMPH (7 pt) except panel letters (FS_PANEL, 8 pt).

FS_PANEL = 8.0  # NAVIGATION — panel letters a, b, c (bold)
FS_EMPH = 7.0  # EMPHASIS    — direction cues, banner bold names (== 7 pt max)
FS_NARR = 6.5  # NARRATIVE   — axis labels, legends, week pills, counts
FS_BODY = 6.0  # BODY        — gene labels, marker IDs, forest y-labels
FS_AUX = 5.5  # AUXILIARY   — tick labels, banner taglines
FS_MIN = 5.0  # FLOOR       — never render text smaller than this

# Semantic aliases onto the FS tiers above, so callers can use descriptive
# names while sharing one source of truth.
FS_PANEL_LABEL = FS_PANEL
FS_BANNER_BOLD = FS_EMPH
FS_BANNER_TAIL = FS_AUX
FS_PILL = FS_NARR
FS_AXLABEL = FS_NARR
FS_TICK = FS_AUX
FS_GENE_LABEL = FS_BODY
FS_FOREST_YLABEL = FS_BODY
FS_DIRECTION = FS_EMPH
FS_LEGEND = FS_NARR
FS_LEGEND_TITLE = FS_EMPH

_INTER_FONTS = [
    "Inter-Regular.ttf",
    "Inter-Bold.ttf",
    "Inter-Italic.ttf",
    "Inter-BoldItalic.ttf",
]
# The environment's matplotlib font directory, where `pixi run setup-fonts` installs Inter.
_ENV_FONT_DIR = os.path.join(mpl.get_data_path(), "fonts", "ttf")


# Every rcParam the figures depend on: 5-7 pt body text, 8 pt bold panel letters,
# sans-serif; line widths thin enough that 600 dpi prints stay crisp at 88-180 mm widths.
FIGURE_RC: dict[str, object] = {
    # Fonts. Inter first, then Arial and Linux fallbacks.
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS_NARR,
    "axes.titlesize": FS_EMPH,
    "axes.labelsize": FS_AXLABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND_TITLE,
    "figure.titlesize": FS_PANEL,
    "figure.titleweight": "bold",
    # Math text is set in the sans face ("sans" resolves through font.sans-serif); STIX
    # sans covers glyphs Inter lacks. The custom fontset looks up every style when a label
    # is parsed, so the unused calligraphic slot points at an installed face.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.sf": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans:italic",
    "mathtext.fallback": "stixsans",
    # Lines and ticks.
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    "errorbar.capsize": 3.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "xtick.minor.pad": 2.0,
    "ytick.minor.pad": 2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    # Axes, grid and the default color cycle.
    "axes.prop_cycle": cycler(color=[c[name] for name in CATEGORICAL_ORDER]),
    "axes.titlecolor": "black",
    "axes.titlepad": 5.0,
    "axes.facecolor": "white",
    "axes.grid.which": "both",
    "grid.alpha": 0.1,
    "grid.color": "black",
    "grid.linewidth": 0.6,
    "image.interpolation": "none",
    "figure.figsize": [4.0, 4.0],
    # White figure background.
    "figure.facecolor": "white",
    # Legends without a frame (house style); the box keys apply when a figure
    # turns the frame on.
    "legend.frameon": False,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.borderpad": 0.5,
    "legend.borderaxespad": 0.0,
    "legend.handleheight": 1.0,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    # Save defaults: vector PDF with TrueType (editable) text, 600 dpi raster copies.
    "savefig.format": "pdf",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

class TickFormatter(ScalarFormatter):
    """Default major tick labels: ``ScalarFormatter`` numbers without the clutter.

    Labels drop trailing decimal zeros and never read as negative zero. A tick that is zero
    up to floating-point error (float32 ticks, drifted ``arange`` grids) reads ``0``. A
    nonzero tick that the axis-wide precision would round to zero is shown to three
    significant digits instead. The offset and the ``×10ⁿ`` scale are set in math text.

    The class reads ``ScalarFormatter``'s ``locs``, ``format`` and ``orderOfMagnitude``
    attributes (matplotlib 3.11 deprecates all three and 3.13 removes them, so a matplotlib
    upgrade needs this class revisited). Labels are plain text with no
    locale-aware decimal separator; under usetex they keep a hyphen for the minus sign. On
    a log axis the tick values are powers of ten, so the labels from ``_LogScale`` are
    meant for base 10.
    """

    # A tick at most this fraction of the smallest tick gap away from zero is noise.
    NOISE_FRACTION = 1e-9
    # Ticks built by repeated addition or in float32 drift from the exact grid by a few
    # units in the last place of their largest value; 16 covers a grid of about that many
    # steps and stays far below any tick a figure means to show.
    EPSILON_MULTIPLE = 16
    # A tick inside the rounding-error bound is still a real tick when another tick lies
    # within this multiple of its distance from zero (ticks 1e-3 apart on a 1e12 offset).
    NEIGHBOR_FRACTION = 1e-3
    # Significant digits of a nonzero tick the shared precision would round to zero.
    TINY_DIGITS = 3

    def __init__(self, useOffset: bool | float | None = None) -> None:
        """Create the formatter with math-text offsets.

        Args:
            useOffset: Offset mode as in ``ScalarFormatter``; ``None`` follows rcParams.
        """
        super().__init__(useOffset=useOffset, useMathText=True)

    def set_locs(self, locs) -> None:
        """Set the tick locations, the shared precision and the noise thresholds.

        Args:
            locs: Tick locations in data units.
        """
        super().set_locs(locs)
        if len(self.locs) == 0:
            return
        # The shared format is "%1.<decimals>f", wrapped in $\mathdefault{...}$ by math text.
        self._decimals = int(self.format.partition("%1.")[2].partition("f")[0])
        self._ticks = np.unique(np.asarray(self.locs))
        steps = np.diff(self._ticks)
        spacing = steps.min() if steps.size else np.ptp(self.axis.get_view_interval())
        self._noise = self.NOISE_FRACTION * spacing
        # Rounding error scales with the largest tick, in the ticks' own precision.
        kind = self._ticks.dtype.kind
        precision = np.finfo(self._ticks.dtype if kind == "f" else float).eps
        self._rounding_error = self.EPSILON_MULTIPLE * precision * np.abs(self._ticks).max()

    def _is_zero(self, x: float, shifted: float) -> bool:
        """Whether the tick at ``x`` is zero up to floating-point error.

        Args:
            x: Tick value in data units.
            shifted: ``x`` minus the offset.

        Returns:
            True when ``shifted`` is within the noise threshold, or within the rounding
            error of the ticks while no other tick is close to ``x``.
        """
        if abs(shifted) <= self._noise:
            return True
        if abs(shifted) > self._rounding_error:
            return False
        others = np.abs(self._ticks - x)
        others = others[others > 0]
        nearest = others.min() if others.size else np.inf
        return abs(shifted) <= self.NEIGHBOR_FRACTION * nearest

    def __call__(self, x: float, pos: int | None = None) -> str:  # noqa: ARG002 - matplotlib signature
        """Format the tick at ``x``.

        Args:
            x: Tick value in data units.
            pos: Tick position index (unused).

        Returns:
            The tick label.
        """
        if len(self.locs) == 0:
            return ""
        shifted = x - self.offset
        if self._is_zero(x, shifted):
            return "0"
        value = shifted / 10.0**self.orderOfMagnitude
        # Rounding through a string also turns 9.999999999999999e-06 into 1e-05.
        rounded = float(f"{value:.{self._decimals}f}")
        if rounded == 0:
            rounded = float(f"{value:.{self.TINY_DIGITS}g}")
        label = np.format_float_positional(rounded, trim="-")
        return label if self.get_usetex() else self.fix_minus(label)


def _minor_ticks_visible(axis) -> bool:
    name = axis.axis_name if axis.axis_name in "xy" else "x"
    return bool(mpl.rcParams[f"{name}tick.minor.visible"])


class _LinearScale(mscale.LinearScale):
    """Linear axis whose default major labels come from ``TickFormatter``.

    Together with ``_LogScale`` these are the axis-scale defaults the figures rely on;
    symlog, logit and function scales keep matplotlib's defaults.
    """

    def set_default_locators_and_formatters(self, axis) -> None:
        """Set matplotlib's linear defaults, then the ``TickFormatter`` labels.

        Args:
            axis: The matplotlib axis to configure.
        """
        super().set_default_locators_and_formatters(axis)
        axis.set_major_formatter(TickFormatter())


class _LogScale(mscale.LogScale):
    """Log axis with ``TickFormatter`` labels and unlabeled minor ticks at 1-9."""

    def __init__(self, axis, *, subs=None, **kwargs) -> None:
        super().__init__(axis, subs=np.arange(1, 10) if subs is None else subs, **kwargs)

    def set_default_locators_and_formatters(self, axis) -> None:
        """Major ticks at powers of the base, minor ticks at each multiple 1-9.

        Args:
            axis: The matplotlib axis to configure.
        """
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(TickFormatter())
        axis.set_minor_locator(
            LogLocator(self.base, self.subs) if _minor_ticks_visible(axis) else NullLocator()
        )
        axis.set_minor_formatter(NullFormatter())


def _raw_save_figure(
    fig,
    path: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[str]:
    """Save ``fig`` once per format; PDFs carry ``PDF_METADATA``, so no dates.

    Args:
        fig: Figure to save.
        path: Output path; its suffix is replaced by each format's extension.
        formats: File formats to write.
        dpi: Resolution of raster formats.
        **savefig_kwargs: Passed to ``Figure.savefig``.

    Returns:
        The written paths, one per format.
    """
    base = Path(path).with_suffix("")
    written = []
    for fmt in formats:
        out = f"{base}.{fmt}"
        kwargs = {"metadata": PDF_METADATA, **savefig_kwargs} if fmt == "pdf" else savefig_kwargs
        if fmt == "png":
            fig.savefig(out, dpi=dpi, **kwargs)
        else:
            fig.savefig(out, **kwargs)
        written.append(out)
    return written


def init_figure_theme() -> None:
    """Register the Inter fonts and the axis-scale defaults, then apply ``FIGURE_RC``.

    Raises:
        RuntimeError: When the Inter font is not registered with matplotlib.
    """
    # setup-fonts installs Inter into the environment's matplotlib font directory. The font list
    # cached in the user's matplotlib directory is shared with other environments and may name
    # another Inter build, so every Inter entry outside that directory is dropped, and each Inter
    # file there that the font manager has not registered is added; if none is found, fail
    # rather than let the sans-serif list fall through to Arial or DejaVu.
    env_dir = os.path.realpath(_ENV_FONT_DIR)
    fm.fontManager.ttflist = [
        entry
        for entry in fm.fontManager.ttflist
        if entry.name != "Inter" or os.path.dirname(os.path.realpath(entry.fname)) == env_dir
    ]
    fm.fontManager._findfont_cached.cache_clear()
    registered = {os.path.realpath(entry.fname) for entry in fm.fontManager.ttflist}
    for font_name in _INTER_FONTS:
        path = os.path.join(_ENV_FONT_DIR, font_name)
        if os.path.exists(path) and os.path.realpath(path) not in registered:
            fm.fontManager.addfont(path)
    if "Inter" not in {entry.name for entry in fm.fontManager.ttflist}:
        raise RuntimeError(
            f"The Inter font is not registered with matplotlib (looked in {_ENV_FONT_DIR}). "
            "Run `pixi run setup-fonts` once, and again after every `pixi install`."
        )
    mscale.register_scale(_LinearScale)
    mscale.register_scale(_LogScale)
    mpl.rcParams.update(FIGURE_RC)


# ---------------------------------------------------------------------------
# Color constants (volcano / direction conventions)
# ---------------------------------------------------------------------------

# Directional convention (volcano / scatter / forest). Arm colors: TZP dark navy,
# SEMA medium gray (distinct in grayscale).
COLOR_UP = "#0C376D"  # higher in TZP — RGB(12, 55, 109)
COLOR_DOWN = "#5A5A5A"  # higher in SEMA — RGB(90, 90, 90)
COLOR_NS = "#D3D3D3"  # not significant (neutral light gray, variant-agnostic)


def fdr_stars(fdr: float | None) -> str:
    """Significance stars of an FDR: ``***`` < 0.001, ``**`` < 0.01, ``*`` < ``FDR_THRESHOLD``.

    Args:
        fdr: The q-value, or None when the contrast is missing.

    Returns:
        The stars, or an empty string when ``fdr`` is None, NaN or not below the threshold.
    """
    if fdr is None:
        return ""
    if fdr < 1e-3:
        return "***"
    if fdr < 1e-2:
        return "**"
    if fdr < FDR_THRESHOLD:
        return "*"
    return ""


# FDR star levels (``fdr_stars``) and the bound each stands for, in key order.
FDR_STAR_KEY = {"*": f"<{FDR_LABEL}", "**": "<0.01", "***": "<0.001"}
# The star key printed under the trajectory figures.
BETWEEN_TREATMENT_FDR_KEY = "Between-treatment FDR: " + "   ".join(
    f"{stars} {bound}" for stars, bound in FDR_STAR_KEY.items()
)

# ---------------------------------------------------------------------------
# Treatment arm constants
# ---------------------------------------------------------------------------

ARM_COLORS = {"TZP": COLOR_UP, "SEMA": COLOR_DOWN}
# ARM_LABELS (short arm key -> legend label) is analysis.design.ARM_LABELS, imported above.

# ---------------------------------------------------------------------------
# Trajectory classification constants
# ---------------------------------------------------------------------------

TRAJECTORY_COLORS = {
    "Late-onset ↓": c["neutral_stone"],
    "Progressive ↓": c["bold_blue"],
    "Sustained ↓": c["vibrant_azure"],
    "Transient ↓": c["bold_grey"],
    "Reversal ↓↑": c["bold_green"],
    "Reversal ↑↓": c["bold_brown"],
    "Transient ↑": c["vibrant_gold"],
    "Sustained ↑": c["vibrant_coral"],
    "Progressive ↑": c["red"],
    "Late-onset ↑": c["neutral_orange"],
}

# ---------------------------------------------------------------------------
# Journal page boxes and shared diagram/forest tokens
# ---------------------------------------------------------------------------

# PDFs carry no creation or modification timestamp, so two runs write the same bytes.
PDF_METADATA = {"CreationDate": None, "ModDate": None}
INK = "#1A1A1A"  # primary label text
# Document-level "Figure N" page stamp; a page label, so above the panel-letter ladder.
FS_FIGURE_LABEL = 10.0

# Main-figure page box: at most 183 x 247 mm. Extended Data box: at most 180 x 170 mm.
# A larger figure is reported, not scaled.
MM_PER_INCH = 25.4
MAIN_MAX_W_MM = 183.0
MAIN_MAX_H_MM = 247.0
MAIN_MAX_W_IN = MAIN_MAX_W_MM / MM_PER_INCH
MAIN_MAX_H_IN = MAIN_MAX_H_MM / MM_PER_INCH
ED_MAX_W_MM = 180.0
ED_MAX_H_MM = 170.0
ED_MAX_W_IN = ED_MAX_W_MM / MM_PER_INCH
ED_MAX_H_IN = ED_MAX_H_MM / MM_PER_INCH

# Neutral grays and line widths of the study-design diagram and the mediation forests.
GRAY_ZERO_LINE = "#888888"
GRAY_THRESHOLD = "#CCCCCC"
GRAY_MID = "#666666"
GRAY_LIGHT = "#D9D9D9"
GRAY_RULE = "#E0E0E0"
GRAY_MUTED = "#8A8A8A"
GRAY_BAND = "#F2F2F2"  # shaded alternate-class bands of the Fig 3b skyline
GRAY_BASELINE = "#BBBBBB"  # zero line of the Fig 3b skyline
GRAY_LEGEND_INK = "#444444"  # text of the Fig 3b legend strip (the Fig 3a "Sig:" ink)
CALLOUT_INK = "#5C4A48"
LW_NONE = 0.0
LW_GUIDE = 0.6
LW_FOREST_CI = 1.0
LW_FOREST_MARKER_EDGE = 0.6
LW_DIAGRAM_ARROW = 0.6
LW_DIAGRAM_ARROW_MAIN = 0.7
LW_DIAGRAM_LINE = 0.5
LW_DIAGRAM_BOX = 0.5
LW_DIAGRAM_CARD = 0.6
LW_DIAGRAM_CALLOUT = 0.4
LW_DIAGRAM_RULE = 0.3
LW_DIAGRAM_SPINE = 0.7
LW_DIAGRAM_BRANCH = 0.6
LW_DIAGRAM_SYMBOL_EDGE = 0.6

# Platform colors of the study-design diagram (Olink green, SomaScan brown).
PLATFORM_COLORS = {"olink": c["bold_green"], "soma": c["bold_brown"]}
PLATFORM_LABELS = {"olink": "Olink Explore HT", "soma": "SomaScan 11K"}
# The name each platform is printed with; the one definition for the figures.
PLATFORM_SHORT = {"olink": "Olink", "soma": "SomaScan"}


def fade_hex(hex_color: str, frac: float) -> str:
    """Fade a color toward white by ``frac`` (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(tuple(ch + (1.0 - ch) * frac for ch in rgb))
