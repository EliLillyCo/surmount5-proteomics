"""Theme and style constants shared across the figure scripts.

Holds the Lilly web palette, the matplotlib rcParams (``FIGURE_RC``), the
default tick labels of linear and log axes, and the PDF writer. Figures are plain matplotlib.
"""

from __future__ import annotations

import os
import re
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

# A tick label that is all zeros (optionally signed), and the same with a sign.
_ZERO_LABEL = re.compile("\\A[-\N{MINUS SIGN}]?0(.0*)?\\Z")
_SIGNED_ZERO_LABEL = re.compile("\\A[-\N{MINUS SIGN}]0(.0*)?\\Z")


class TickFormatter(ScalarFormatter):
    """Default major tick labels: trailing decimal zeros trimmed, no negative zero.

    ``ScalarFormatter`` picks one precision for the whole axis from its limits, so ticks
    print as ``0.0, 0.5, 1.0``; these labels read ``0, 0.5, 1``. A nonzero tick that the
    shared precision would round to zero keeps enough digits to show its value, and the
    offset text is always set in math text.
    """

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format the tick at ``x``.

        Args:
            x: Tick value in data units.
            pos: Tick position index, passed through to ``ScalarFormatter``.

        Returns:
            The tick label.
        """
        string = self._fix_small_number(x, super().__call__(x, pos))
        if mpl.rcParams["axes.unicode_minus"] and not mpl.rcParams["text.usetex"]:
            string = string.replace("-", "\N{MINUS SIGN}")
        if _SIGNED_ZERO_LABEL.match(string):
            string = string[1:]
        if "." in string:
            string = string.rstrip("0").rstrip(".")
        return string

    def get_offset(self) -> str:
        """Offset text, always in math text.

        Returns:
            The offset label (empty when the axis has no offset).
        """
        use_math_text = self._useMathText
        self._useMathText = True
        offset = super().get_offset()
        self._useMathText = use_math_text
        return offset

    def _fix_small_number(self, x: float, string: str, precision_offset: int = 2) -> str:
        """Show enough digits for a nonzero tick that formatted as zero."""
        m = _ZERO_LABEL.match(string)
        if not m or x == 0:
            return string
        (decimals,) = m.groups()
        precision_init = len(decimals.lstrip(".")) if decimals else 0
        # Precision below floating-point error, on the axis' offset and scale.
        x -= getattr(self, "offset", 0)
        x /= 10 ** getattr(self, "orderOfMagnitude", 0)
        precision_true = max(0, 0 if x == 0 else -int(np.log10(abs(x)) // 1))
        precision_max = max(0, np.finfo(type(x)).precision - precision_offset)
        precision = min(precision_true, precision_max)
        string = ("{:.%df}" % precision).format(x)
        # Zero up to floating-point error: keep the axis' own precision.
        if _ZERO_LABEL.match(string):
            string = ("{:.%df}" % precision_init).format(0)
        return string


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
    # setup-fonts installs Inter into the environment's matplotlib font directory. A font list
    # cached before that install does not hold it, so each Inter file there that the font
    # manager has not registered is added here; if none is found, fail rather than let the
    # sans-serif list fall through to Arial or DejaVu.
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
