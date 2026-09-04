"""Theme and style constants shared across manuscript figure scripts."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.font_manager as fm

# Lilly theme assets live alongside this module.
from .lilly_theme import (  # noqa: F401
    LILLY_COLORS,
    apply_lilly_theme,
    build_cmap,
    save_figure as _raw_save_figure,
)

# Active palette variant. Nature recommends supplying RGB artwork (wider gamut;
# auto-converted to CMYK at print). All figures source their palette from this
# single dict so the variant can be flipped in one place.
PALETTE_VARIANT = "web"
c = LILLY_COLORS[PALETTE_VARIANT]

# ---------------------------------------------------------------------------
# Font ladder (Nature: 5 pt min, 7 pt max body text, 8 pt bold panel letters)
# ---------------------------------------------------------------------------
# Tiers in half-point steps so they read distinctly. Figures are authored at
# print size, so these are literal printed point sizes. Nothing may go below
# FS_MIN (5 pt) or above FS_EMPH (7 pt) except panel letters (FS_PANEL, 8 pt).

FS_PANEL = 8.0  # NAVIGATION — panel letters a, b, c (bold; Nature spec)
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
FS_COUNTS = FS_NARR
FS_LEGEND = FS_NARR
FS_LEGEND_TITLE = FS_EMPH

_INTER_FONTS = [
    "Inter-Regular.ttf",
    "Inter-Bold.ttf",
    "Inter-Italic.ttf",
    "Inter-BoldItalic.ttf",
]


def init_figure_theme() -> None:
    """Register Inter fonts, apply Lilly theme (RGB/web variant), set rcParams.

    Forces the ``web`` palette so every figure renders in the RGB color space
    Nature recommends, regardless of any earlier module-level ``apply_lilly_theme``
    call that may have registered the print variant first.
    """
    for font_name in _INTER_FONTS:
        path = os.path.expanduser(f"~/.local/share/fonts/{font_name}")
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    apply_lilly_theme(variant=PALETTE_VARIANT, force=True)
    mpl.rcParams["font.sans-serif"] = [
        "Inter",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    # Align matplotlib's default text sizes with the FS_* ladder so figures
    # that rely on rcParams (ticks, axis labels, legends) match figures that
    # set sizes explicitly. This is the paper's font policy, layered on top of
    # the brand theme; lilly_theme.py is left untouched.
    mpl.rcParams.update(
        {
            "font.size": FS_NARR,
            "axes.titlesize": FS_EMPH,
            "axes.labelsize": FS_AXLABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "legend.title_fontsize": FS_LEGEND_TITLE,
            "figure.titlesize": FS_PANEL,
        }
    )


# ---------------------------------------------------------------------------
# Color constants (volcano / direction conventions)
# ---------------------------------------------------------------------------

# Directional convention (volcano / scatter / forest). Manuscript-specific
# arm colors: TZP = dark navy, SEMA = medium grey. These override the Lilly
# brand red/bold_blue so the two arms are distinguishable in greyscale print
# and clearly encode the active-comparator contrast rather than implying a
# positive/negative valence via red/blue.
COLOR_UP = "#0C376D"  # higher in TZP — RGB(12, 55, 109)
COLOR_DOWN = "#5A5A5A"  # higher in SEMA — RGB(90, 90, 90)
COLOR_NS = "#D3D3D3"  # not significant (neutral light grey, variant-agnostic)
FDR_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Treatment arm constants
# ---------------------------------------------------------------------------

ARM_COLORS = {"TZP": "#0C376D", "SEMA": "#5A5A5A"}  # RGB(12,55,109), RGB(90,90,90)
ARM_LABELS = {"TZP": "TZP", "SEMA": "SEMA"}

# ---------------------------------------------------------------------------
# Trajectory classification constants
# ---------------------------------------------------------------------------

TRAJECTORY_ORDER = [
    "Late-onset ↓",
    "Progressive ↓",
    "Sustained ↓",
    "Transient ↓",
    "Reversal ↓↑",
    "Reversal ↑↓",
    "Transient ↑",
    "Sustained ↑",
    "Progressive ↑",
    "Late-onset ↑",
]

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
