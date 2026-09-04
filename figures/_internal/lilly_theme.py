"""Lilly brand theme for UltraPlot figures.

Programmatic registration of Lilly colors, colormaps, and Nature-compliant
rcParams. Idempotent; safe to call apply_lilly_theme() multiple times.

Usage:
    import ultraplot as uplt
    from lilly_theme import apply_lilly_theme, LILLY_COLORS

    apply_lilly_theme()
    fig, axs = uplt.subplots(ncols=2, journal='nat2')

Color values are taken from Lilly's brand reference. Print values are calibrated
for CMYK reproduction and used by default; web values are available via
LILLY_COLORS['web'] for digital-only output.
"""
from __future__ import annotations

import warnings
from typing import Literal

import matplotlib as mpl
import ultraplot as uplt
from matplotlib.colors import LinearSegmentedColormap

ColorVariant = Literal["print", "web"]

# ---------------------------------------------------------------------------
# Color reference
# ---------------------------------------------------------------------------

LILLY_COLORS: dict[str, dict[str, str]] = {
    "print": {
        "red": "#E1251B",
        "pink": "#FBCFC8",
        "white": "#FFFFFF",
        "black": "#212121",
        "bold_brown": "#521207",
        "vibrant_coral": "#F58E7D",
        "vibrant_gold": "#FFC709",
        "bold_blue": "#0F3A85",
        "vibrant_azure": "#99BFE5",
        "bold_green": "#144B2D",
        "bold_grey": "#8A969E",
        "neutral_rose": "#FDE8E5",
        "neutral_orange": "#FDD1B0",
        "neutral_cream": "#FFF0D8",
        "neutral_sage": "#C6DCD8",
        "neutral_stone": "#E4EBF1",
    },
    "web": {
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
    },
}

# CVD-safe categorical ordering. Blue first; orange/coral and gold next;
# red and green separated as far as possible.
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



def get_cycle(
    n: int | None = None, variant: ColorVariant = "print"
) -> list[str]:
    """Return the Lilly categorical cycle (full or first n colors)."""
    cycle = [LILLY_COLORS[variant][name] for name in CATEGORICAL_ORDER]
    return cycle[:n] if n else cycle


def build_cmap(
    name: str = "Lilly_Diverging", variant: ColorVariant = "print"
) -> LinearSegmentedColormap:
    """Return a fresh Lilly colormap object by name.

    Use this when handing a cmap to a library that uses matplotlib's registry
    in a quirky way (e.g. PyComplexHeatmap). Passing the object directly
    bypasses string-based lookup and avoids cross-library registry conflicts.

    Names: "Lilly_Diverging", "Lilly_Reds", "Lilly_Blues" (case-insensitive).
    """
    cmaps = _build_cmaps(variant)
    # Case-insensitive lookup
    for cmap_name, cmap in cmaps.items():
        if cmap_name.lower() == name.lower():
            return cmap
    available = sorted(cmaps)
    raise KeyError(f"Unknown Lilly colormap '{name}'. Available: {available}")


# ---------------------------------------------------------------------------
# Colormap construction
# ---------------------------------------------------------------------------


def _build_cmaps(variant: ColorVariant = "print") -> dict[str, LinearSegmentedColormap]:
    """Construct Lilly diverging, reds, and blues colormaps."""
    c = LILLY_COLORS[variant]
    return {
        "Lilly_Diverging": LinearSegmentedColormap.from_list(
            "Lilly_Diverging",
            [c["bold_blue"], c["vibrant_azure"], c["white"], c["vibrant_coral"], c["red"]],
            N=256,
        ),
        "Lilly_Reds": LinearSegmentedColormap.from_list(
            "Lilly_Reds",
            [c["white"], c["pink"], c["vibrant_coral"], c["red"], c["bold_brown"]],
            N=256,
        ),
        "Lilly_Blues": LinearSegmentedColormap.from_list(
            "Lilly_Blues",
            [c["white"], c["vibrant_azure"], c["bold_blue"]],
            N=256,
        ),
    }


# ---------------------------------------------------------------------------
# rcParams for Nature-family submission
# ---------------------------------------------------------------------------

# Nature: 5-7pt body text, 8pt bold panel labels, sans-serif (Arial/Helvetica).
# Line widths kept thin so 600 dpi prints stay crisp at 88-180 mm widths.
NATURE_RC: dict[str, object] = {
    # Fonts. Arial first (Nature spec), then fallbacks for Linux/DNAnexus.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "legend.title_fontsize": 6.5,
    "figure.titlesize": 8.0,
    # Lines and ticks. Thin enough to print clean at 88 mm.
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    # No mirror ticks. scienceplots adds these; Nature figures rarely use them.
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend without frame (Nature house style).
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    # Save defaults: vector PDF, 600 dpi PNG, tight bbox, transparent.
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "pdf.fonttype": 42,  # TrueType, editable text in Illustrator/Acrobat.
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # keep text as text in SVG
    # Keep the figure background white for screenshots into slides.
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# UltraPlot-specific rc settings (panel labels, layout).
ULTRAPLOT_RC: dict[str, object] = {
    "abc": "a",  # lowercase a, b, c (Nature style; Cell uses uppercase)
    "abcloc": "ul",  # upper-left
    "abcweight": "bold",
    "abcsize": 8.0,  # Nature: 8pt bold panel labels
    "subplots.refwidth": "1.7in",  # ~43 mm reference subplot, fits 2-col Nature
    "subplots.tight": True,
    "grid": False,  # Nature figures rarely use grids
    "subplots.share": False,  # paper panels usually have different scales
    "title.weight": "normal",
    "label.weight": "normal",
    "tick.minor": True,
}


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

_APPLIED: bool = False


def apply_lilly_theme(
    variant: ColorVariant = "print",
    register_globally: bool = True,
    force: bool = False,
) -> None:
    """Apply the Lilly theme to UltraPlot and matplotlib.

    Parameters
    ----------
    variant : "print" or "web"
        Use print colors (CMYK-calibrated, default) or web colors (digital).
    register_globally : bool
        If True, register named colors and the 'lilly' cycle so they can be
        accessed by name from any UltraPlot call (e.g. cycle='lilly',
        color='lilly_red'). If False, only set rcParams.
    force : bool
        Re-apply even if already applied this session. Useful if you've
        switched variants.
    """
    global _APPLIED
    if _APPLIED and not force:
        return

    # 1. Register named colors with UltraPlot. Prefix with "lilly_" to avoid
    #    clobbering standard color names (e.g. matplotlib's 'red').
    if register_globally:
        named = {f"lilly_{name}": hex_ for name, hex_ in LILLY_COLORS[variant].items()}
        uplt.register_colors(**named)

        # 2. Register the categorical cycle as 'lilly'. UltraPlot expects a
        #    DiscreteColormap instance whose .name becomes the registered name.
        from ultraplot.colors import DiscreteColormap

        cycle = DiscreteColormap(get_cycle(variant=variant), name="lilly")
        uplt.register_cycles(cycle)

        # 3. Build and register continuous colormaps. Each cmap is named on
        #    construction; register_cmaps takes the instance positionally.
        #    Also register with matplotlib's registry so libraries that bypass
        #    UltraPlot's namespace (like PyComplexHeatmap) can find them.
        with warnings.catch_warnings():
            # UltraPlot emits "Overwriting 'X' that was already registered" on
            # every re-apply; silence since we register intentionally.
            warnings.filterwarnings("ignore", message="Overwriting")
            for cmap in _build_cmaps(variant).values():
                uplt.register_cmaps(cmap)
                try:
                    mpl.colormaps.register(cmap, name=cmap.name, force=True)
                except (ValueError, AttributeError):
                    pass

                rev = cmap.reversed()
                rev.name = f"{cmap.name}_r"
                uplt.register_cmaps(rev)
                try:
                    mpl.colormaps.register(rev, name=rev.name, force=True)
                except (ValueError, AttributeError):
                    pass

    # 4. matplotlib rcParams.
    mpl.rcParams.update(NATURE_RC)

    # 5. UltraPlot rc (its own keyspace; uplt.rc accepts both mpl + ultraplot keys).
    for key, val in ULTRAPLOT_RC.items():
        uplt.rc[key] = val

    # 6. Default cycle and continuous cmap.
    if register_globally:
        uplt.rc["cycle"] = "lilly"
        uplt.rc["cmap.diverging"] = "Lilly_Diverging"
        uplt.rc["cmap.sequential"] = "Lilly_Blues"

    _APPLIED = True


# ---------------------------------------------------------------------------
# Convenience helpers for common figure operations
# ---------------------------------------------------------------------------


def save_figure(
    fig,
    path: str,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[str]:
    """Save a figure in multiple formats. Returns list of written paths.

    PDF: vector, editable text (Nature production requirement).
    PNG: 600 dpi (line art per Nature spec; safe for slide reuse).
    """
    from pathlib import Path

    base = Path(path).with_suffix("")
    written = []
    for fmt in formats:
        out = f"{base}.{fmt}"
        if fmt == "png":
            fig.savefig(out, dpi=dpi, **savefig_kwargs)
        else:
            fig.savefig(out, **savefig_kwargs)
        written.append(out)
    return written
