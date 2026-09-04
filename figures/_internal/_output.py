"""Figure output helpers with regression-harness overrides."""

from __future__ import annotations

import os
from pathlib import Path

from ._theme import _raw_save_figure

_FIGURE_OUTPUT_DIR_ENV = "FIGURE_OUTPUT_DIR"
_FIGURE_OUTPUT_FORMATS_ENV = "FIGURE_OUTPUT_FORMATS"


def _resolve_output_path(path: str | Path) -> Path:
    """Return the final output stem, honoring harness output-dir overrides."""
    base = Path(path).with_suffix("")
    override_dir = os.environ.get(_FIGURE_OUTPUT_DIR_ENV)
    if not override_dir:
        return base
    return Path(override_dir) / base.name


def _resolve_output_formats(formats: tuple[str, ...]) -> tuple[str, ...]:
    """Return the requested formats, honoring harness format overrides."""
    override = os.environ.get(_FIGURE_OUTPUT_FORMATS_ENV)
    if not override:
        return formats
    resolved = tuple(
        fmt.strip().lstrip(".").lower() for fmt in override.split(",") if fmt.strip()
    )
    if not resolved:
        raise ValueError("FIGURE_OUTPUT_FORMATS must name at least one format.")
    return resolved


def save_figure(
    fig,
    path: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[str]:
    """Save a figure, with optional harness-controlled path/format overrides."""
    out_path = _resolve_output_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _raw_save_figure(
        fig,
        str(out_path),
        formats=_resolve_output_formats(formats),
        dpi=dpi,
        **savefig_kwargs,
    )
