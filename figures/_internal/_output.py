"""Figure output helpers with regression-harness overrides."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
from PIL import Image
from pypdf import PdfReader, PdfWriter

from ._figure_bundle import stack_pages_vertically
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


def save_figure_stack(
    figs: Sequence,
    path: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
) -> list[str]:
    """Save same-width figures stacked top to bottom, edge to edge, as one figure per format.

    Each figure is saved at its own page size (``savefig.bbox`` standard, so no figure is
    cropped), then the PDFs are merged onto one page with pypdf and the PNGs pasted onto
    one canvas. Honors the same harness overrides as ``save_figure``.

    Args:
        figs: The figures in top-to-bottom order.
        path: Output stem (its suffix is dropped).
        formats: ``"pdf"`` and/or ``"png"``.
        dpi: Resolution of the PNG.

    Returns:
        The written paths.

    Raises:
        ValueError: When the figures differ in rendered width or a format is not pdf/png.
    """
    out_path = _resolve_output_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmts = _resolve_output_formats(formats)
    bad = [f for f in fmts if f not in ("pdf", "png")]
    if bad:
        raise ValueError(f"save_figure_stack writes pdf and png only, not {bad}")
    written = []
    with (
        tempfile.TemporaryDirectory(prefix="figure-stack-") as td,
        mpl.rc_context({"savefig.bbox": "standard"}),
    ):
        stems = [Path(td) / f"panel{i}" for i in range(len(figs))]
        for fig, stem in zip(figs, stems):
            _raw_save_figure(fig, str(stem), formats=fmts, dpi=dpi)
        for fmt in fmts:
            out = f"{out_path}.{fmt}"
            if fmt == "png":
                images = [Image.open(f"{stem}.png").convert("RGB") for stem in stems]
                if len({im.width for im in images}) != 1:
                    raise ValueError(f"panel widths differ: {[im.size for im in images]}")
                height = sum(im.height for im in images)
                canvas = Image.new("RGB", (images[0].width, height), "white")
                top = 0
                for im in images:
                    canvas.paste(im, (0, top))
                    top += im.height
                canvas.save(out, dpi=(dpi, dpi))
            else:
                pages = [PdfReader(f"{stem}.pdf").pages[0] for stem in stems]
                widths = {round(float(p.mediabox.width), 3) for p in pages}
                if len(widths) != 1:
                    raise ValueError(f"panel widths differ: {sorted(widths)} pt")
                writer = PdfWriter()
                writer.add_page(stack_pages_vertically(pages))
                writer.write(out)
            written.append(out)
    return written
