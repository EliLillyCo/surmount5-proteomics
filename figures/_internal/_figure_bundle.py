"""Shared helpers for combined manuscript figure bundle wrappers."""

from __future__ import annotations

import contextlib
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
from pypdf import PageObject, PdfReader, Transformation

from analysis.tools import publication_manifest as pm

from ._common import FS_BODY, FS_NARR, init_figure_theme

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
DEFAULT_RENDER_ROOT = ROOT / "analysis" / "outputs" / "_figure_bundle"

A4_WIDTH_PT = 210.0 / 25.4 * 72.0
A4_HEIGHT_PT = 297.0 / 25.4 * 72.0
MIN_FIGURE_HEIGHT_PT = 180.0

TITLE_FS = FS_NARR
BODY_FS = FS_BODY
TOP_MARGIN_IN = 0.20
BOTTOM_MARGIN_IN = 0.18
SIDE_MARGIN_IN = 0.30
TITLE_GAP_IN = 0.08
TITLE_LEADING = 1.15
BODY_LEADING = 1.25


def _new_ephemeral_render_dir(prefix: str) -> Path:
    DEFAULT_RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = DEFAULT_RENDER_ROOT / f"{prefix}{stamp}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = DEFAULT_RENDER_ROOT / f"{prefix}{stamp}-{counter}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def load_figure_commands(manifest: dict, group: str) -> dict[str, str]:
    figures = manifest.get("figures", {}).get(group, {})
    if not isinstance(figures, dict) or not figures:
        raise ValueError(f"Manifest is missing figures.{group} commands.")
    return {str(key): str(value) for key, value in figures.items()}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _wrap_text_to_width(
    fig: plt.Figure,
    text: str,
    max_width_pt: float,
    *,
    fontsize: float,
    fontweight: str,
) -> str:
    words = text.split()
    if not words:
        return ""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_width_px = max_width_pt * fig.dpi / 72.0
    probe = fig.text(
        0.0,
        0.0,
        "",
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight=fontweight,
        alpha=0.0,
    )
    try:
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            probe.set_text(candidate)
            if probe.get_window_extent(renderer=renderer).width <= max_width_px:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines)
    finally:
        probe.remove()


def write_text_block_pdf(
    title: str,
    page_width_pt: float,
    output_path: Path,
    *,
    body: str | None = None,
    title_fs: float = TITLE_FS,
    body_fs: float = BODY_FS,
    side_margin_in: float = SIDE_MARGIN_IN,
    top_margin_in: float = TOP_MARGIN_IN,
    bottom_margin_in: float = BOTTOM_MARGIN_IN,
    title_gap_in: float = TITLE_GAP_IN,
) -> Path:
    page_width_in = page_width_pt / 72.0
    usable_width_pt = max(page_width_pt - 2 * side_margin_in * 72.0, 72.0)

    init_figure_theme()
    fig = plt.figure(figsize=(page_width_in, 1.0), facecolor="white")
    title_text = _wrap_text_to_width(
        fig,
        title,
        usable_width_pt,
        fontsize=title_fs,
        fontweight="bold",
    )
    body_text = None
    if body:
        body_text = _wrap_text_to_width(
            fig,
            " ".join(body.split()),
            usable_width_pt,
            fontsize=body_fs,
            fontweight="normal",
        )

    title_lines = title_text.count("\n") + 1
    title_height_in = title_lines * (title_fs / 72.0) * TITLE_LEADING

    body_height_in = 0.0
    gap_in = 0.0
    if body_text is not None:
        body_lines = body_text.count("\n") + 1
        body_height_in = body_lines * (body_fs / 72.0) * BODY_LEADING
        gap_in = title_gap_in

    page_height_in = (
        top_margin_in
        + title_height_in
        + gap_in
        + body_height_in
        + bottom_margin_in
    )

    fig.set_size_inches(page_width_in, page_height_in, forward=True)
    fig.clear()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    left = side_margin_in / page_width_in
    title_top = 1.0 - (top_margin_in / page_height_in)

    fig.text(
        left,
        title_top,
        title_text,
        ha="left",
        va="top",
        fontsize=title_fs,
        fontweight="bold",
    )
    if body_text is not None:
        body_top = title_top - ((title_height_in + title_gap_in) / page_height_in)
        fig.text(
            left,
            body_top,
            body_text,
            ha="left",
            va="top",
            fontsize=body_fs,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)
    return output_path


def fit_box(
    width_pt: float,
    height_pt: float,
    *,
    max_width_pt: float,
    max_height_pt: float,
    allow_scale_up: bool = False,
) -> tuple[float, float, float]:
    if width_pt <= 0 or height_pt <= 0:
        raise ValueError("Box dimensions must be positive.")
    scale = min(max_width_pt / width_pt, max_height_pt / height_pt)
    if not allow_scale_up:
        scale = min(scale, 1.0)
    return width_pt * scale, height_pt * scale, scale


def stack_pdf_pages(pdf_paths: Sequence[Path]) -> PageObject:
    if not pdf_paths:
        raise ValueError("At least one PDF page is required.")

    page_infos: list[tuple[object, float, float]] = []
    for pdf_path in pdf_paths:
        pdf_page = PdfReader(str(pdf_path)).pages[0]
        page_infos.append(
            (
                pdf_page,
                float(pdf_page.mediabox.width),
                float(pdf_page.mediabox.height),
            )
        )

    page_width = max(width for _, width, _ in page_infos)
    page_height = sum(height for _, _, height in page_infos)

    page = PageObject.create_blank_page(width=page_width, height=page_height)
    current_y = 0.0
    for pdf_page, width, height in reversed(page_infos):
        page.merge_transformed_page(
            pdf_page,
            Transformation().translate(tx=(page_width - width) / 2.0, ty=current_y),
        )
        current_y += height
    return page


def compose_fixed_page(
    figure_pdf: Path,
    text_block_pdf: Path,
    *,
    page_width_pt: float = A4_WIDTH_PT,
    page_height_pt: float = A4_HEIGHT_PT,
    min_figure_height_pt: float = MIN_FIGURE_HEIGHT_PT,
    side_margin_pt: float = 0.0,
    top_margin_pt: float = 0.0,
    bottom_margin_pt: float = 0.0,
    allow_scale_up: bool = False,
) -> PageObject:
    figure_page = PdfReader(str(figure_pdf)).pages[0]
    text_page = PdfReader(str(text_block_pdf)).pages[0]

    figure_left = float(figure_page.mediabox.left)
    figure_bottom = float(figure_page.mediabox.bottom)
    if figure_left != 0.0 or figure_bottom != 0.0:
        raise ValueError(
            f"{figure_pdf} has non-zero lower-left mediabox ({figure_left}, {figure_bottom})."
        )

    figure_width_pt = float(figure_page.mediabox.width)
    figure_height_pt = float(figure_page.mediabox.height)
    text_width_pt = float(text_page.mediabox.width)
    text_height_pt = float(text_page.mediabox.height)

    content_width_pt = page_width_pt - 2 * side_margin_pt
    content_height_pt = page_height_pt - top_margin_pt - bottom_margin_pt
    if content_width_pt <= 0 or content_height_pt <= 0:
        raise ValueError("Page margins leave no usable content area.")

    available_figure_height_pt = content_height_pt - text_height_pt
    if available_figure_height_pt < min_figure_height_pt:
        raise ValueError(
            f"Legend/text block is too tall ({text_height_pt:.1f} pt); only "
            f"{available_figure_height_pt:.1f} pt remain for the figure."
        )

    scaled_width_pt, scaled_height_pt, scale = fit_box(
        figure_width_pt,
        figure_height_pt,
        max_width_pt=content_width_pt,
        max_height_pt=available_figure_height_pt,
        allow_scale_up=allow_scale_up,
    )

    page = PageObject.create_blank_page(width=page_width_pt, height=page_height_pt)
    figure_y = page_height_pt - top_margin_pt - scaled_height_pt
    text_y = figure_y - text_height_pt
    if text_y < bottom_margin_pt:
        raise ValueError(
            f"{text_block_pdf} leaves negative text placement ({text_y:.1f} pt)."
        )

    page.merge_transformed_page(
        figure_page,
        Transformation()
        .scale(scale, scale)
        .translate(
            tx=side_margin_pt + (content_width_pt - scaled_width_pt) / 2.0,
            ty=figure_y,
        ),
    )
    page.merge_transformed_page(
        text_page,
        Transformation().translate(
            tx=side_margin_pt + (content_width_pt - text_width_pt) / 2.0,
            ty=text_y,
        ),
    )
    return page


def compose_stacked_page(
    blocks: Sequence[tuple[Path, Path]],
    *,
    page_width_pt: float = A4_WIDTH_PT,
    page_height_pt: float = A4_HEIGHT_PT,
    gap_pt: float = 10.0,
    side_margin_pt: float = 0.0,
    top_margin_pt: float = 0.0,
    bottom_margin_pt: float = 0.0,
    allow_scale_up: bool = False,
) -> PageObject:
    if not blocks:
        raise ValueError("At least one figure/text block is required.")
    if len(blocks) == 1:
        figure_pdf, text_block_pdf = blocks[0]
        return compose_fixed_page(
            figure_pdf,
            text_block_pdf,
            page_width_pt=page_width_pt,
            page_height_pt=page_height_pt,
            side_margin_pt=side_margin_pt,
            top_margin_pt=top_margin_pt,
            bottom_margin_pt=bottom_margin_pt,
            allow_scale_up=allow_scale_up,
        )

    infos: list[tuple[object, object, float, float, float, float, float]] = []
    content_width_pt = page_width_pt - 2 * side_margin_pt
    content_height_pt = page_height_pt - top_margin_pt - bottom_margin_pt
    if content_width_pt <= 0 or content_height_pt <= 0:
        raise ValueError("Page margins leave no usable content area.")

    total_height_pt = gap_pt * (len(blocks) - 1)
    for figure_pdf, text_block_pdf in blocks:
        figure_page = PdfReader(str(figure_pdf)).pages[0]
        text_page = PdfReader(str(text_block_pdf)).pages[0]

        figure_left = float(figure_page.mediabox.left)
        figure_bottom = float(figure_page.mediabox.bottom)
        if figure_left != 0.0 or figure_bottom != 0.0:
            raise ValueError(
                f"{figure_pdf} has non-zero lower-left mediabox ({figure_left}, {figure_bottom})."
            )

        figure_width_pt = float(figure_page.mediabox.width)
        figure_height_pt = float(figure_page.mediabox.height)
        text_width_pt = float(text_page.mediabox.width)
        text_height_pt = float(text_page.mediabox.height)

        scaled_width_pt, scaled_height_pt, scale = fit_box(
            figure_width_pt,
            figure_height_pt,
            max_width_pt=content_width_pt,
            max_height_pt=content_height_pt,
            allow_scale_up=allow_scale_up,
        )
        total_height_pt += scaled_height_pt + text_height_pt
        infos.append(
            (
                figure_page,
                text_page,
                text_width_pt,
                text_height_pt,
                scaled_width_pt,
                scaled_height_pt,
                scale,
            )
        )

    if total_height_pt > content_height_pt:
        raise ValueError(
            f"Stacked page overflows A4 content height ({total_height_pt:.1f} pt > {content_height_pt:.1f} pt)."
        )

    page = PageObject.create_blank_page(width=page_width_pt, height=page_height_pt)
    current_top = page_height_pt - top_margin_pt
    for index, (
        figure_page,
        text_page,
        text_width_pt,
        text_height_pt,
        scaled_width_pt,
        scaled_height_pt,
        scale,
    ) in enumerate(infos):
        current_top -= scaled_height_pt
        page.merge_transformed_page(
            figure_page,
            Transformation()
            .scale(scale, scale)
            .translate(
                tx=side_margin_pt + (content_width_pt - scaled_width_pt) / 2.0,
                ty=current_top,
            ),
        )
        current_top -= text_height_pt
        page.merge_transformed_page(
            text_page,
            Transformation().translate(
                tx=side_margin_pt + (content_width_pt - text_width_pt) / 2.0,
                ty=current_top,
            ),
        )
        if index < len(infos) - 1:
            current_top -= gap_pt
    return page


def render_figure_pdf(key: str, command: str, render_root: Path) -> Path:
    figure_dir = render_root / key
    if figure_dir.exists():
        shutil.rmtree(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"RENDER {key}\t{display_path(figure_dir)}")
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise RuntimeError(f"{key}: invalid command ({exc})") from exc
    completed = subprocess.run(
        parts,
        cwd=ROOT,
        env=pm._figure_environment(figure_dir, ("pdf",)),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{key} failed with exit code {completed.returncode}")

    pdfs = sorted(figure_dir.glob("*.pdf"))
    if len(pdfs) != 1:
        raise RuntimeError(
            f"{key} produced {len(pdfs)} PDFs in {figure_dir}; expected exactly one."
        )
    return pdfs[0]


@contextlib.contextmanager
def figure_render_root(
    render_dir: Path | None,
    *,
    prefix: str,
) -> Iterator[Path]:
    if render_dir is None:
        scratch_dir = _new_ephemeral_render_dir(prefix)
        try:
            yield scratch_dir
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return

    render_dir.mkdir(parents=True, exist_ok=True)
    yield render_dir
