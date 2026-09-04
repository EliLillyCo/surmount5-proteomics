"""Build a combined Supplementary Figures PDF from the public figure wrappers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.tools import publication_manifest as pm  # noqa: E402
from figures._internal import _figure_bundle as bundle  # noqa: E402

FIGURES_DIR = bundle.FIGURES_DIR
LEGEND_CONFIG = FIGURES_DIR / "configs" / "supplementary_legends.yaml"
DEFAULT_OUTPUT = FIGURES_DIR / "supplementary_figures.pdf"
SUPP_PAGE_MARGIN_IN = 0.5
SUPP_TITLE_FS = 9.0
SUPP_BODY_FS = 8.0
SUPP_CAPTION_SIDE_MARGIN_IN = 0.0
SUPP_CAPTION_TOP_MARGIN_IN = 24.0 / 72.0
SUPP_CAPTION_BOTTOM_MARGIN_IN = 0.12
SUPP_TITLE_GAP_IN = 0.05
SUPP_PAGE_GAP_PT = 64.0
SUPP_MAX_ITEMS_PER_PAGE = 2


@dataclass(frozen=True)
class SupplementSpec:
    key: str
    number: int
    title: str
    legend: str

    @property
    def full_title(self) -> str:
        return f"Supplementary Figure {self.number}: {self.title}"


@dataclass(frozen=True)
class SupplementBlock:
    spec: SupplementSpec
    figure_pdf: Path
    caption_pdf: Path
    packed_height_pt: float


def _supp_page_margin_pt() -> float:
    return SUPP_PAGE_MARGIN_IN * 72.0


def _supp_content_width_pt() -> float:
    return bundle.A4_WIDTH_PT - 2 * _supp_page_margin_pt()


def _supp_content_height_pt() -> float:
    return bundle.A4_HEIGHT_PT - 2 * _supp_page_margin_pt()


def load_supplement_specs(path: Path = LEGEND_CONFIG) -> list[SupplementSpec]:
    payload = yaml.safe_load(path.read_text())
    items = payload.get("figures") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"Malformed supplementary legend config: {path}")

    specs: list[SupplementSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed supplementary legend entry: {item!r}")
        specs.append(
            SupplementSpec(
                key=str(item["key"]),
                number=int(item["number"]),
                title=str(item["title"]),
                legend=" ".join(str(item["legend"]).split()),
            )
        )

    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("Supplementary legend config contains duplicate figure keys.")
    return specs


def _print_available_specs(specs: Sequence[SupplementSpec]) -> None:
    for spec in specs:
        print(f"{spec.key}\t{spec.full_title}")


def _select_specs(
    specs: Sequence[SupplementSpec], requested: Sequence[str] | None
) -> list[SupplementSpec]:
    if not requested:
        return list(specs)
    requested_set = set(requested)
    known = {spec.key for spec in specs}
    missing = sorted(requested_set - known)
    if missing:
        raise ValueError(
            f"Unknown supplementary figure key(s): {', '.join(missing)}"
        )
    return [spec for spec in specs if spec.key in requested_set]


def _block_height_pt(figure_pdf: Path, caption_pdf: Path) -> float:
    figure_page = PdfReader(str(figure_pdf)).pages[0]
    caption_page = PdfReader(str(caption_pdf)).pages[0]
    figure_width_pt = float(figure_page.mediabox.width)
    figure_height_pt = float(figure_page.mediabox.height)
    caption_height_pt = float(caption_page.mediabox.height)
    _, scaled_figure_height_pt, _ = bundle.fit_box(
        figure_width_pt,
        figure_height_pt,
        max_width_pt=_supp_content_width_pt(),
        max_height_pt=_supp_content_height_pt(),
        allow_scale_up=False,
    )
    return scaled_figure_height_pt + caption_height_pt


def _paginate_blocks(blocks: Sequence[SupplementBlock]) -> list[list[SupplementBlock]]:
    pages: list[list[SupplementBlock]] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if i + 1 < len(blocks) and SUPP_MAX_ITEMS_PER_PAGE >= 2:
            nxt = blocks[i + 1]
            combined_height = (
                current.packed_height_pt + nxt.packed_height_pt + SUPP_PAGE_GAP_PT
            )
            if combined_height <= _supp_content_height_pt():
                pages.append([current, nxt])
                i += 2
                continue
        pages.append([current])
        i += 1
    return pages


def build_supplementary_figures(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    requested: Sequence[str] | None = None,
    render_dir: Path | None = None,
) -> Path:
    manifest = pm.load_manifest()
    specs = load_supplement_specs()
    commands = bundle.load_figure_commands(manifest, "supplementary")

    missing_commands = sorted({spec.key for spec in specs} - set(commands))
    if missing_commands:
        raise ValueError(
            "Supplementary legends are missing manifest commands for: "
            + ", ".join(missing_commands)
        )

    selected = _select_specs(specs, requested)
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Supplementary Figures"})

    with bundle.figure_render_root(
        render_dir,
        prefix="supplementary-build-",
    ) as render_root:
        rendered_blocks: list[SupplementBlock] = []
        for spec in selected:
            figure_pdf = bundle.render_figure_pdf(
                spec.key,
                commands[spec.key],
                render_root,
            )
            caption_pdf = bundle.write_text_block_pdf(
                spec.full_title,
                _supp_content_width_pt(),
                render_root / spec.key / f"{spec.key}_legend.pdf",
                body=spec.legend,
                title_fs=SUPP_TITLE_FS,
                body_fs=SUPP_BODY_FS,
                side_margin_in=SUPP_CAPTION_SIDE_MARGIN_IN,
                top_margin_in=SUPP_CAPTION_TOP_MARGIN_IN,
                bottom_margin_in=SUPP_CAPTION_BOTTOM_MARGIN_IN,
                title_gap_in=SUPP_TITLE_GAP_IN,
            )
            rendered_blocks.append(
                SupplementBlock(
                    spec=spec,
                    figure_pdf=figure_pdf,
                    caption_pdf=caption_pdf,
                    packed_height_pt=_block_height_pt(figure_pdf, caption_pdf),
                )
            )

        for page_blocks in _paginate_blocks(rendered_blocks):
            writer.add_page(
                bundle.compose_stacked_page(
                    [(block.figure_pdf, block.caption_pdf) for block in page_blocks],
                    gap_pt=SUPP_PAGE_GAP_PT,
                    side_margin_pt=_supp_page_margin_pt(),
                    top_margin_pt=_supp_page_margin_pt(),
                    bottom_margin_pt=_supp_page_margin_pt(),
                )
            )

    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    print(f"Wrote {bundle.display_path(output_path)}")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render supplementary figures and assemble a single Supplementary Figures PDF."
    )
    parser.add_argument(
        "--list-figures",
        action="store_true",
        help="List available manifest figure keys and exit.",
    )
    parser.add_argument(
        "--figure",
        action="append",
        dest="figures",
        help=(
            "Supplementary figure key to include (repeatable). "
            "Use --list-figures to inspect valid keys."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help="Optional directory where intermediate per-figure renders should be kept.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_figures:
        _print_available_specs(load_supplement_specs())
        return 0
    build_supplementary_figures(
        args.output,
        requested=args.figures,
        render_dir=args.render_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
