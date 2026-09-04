"""Build a combined main-figure PDF from the public figure wrappers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.tools import publication_manifest as pm  # noqa: E402
from figures._internal import _figure_bundle as bundle  # noqa: E402

FIGURES_DIR = bundle.FIGURES_DIR
DEFAULT_OUTPUT = FIGURES_DIR / "main_figures.pdf"


@dataclass(frozen=True)
class MainFigureSpec:
    key: str
    number: int

    @property
    def full_title(self) -> str:
        return f"Figure {self.number}"


def load_main_figure_specs(manifest: dict | None = None) -> list[MainFigureSpec]:
    if manifest is None:
        manifest = pm.load_manifest()
    commands = bundle.load_figure_commands(manifest, "main")
    return [
        MainFigureSpec(key=key, number=i)
        for i, key in enumerate(commands, start=1)
    ]


def _print_available_specs(specs: Sequence[MainFigureSpec]) -> None:
    for spec in specs:
        print(f"{spec.key}\t{spec.full_title}")


def _select_specs(
    specs: Sequence[MainFigureSpec], requested: Sequence[str] | None
) -> list[MainFigureSpec]:
    if not requested:
        return list(specs)
    requested_set = set(requested)
    known = {spec.key for spec in specs}
    missing = sorted(requested_set - known)
    if missing:
        raise ValueError(f"Unknown main figure key(s): {', '.join(missing)}")
    return [spec for spec in specs if spec.key in requested_set]


def build_main_figures(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    requested: Sequence[str] | None = None,
    render_dir: Path | None = None,
) -> Path:
    manifest = pm.load_manifest()
    specs = load_main_figure_specs(manifest)
    commands = bundle.load_figure_commands(manifest, "main")

    selected = _select_specs(specs, requested)
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Main Figures"})

    with bundle.figure_render_root(render_dir, prefix="main-build-") as render_root:
        for spec in selected:
            figure_pdf = bundle.render_figure_pdf(spec.key, commands[spec.key], render_root)
            figure_width_pt = float(PdfReader(str(figure_pdf)).pages[0].mediabox.width)
            label_pdf = bundle.write_text_block_pdf(
                spec.full_title,
                figure_width_pt,
                render_root / spec.key / f"{spec.key}_label.pdf",
            )
            writer.add_page(bundle.stack_pdf_pages([label_pdf, figure_pdf]))

    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    print(f"Wrote {bundle.display_path(output_path)}")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render main figures and assemble a single combined PDF."
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
        help="Main figure key to include (repeatable). Use --list-figures to inspect valid keys.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {bundle.display_path(DEFAULT_OUTPUT)})",
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
        _print_available_specs(load_main_figure_specs())
        return 0
    build_main_figures(
        args.output,
        requested=args.figures,
        render_dir=args.render_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
