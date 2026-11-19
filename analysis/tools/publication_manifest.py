"""Command line for the manuscript reproduction manifest (analysis/manifest.yaml).

    pixi run python analysis/tools/publication_manifest.py list
    pixi run python analysis/tools/publication_manifest.py check
    pixi run python analysis/tools/publication_manifest.py readme [--write]
    pixi run python analysis/tools/publication_manifest.py audit-figures

Each subcommand dispatches to the module that does the job: ``check`` to
``analysis.tools.manifest_check`` (commands, path templates, pixi tasks, the figure registry, the
stage graph and the entry-point imports), ``readme`` to ``analysis.tools.readme_blocks``,
``audit-figures`` to ``analysis.tools.figure_audit``, and ``render-figures``,
``record-figure-baseline`` and ``compare-figure-baseline`` to ``analysis.tools.figure_baseline``.
``list`` and ``run`` read the manifest through ``analysis.tools.manifest_stages``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from analysis.output_layout import ROOT
from analysis.tools.figure_audit import cmd_audit_figures
from analysis.tools.figure_baseline import _figure_environment as _figure_environment
from analysis.tools.figure_baseline import (
    cmd_compare_figure_baseline,
    cmd_record_figure_baseline,
    cmd_render_figures,
)
from analysis.tools.manifest_check import cmd_check
from analysis.tools.manifest_stages import iter_commands, load_manifest
from analysis.tools.manifest_stages import require_stage_outputs as require_stage_outputs
from analysis.tools.readme_blocks import cmd_readme


def cmd_list(_: argparse.Namespace) -> int:
    for name, command in iter_commands(load_manifest()):
        print(f"{name}\t{command}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    commands = dict(iter_commands(load_manifest()))
    if args.name not in commands:
        print(f"Unknown command: {args.name}")
        return 1
    return subprocess.call(commands[args.name], shell=True, cwd=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List manifest commands")
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser(
        "check",
        help="Validate manifest scripts, pixi tasks, path templates, and figure configs",
    )
    p_check.set_defaults(func=cmd_check)

    p_readme = sub.add_parser(
        "readme",
        help="Check (or with --write, regenerate) the README stage and display-item blocks",
    )
    p_readme.add_argument("--write", action="store_true", help="Rewrite the blocks in README.md")
    p_readme.set_defaults(func=cmd_readme)

    p_audit = sub.add_parser("audit-figures", help="Run static figure-style audit")
    p_audit.set_defaults(func=cmd_audit_figures)

    p_run = sub.add_parser("run", help="Run one manifest command by name")
    p_run.add_argument("name")
    p_run.set_defaults(func=cmd_run)

    p_render = sub.add_parser(
        "render-figures",
        help="Render manifest figures into per-figure output directories",
    )
    p_render.add_argument(
        "--name",
        action="append",
        help="Manifest figure name to render (for example main:fig4_mediation).",
    )
    p_render.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Directory to keep rendered artifacts. Defaults to analysis/outputs/"
            "_publication_manifest/<timestamp>."
        ),
    )
    p_render.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats passed through the figure save helper.",
    )
    p_render.set_defaults(func=cmd_render_figures)

    p_record = sub.add_parser(
        "record-figure-baseline",
        help="Render figures and write a PNG-hash baseline manifest",
    )
    p_record.add_argument(
        "--name",
        action="append",
        help="Manifest figure name to include. Defaults to all figures.",
    )
    p_record.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Where to write the baseline JSON.",
    )
    p_record.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Directory to keep rendered baseline artifacts. Defaults to analysis/outputs/"
            "_publication_manifest/<timestamp>."
        ),
    )
    p_record.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats to render while recording the baseline.",
    )
    p_record.set_defaults(func=cmd_record_figure_baseline)

    p_compare = sub.add_parser(
        "compare-figure-baseline",
        help="Rerender figures and compare them to a recorded baseline",
    )
    p_compare.add_argument(
        "--name",
        action="append",
        help="Manifest figure name to compare. Defaults to every figure in the baseline.",
    )
    p_compare.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline JSON to compare against.",
    )
    p_compare.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Directory to keep candidate artifacts. Defaults to analysis/outputs/"
            "_publication_manifest/<timestamp>."
        ),
    )
    p_compare.add_argument(
        "--formats",
        default="pdf,png",
        help="Fallback formats when the baseline does not declare them.",
    )
    p_compare.set_defaults(func=cmd_compare_figure_baseline)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
