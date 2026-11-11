"""The two README blocks generated from the manifest: the ordered stage table and the figure/table map.

``readme --write`` rewrites them between their marker comments; ``readme_problems`` (run by ``check``)
reports a block that is missing or differs from what the manifest generates.
"""

from __future__ import annotations

import argparse

from analysis.output_layout import ROOT
from analysis.tools.manifest_stages import (
    _relative_display,
    load_manifest,
    registered_figures,
    registered_tables,
)


README = ROOT / "README.md"


# ---------------------------------------------------------------------------
# README blocks generated from the manifest
# ---------------------------------------------------------------------------

STAGES_BEGIN = "<!-- stages:begin -->"


STAGES_END = "<!-- stages:end -->"


DISPLAY_ITEMS_BEGIN = "<!-- display-items:begin -->"


DISPLAY_ITEMS_END = "<!-- display-items:end -->"


FIGURE_LABEL = {
    "main": "Fig.",
    "extended": "Extended Data Fig.",
    "supplementary": "Supplementary Fig.",
}


def _markdown_block(begin: str, end: str, header: list[str], rows: list[list[str]]) -> str:
    """A markdown table between two marker comments, ending in a newline."""
    lines = [begin, "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append(end)
    return "\n".join(lines) + "\n"


def stages_block(manifest: dict) -> str:
    """The README's ordered stage table: one row per stage, then the figure and table builds.

    Args:
        manifest: The loaded manifest.

    Returns:
        The block, markers included, ending in a newline.
    """
    rows = [[f"`{name}`", spec["purpose"]] for name, spec in manifest["analysis_outputs"].items()]
    rows.append(["`figures`", manifest["figure_bundles"]["all_figures"]["purpose"]])
    rows.append(["`tables`", manifest["tables"]["supplementary_workbook"]["purpose"]])
    return _markdown_block(STAGES_BEGIN, STAGES_END, ["Stage", "What it does"], rows)


def display_items_block(manifest: dict) -> str:
    """The README's map from each figure and supplementary table to the code that builds it.

    Args:
        manifest: The loaded manifest.

    Returns:
        The block, markers included, ending in a newline; paper order (main, Extended Data,
        supplementary figures, then Tables S1 onwards).
    """
    items = [
        (f"{FIGURE_LABEL[group]} {i}", f"figures/{script}")
        for group, entries in registered_figures().items()
        for i, (script, _) in enumerate(entries, 1)
    ]
    items += [
        (f"{tab} ({description})", f"tables/{module}.py")
        for tab, description, module, _ in registered_tables()
    ]
    rows = [[item, f"[`{path}`]({path})"] for item, path in items]
    return _markdown_block(DISPLAY_ITEMS_BEGIN, DISPLAY_ITEMS_END, ["Item", "Built by"], rows)


_README_BLOCKS = (
    ("stages", STAGES_BEGIN, STAGES_END, stages_block),
    ("display-items", DISPLAY_ITEMS_BEGIN, DISPLAY_ITEMS_END, display_items_block),
)


def _block_span(text: str, begin: str, end: str) -> tuple[int, int] | None:
    start = text.find(begin)
    stop = text.find(end, start)
    if start < 0 or stop < 0:
        return None
    stop += len(end)
    if text[stop : stop + 1] == "\n":
        stop += 1
    return start, stop


def readme_problems(manifest: dict) -> list[str]:
    """README blocks that are missing or differ from what the manifest generates."""
    text = README.read_text()
    problems = []
    for name, begin, end, build in _README_BLOCKS:
        span = _block_span(text, begin, end)
        if span is None:
            problems.append(f"README.md has no {begin} ... {end} block")
        elif text[span[0] : span[1]] != build(manifest):
            problems.append(
                f"README.md {name} block is stale; run "
                "`pixi run python analysis/tools/publication_manifest.py readme --write`"
            )
    return problems


def write_readme_blocks(manifest: dict) -> None:
    """Rewrite the README blocks from the manifest, leaving the rest of the file as it is.

    Raises:
        ValueError: When a block's markers are missing.
    """
    text = README.read_text()
    for _, begin, end, build in _README_BLOCKS:
        span = _block_span(text, begin, end)
        if span is None:
            raise ValueError(f"README.md has no {begin} ... {end} block to rewrite")
        text = text[: span[0]] + build(manifest) + text[span[1] :]
    README.write_text(text)


def cmd_readme(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if args.write:
        write_readme_blocks(manifest)
        print(f"Wrote the stage and display-item blocks of {_relative_display(README)}")
        return 0
    problems = readme_problems(manifest)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("README blocks match the manifest")
    return 1 if problems else 0
