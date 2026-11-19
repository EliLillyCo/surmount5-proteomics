"""The reproduction manifest (``analysis/manifest.yaml``): loading it, the figure and table registries, the stage graph.

Shared by the manifest check, the README generator, the figure-baseline commands, the stage runner and
the provenance records. ``require_stage_outputs`` is the pre-flight of the figure and table builds: it
names the stage that writes each missing ``analysis/outputs/`` file.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Iterable

import yaml

from analysis.output_layout import ANALYSIS_OUTPUTS, ROOT


MANIFEST = ROOT / "analysis" / "manifest.yaml"


FIGURES_DIR = ROOT / "figures"


BUILD_FIGURES = FIGURES_DIR / "build_figures.py"


BUILD_TABLES = ROOT / "tables" / "build_all.py"


_REGISTRY_ENTRY = re.compile(r'^\s*\("([^"]+\.py)",\s*\[([^\]]*)\],\s*"([^"]+)"\),', re.M)


def load_manifest(path: Path = MANIFEST) -> dict:
    """Load the manuscript manifest."""
    with path.open() as f:
        return yaml.safe_load(f)


def registered_figures(path: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """``(script, stem)`` per registry in ``figures/build_figures.py``, read without importing it.

    Returns:
        ``{"main" | "extended" | "supplementary": [(script, stem), ...]}`` in registry order.
    """
    text = (BUILD_FIGURES if path is None else path).read_text()
    out: dict[str, list[tuple[str, str]]] = {}
    for group, name in (("main", "MAIN"), ("extended", "EXTENDED"), ("supplementary", "SUPP")):
        start = text.index(f"\n{name}: list[Entry] = [")
        end = text.index("\n]\n", start)
        out[group] = [(script, stem) for script, _, stem in _REGISTRY_ENTRY.findall(text[start:end])]
    return out


def registered_tables(path: Path | None = None) -> list[tuple[str, str, str, int]]:
    """``tables/build_all.py`` ``MANIFEST`` rows, read without importing it.

    Returns:
        ``(tab, description, module, sheet index)`` per workbook sheet, in sheet order.
    """
    tree = ast.parse((BUILD_TABLES if path is None else path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MANIFEST" for t in node.targets
        ):
            return [tuple(row) for row in ast.literal_eval(node.value)]
    raise ValueError(f"No MANIFEST list in {BUILD_TABLES}")


def _stage_scripts(spec: dict) -> set[str]:
    """Repo paths a stage's command runs: its ``.py`` script, or the ``-m`` package and module."""
    parts = shlex.split(spec.get("command", ""))
    found = {p for p in parts if p.endswith(".py")}
    if "-m" in parts and parts.index("-m") + 1 < len(parts):
        rel = parts[parts.index("-m") + 1].replace(".", "/")
        found |= {f"{rel}.py", rel, f"{rel}/", f"{rel}/__main__.py"}
    return found


def stage_consumers(manifest: dict) -> dict[str, list[str]]:
    """Stages each stage needs: the ones whose ``downstream`` names its script, in manifest order."""
    stages = manifest.get("analysis_outputs", {})
    return {
        name: [
            other
            for other, spec_other in stages.items()
            if other != name and _stage_scripts(spec) & set(spec_other.get("downstream", []))
        ]
        for name, spec in stages.items()
    }


def stage_order_problems(manifest: dict) -> list[str]:
    """A stage listed before a stage whose ``downstream`` names its script.

    Returns:
        One message per out-of-order pair; empty when the manifest order runs.
    """
    order = list(manifest.get("analysis_outputs", {}))
    return [
        f"stage {name} (#{order.index(name) + 1}) is listed before stage {needed} "
        f"(#{order.index(needed) + 1}), whose downstream names its script; move it after"
        for name, needs in stage_consumers(manifest).items()
        for needed in needs
        if order.index(needed) > order.index(name)
    ]


def _listed_downstream(spec: dict, consumer: str) -> bool:
    downstream = spec.get("downstream", [])
    return consumer in downstream or consumer.split("/", 1)[0] in downstream


def iter_figure_commands(manifest: dict) -> Iterable[tuple[str, str]]:
    """Yield manifest figure commands."""
    for group, figures in manifest.get("figures", {}).items():
        for name, command in figures.items():
            yield f"{group}:{name}", command


def iter_commands(manifest: dict) -> Iterable[tuple[str, str]]:
    """Yield named commands from the manifest."""
    for section in ("analysis_outputs",):
        for name, spec in manifest.get(section, {}).items():
            command = spec.get("command")
            if command:
                yield name, command
            for i, cmd in enumerate(spec.get("commands", []), start=1):
                yield f"{name}:{i}", cmd
    for name, spec in manifest.get("validation", {}).items():
        command = spec.get("command")
        if command:
            yield f"validation:{name}", command
        for i, cmd in enumerate(spec.get("commands", []), start=1):
            yield f"validation:{name}:{i}", cmd
    for module_name, spec in manifest.get("modules", {}).items():
        if not isinstance(spec, dict):
            continue
        command = spec.get("entrypoint")
        if command:
            yield f"modules:{module_name}:entrypoint", command
        for key in ("entrypoints", "study_wrappers", "r_tasks", "pixi_tasks"):
            for i, cmd in enumerate(spec.get(key, []), start=1):
                yield f"modules:{module_name}:{key}:{i}", cmd
    yield from iter_figure_commands(manifest)
    for name, spec in manifest.get("figure_bundles", {}).items():
        command = spec.get("command")
        if command:
            yield f"figure_bundles:{name}", command
    table = manifest.get("tables", {}).get("supplementary_workbook", {})
    if table.get("command"):
        yield "tables:supplementary_workbook", table["command"]


ANALYSIS_OUTPUTS_PREFIX = f"{ANALYSIS_OUTPUTS.relative_to(ROOT).as_posix()}/"


def feeding_stages(manifest: dict, kind: str, scripts: Iterable[str] = ()) -> list[tuple[str, dict]]:
    """Stages whose outputs a figure or table build reads, from the manifest alone.

    A stage's ``downstream`` names its consumers: the bare word ``figures`` or ``tables``, or
    repo paths. A stage feeds the build when its downstream holds the bare word, any path under
    ``<kind>/`` (``tables``) or one of ``scripts`` (``figures``, given as ``figures/<script>``).

    Args:
        manifest: The loaded manifest.
        kind: ``"figures"`` or ``"tables"``.
        scripts: For ``figures``, the ``figures/<script>`` paths being built; empty means any.

    Returns:
        ``(stage name, stage spec)`` in manifest order.
    """
    wanted = set(scripts)
    found: list[tuple[str, dict]] = []
    for stage, spec in manifest.get("analysis_outputs", {}).items():
        downstream = spec.get("downstream", [])
        feeds = kind in downstream or any(
            d.startswith(f"{kind}/") and (kind == "tables" or not wanted or d in wanted)
            for d in downstream
        )
        if feeds:
            found.append((stage, spec))
    return found


def stage_outputs_for(manifest: dict, kind: str, scripts: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    """Repo-relative stage outputs a figure or table build reads, from the manifest alone.

    Args:
        manifest: The loaded manifest.
        kind: ``"figures"`` or ``"tables"``.
        scripts: For ``figures``, the ``figures/<script>`` paths being built; empty means any.

    Returns:
        ``(output path, stage name, stage command)`` for each ``analysis/outputs/`` file of a
        stage that ``feeding_stages`` returns.
    """
    return [
        (out, stage, spec["command"])
        for stage, spec in feeding_stages(manifest, kind, scripts)
        for out in spec.get("outputs", [])
        if out.startswith(ANALYSIS_OUTPUTS_PREFIX)
    ]


def require_stage_outputs(kind: str, scripts: Iterable[str] = ()) -> None:
    """Fail before a build when a manifest stage output it reads is absent.

    Raises:
        FileNotFoundError: Naming each missing file with the manifest stage and command that writes it.
    """
    missing = [
        f"  {out}  <- stage {stage}: {command}"
        for out, stage, command in stage_outputs_for(load_manifest(), kind, scripts)
        if not (ROOT / out).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"The {kind} build needs outputs that are not there; run the stage that writes each "
            "(analysis/manifest.yaml analysis_outputs):\n" + "\n".join(missing)
        )


def _relative_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
