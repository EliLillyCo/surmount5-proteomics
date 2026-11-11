"""The manifest check (``check``): commands, path templates, pixi tasks, the figure registry and the stage graph.

``check`` also validates the stage graph without data: every stage comes after the stages whose
``downstream`` names its script; a registered figure or table that reads a stage's output (found
statically, ``analysis.tools.static_reads``) is listed in that stage's ``downstream``, and one that is
listed reads it; the two README blocks equal what ``readme --write`` generates; and every manifest entry
point imports under the environment's ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import yaml

from analysis.output_layout import ROOT
from analysis.tools.manifest_stages import (
    _relative_display,
    iter_commands,
    load_manifest,
    registered_figures,
    stage_order_problems,
)
from analysis.tools.readme_blocks import readme_problems
from analysis.tools.static_reads import downstream_problems


PIXI_FILE = ROOT / "pixi.toml"


PATHS_TEMPLATE = ROOT / "paths.example.yaml"


LOCAL_PATHS_FILE = ROOT / "paths.yaml"


# Maximum number of Extended Data items.
MAX_EXTENDED_DATA_FIGURES = 10


PATHS_ENV_VAR = "SURMOUNT5_PATHS_YAML"


CORE_PATH_KEYS = ("mmrm_root", "qc_root", "pheno_root")


_PATHS_REF_RE = re.compile(r"paths\.yaml:([A-Za-z0-9_]+)")


# `pixi run python ...` runs the environment's interpreter, not a task, so it is not a task reference.
_PIXI_RUN_RE = re.compile(r"\bpixi\s+run\s+(?!python\b)([A-Za-z0-9][A-Za-z0-9_-]*)\b")


def _iter_manifest_strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_manifest_strings(value)
        return
    if isinstance(node, list):
        for value in node:
            yield from _iter_manifest_strings(value)


def _collect_paths_keys(manifest: dict) -> list[str]:
    keys = {
        match.group(1)
        for text in _iter_manifest_strings(manifest)
        for match in _PATHS_REF_RE.finditer(text)
    }
    return sorted(keys)


def _collect_manifest_pixi_tasks(manifest: dict) -> list[str]:
    tasks = {
        match.group(1)
        for text in _iter_manifest_strings(manifest)
        for match in _PIXI_RUN_RE.finditer(text)
    }
    return sorted(tasks)


def _load_yaml_mapping(path: Path) -> dict:
    with path.open() as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _resolve_local_paths_file() -> Path:
    override = os.environ.get(PATHS_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return LOCAL_PATHS_FILE


def _load_pixi_tasks(path: Path | None = None) -> set[str]:
    path = PIXI_FILE if path is None else path
    payload = tomllib.loads(path.read_text())
    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"Expected [tasks] table in {path}")
    return {str(name) for name in tasks}


def _check_stage_graph(manifest: dict) -> list[str]:
    """Stage order, downstream lists against the code, and the README blocks."""
    return stage_order_problems(manifest) + downstream_problems(manifest) + readme_problems(manifest)


def _check_code_roots(manifest: dict, root: Path = ROOT) -> list[str]:
    """Stages whose ``code_roots`` name a file that does not exist.

    A misspelt root would never change, so provenance-check would stop seeing edits to the child
    script it was meant to guard.
    """
    return [
        f"{stage}: code_roots path {path} does not exist"
        for stage, spec in manifest.get("analysis_outputs", {}).items()
        for path in spec.get("code_roots", [])
        if not (root / path).is_file()
    ]


def _check_figure_registry(manifest: dict) -> list[str]:
    """Problems between the figure registry and the manifest figure section."""
    problems: list[str] = []
    registry = registered_figures()
    figures = manifest.get("figures", {})
    for group, entries in registry.items():
        listed = figures.get(group, {})
        stems = [stem for _, stem in entries]
        if list(listed) != stems:
            problems.append(
                f"analysis/manifest.yaml figures.{group} differs from build_figures.py: "
                f"manifest={list(listed)} registry={stems}"
            )
            continue
        for script, stem in entries:
            if f"figures/{script}" not in str(listed[stem]):
                problems.append(f"figures.{group}.{stem} does not run figures/{script}")
    if len(registry["extended"]) > MAX_EXTENDED_DATA_FIGURES:
        problems.append(f"more than {MAX_EXTENDED_DATA_FIGURES} Extended Data figures")
    return problems


# Imports one manifest entry point the way `python path/to/script.py` or `python -m pkg` does: the
# script directory (not the repo root) leads sys.path, so only the environment's PYTHONPATH can
# supply `analysis`, `figures`, `tables` and `paths`. The module name is not `__main__`, so no
# `if __name__ == "__main__"` block runs.
_SMOKE_IMPORT = """
import importlib, importlib.util, os, sys
kind, target = sys.argv[1:3]
if kind == "module":
    sys.path.insert(0, os.getcwd())
    importlib.import_module(target)
else:
    sys.path.insert(0, os.path.dirname(target))
    spec = importlib.util.spec_from_file_location("_smoke_entry", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses and pickling look the module up here
    spec.loader.exec_module(module)
"""


SMOKE_IMPORT_WORKERS = 4
# The PYTHONPATH advice applies only when the project's own packages are what failed to import.
_MISSING_PROJECT_MODULE = re.compile(r"ModuleNotFoundError: No module named '(analysis|figures|tables|paths)(\.[^']*)?'")


def _manifest_entry_points(manifest: dict) -> list[tuple[str, str]]:
    """Unique ``("script" | "module", target)`` pairs named by manifest commands."""
    found: dict[tuple[str, str], None] = {}
    for _, command in iter_commands(manifest):
        parts = shlex.split(command)
        script = next((p for p in parts if p.endswith(".py")), None)
        if script:
            found[("script", script)] = None
        if "-m" in parts and parts.index("-m") + 1 < len(parts):
            found[("module", parts[parts.index("-m") + 1])] = None
    return sorted(found)


def _smoke_import(entry: tuple[str, str]) -> str | None:
    """Failure text if importing the entry point raises, else None. Nothing is executed."""
    kind, target = entry
    arg = target if kind == "module" else str(ROOT / target)
    result = subprocess.run(
        [sys.executable, "-P", "-c", _SMOKE_IMPORT, kind, arg],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    if result.returncode == 0:
        return None
    last = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
    return f"{target}: import failed ({last})"


def _check_entry_point_imports(manifest: dict) -> list[str]:
    """Import every manifest entry point under the environment's PYTHONPATH; one problem per failure."""
    entries = _manifest_entry_points(manifest)
    with ThreadPoolExecutor(max_workers=SMOKE_IMPORT_WORKERS) as pool:
        results = list(pool.map(_smoke_import, entries))
    problems = [f"entry point {r}" for r in results if r]
    if any(_MISSING_PROJECT_MODULE.search(r) for r in results if r):
        problems.append(
            "entry points import `analysis`, `figures`, `tables` and `paths` through PYTHONPATH "
            "(pixi.toml activation env); run the check as `pixi run manifest-check`"
        )
    return problems


def check_manifest() -> int:
    """Validate manifest commands, local config templates, and pixi tasks."""
    manifest = load_manifest()
    failures: list[str] = []
    warnings: list[str] = []

    for name, command in iter_commands(manifest):
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            failures.append(f"{name}: invalid shell command ({exc})")
            continue
        script = next((p for p in parts if p.endswith((".py", ".R"))), None)
        if script and not (ROOT / script).exists():
            failures.append(f"{name}: missing {script}")
        if "-m" in parts and parts.index("-m") + 1 < len(parts):
            module = parts[parts.index("-m") + 1]
            module_path = ROOT / module.replace(".", "/")
            if not (
                module_path.with_suffix(".py").exists()
                or (module_path / "__main__.py").exists()
                or (module_path / "__init__.py").exists()
            ):
                failures.append(f"{name}: missing module {module}")

    path_keys = _collect_paths_keys(manifest)
    try:
        template_cfg = _load_yaml_mapping(PATHS_TEMPLATE)
    except FileNotFoundError:
        failures.append(f"missing {PATHS_TEMPLATE.relative_to(ROOT)}")
    except ValueError as exc:
        failures.append(str(exc))
    else:
        missing_template_keys = [key for key in path_keys if key not in template_cfg]
        if missing_template_keys:
            failures.append(
                f"{PATHS_TEMPLATE.name}: missing keys {', '.join(missing_template_keys)}"
            )

    local_paths = _resolve_local_paths_file()
    if local_paths.exists():
        try:
            local_cfg = _load_yaml_mapping(local_paths)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            missing_local_keys = [
                key for key in path_keys if key in CORE_PATH_KEYS and not local_cfg.get(key)
            ]
            if missing_local_keys:
                failures.append(
                    f"{local_paths.name}: missing keys {', '.join(missing_local_keys)}"
                )
            missing_optional_keys = [
                key for key in path_keys if key not in CORE_PATH_KEYS and not local_cfg.get(key)
            ]
            if missing_optional_keys:
                warnings.append(
                    f"{local_paths.name}: optional keys not configured {', '.join(missing_optional_keys)}"
                )
    else:
        warnings.append(
            f"Missing {_relative_display(local_paths)}; copy {PATHS_TEMPLATE.name} to "
            f"{LOCAL_PATHS_FILE.name} or set {PATHS_ENV_VAR} before running data-dependent entrypoints."
        )

    try:
        pixi_tasks = _load_pixi_tasks()
    except FileNotFoundError:
        failures.append(f"missing {PIXI_FILE.relative_to(ROOT)}")
    except ValueError as exc:
        failures.append(str(exc))
    else:
        missing_tasks = [
            task for task in _collect_manifest_pixi_tasks(manifest) if task not in pixi_tasks
        ]
        if missing_tasks:
            failures.append(f"pixi.toml: missing tasks {', '.join(missing_tasks)}")

    failures.extend(_check_code_roots(manifest))
    failures.extend(_check_figure_registry(manifest))
    failures.extend(_check_stage_graph(manifest))
    failures.extend(_check_entry_point_imports(manifest))

    for warning in warnings:
        print(f"WARN {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("Manifest/config checks OK")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    return check_manifest()
