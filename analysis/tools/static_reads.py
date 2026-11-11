"""Which stage outputs each registered figure script and table module reads, found statically.

``stages_read`` follows the names a figure's ``__main__`` block (or a table's ``build``) reaches through
the repo modules and matches their string literals against the stage outputs; ``downstream_problems``
compares the result with the manifest's ``downstream`` lists. Nothing is imported or run.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from analysis.output_layout import ROOT
from analysis.tools.manifest_stages import _listed_downstream, registered_figures, registered_tables


# ---------------------------------------------------------------------------
# Static reads: which stage outputs a figure or table reaches
# ---------------------------------------------------------------------------

_TOP_PACKAGES = ("analysis", "figures", "tables", "paths")


# A stage output is matched against string literals of reached code by its file name; an
# f-string counts only with at least this many literal characters, so `f"{x}"` matches nothing.
_MIN_FSTRING_LITERAL = 8


# Reads the static analysis finds but the script never performs, by consumer and stage, each
# with the reason.
IMPORTED_NOT_READ: dict[str, dict[str, str]] = {}


@dataclass
class _Module:
    """One repo module's top-level names, imports and import-time statements."""

    name: str
    defs: dict[str, list[ast.AST]]
    imports: dict[str, tuple[str, str | None]]
    toplevel: list[ast.AST]
    main: list[ast.AST]


def _module_path(name: str) -> Path | None:
    base = ROOT / name.replace(".", "/")
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_main_guard(stmt: ast.stmt) -> bool:
    test = getattr(stmt, "test", None)
    return (
        isinstance(stmt, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


_MODULES: dict[str, _Module] = {}


def _parse_module(name: str) -> _Module | None:
    """The repo module ``name`` parsed once (None for anything outside the repo packages)."""
    if name.split(".")[0] not in _TOP_PACKAGES:
        return None
    key = f"{ROOT}:{name}"
    if key in _MODULES:
        return _MODULES[key]
    path = _module_path(name)
    if path is None:
        return None
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    mod = _Module(name, {}, {}, [], [])
    for stmt in ast.parse(path.read_text()).body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                mod.imports[local] = (alias.name if alias.asname else local, None)
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level:
                base = package.split(".")[: len(package.split(".")) - stmt.level + 1]
                source = ".".join([*base, *([stmt.module] if stmt.module else [])])
            else:
                source = stmt.module or ""
            for alias in stmt.names:
                local = alias.asname or alias.name
                submodule = f"{source}.{alias.name}"
                mod.imports[local] = (
                    (submodule, None) if _module_path(submodule) else (source, alias.name)
                )
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mod.defs.setdefault(stmt.name, []).append(stmt)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        mod.defs.setdefault(node.id, []).append(stmt)
        elif _is_main_guard(stmt):
            mod.main.append(stmt)
        else:
            mod.toplevel.append(stmt)
    _MODULES[key] = mod
    return mod


def _attribute_chain(node: ast.AST) -> list[str] | None:
    """``a.b.c`` as ``["a", "b", "c"]``; None when the base is not a plain name."""
    chain: list[str] = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(chain)]


def reached_names(
    module: str, *, run_main: bool = True, entry: str | None = None
) -> tuple[set[tuple[str, str]], list[ast.AST]]:
    """Every repo-level ``(module, name)`` that running ``module`` can reach, and their code.

    Reachability is by name: a function, class or module-level assignment counts once some
    reached code refers to it (directly, through an import, or as ``module.attr``);
    import-time statements of every reached module count. A function's whole body counts,
    branches included, so this over-approximates what a run does.

    Args:
        module: Dotted name of the entry module.
        run_main: Include its ``if __name__ == "__main__"`` block.
        entry: A top-level name of ``module`` that is called from outside (a table's ``build``).

    Returns:
        ``(reached names, reached AST nodes)``.
    """
    names: set[tuple[str, str]] = set()
    nodes: list[ast.AST] = []
    executed: set[str] = set()
    work: list[tuple[str, str | None]] = [(module, None), (module, entry)]
    while work:
        mod_name, name = work.pop()
        mod = _parse_module(mod_name)
        if mod is None:
            continue
        body: list[ast.AST] = []
        if mod_name not in executed:
            executed.add(mod_name)
            body += mod.toplevel + (mod.main if run_main and mod_name == module else [])
        if name is not None and (mod_name, name) not in names:
            names.add((mod_name, name))
            body += mod.defs.get(name, [])
            if name not in mod.defs and name in mod.imports:  # a re-export
                work.append(mod.imports[name])
        for stmt in body:
            nodes.append(stmt)
            for node in ast.walk(stmt):
                if isinstance(node, ast.Attribute):
                    chain = _attribute_chain(node)
                    if chain and chain[0] in mod.imports and mod.imports[chain[0]][1] is None:
                        target = mod.imports[chain[0]][0]
                        for attr in chain[1:]:
                            if _module_path(f"{target}.{attr}") is None:
                                work.append((target, attr))
                                break
                            target = f"{target}.{attr}"
                            work.append((target, None))
                elif isinstance(node, ast.Name):
                    if node.id in mod.defs:
                        work.append((mod_name, node.id))
                    elif node.id in mod.imports:
                        work.append(mod.imports[node.id])
    return names, nodes


def _literal_patterns(nodes: list[ast.AST]) -> list[re.Pattern[str]]:
    """Regexes of the string literals and f-strings in ``nodes`` (an f-string field matches a word)."""
    patterns: list[re.Pattern[str]] = []
    for stmt in nodes:
        for node in ast.walk(stmt):
            if isinstance(node, ast.JoinedStr):
                literal = [v.value for v in node.values if isinstance(v, ast.Constant)]
                if sum(len(part) for part in literal) < _MIN_FSTRING_LITERAL:
                    continue
                patterns.append(
                    re.compile(
                        "".join(
                            re.escape(v.value) if isinstance(v, ast.Constant) else r"[\w.-]+"
                            for v in node.values
                        )
                    )
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                patterns.append(re.compile(re.escape(node.value)))
    return patterns


def _consumers() -> dict[str, str]:
    """Registered figure scripts and table modules (repo paths) -> their label."""
    out: dict[str, str] = {}
    for group, entries in registered_figures().items():
        for script, stem in entries:
            out.setdefault(f"figures/{script}", stem)
    for tab, _, module, _ in registered_tables():
        out.setdefault(f"tables/{module}.py", tab)
    return out


def stages_read(consumer: str, manifest: dict) -> dict[str, set[str]]:
    """Stages whose outputs a registered figure script or table module reaches, statically.

    A repo-relative output (``analysis/outputs/...``, ``data/external/...``) is read when a
    string literal of reached code matches its file name; a mounted output is read through
    the readers its stage names in ``read_through`` (dotted ``module.name``). A table module
    is entered through its ``build`` (the workbook calls only that), a figure script through
    its ``__main__`` block.

    Args:
        consumer: ``figures/<script>.py`` or ``tables/<module>.py``.
        manifest: The loaded manifest.

    Returns:
        ``{stage: evidence}``: the matched output paths or reader names.
    """
    module = _module_name(ROOT / consumer)
    if consumer.startswith("tables/"):
        names, nodes = reached_names(module, run_main=False, entry="build")
    else:
        names, nodes = reached_names(module)
    patterns = _literal_patterns(nodes)
    found: dict[str, set[str]] = {}
    for stage, spec in manifest.get("analysis_outputs", {}).items():
        for out in spec.get("outputs", []):
            if out.startswith(("paths.yaml:", "<")) or out.endswith("/"):
                continue
            file_name = out.rsplit("/", 1)[-1]
            if any(p.fullmatch(file_name) or p.fullmatch(out) for p in patterns):
                found.setdefault(stage, set()).add(out)
        for reader in spec.get("read_through", []):
            mod_name, _, name = reader.rpartition(".")
            if (mod_name, name) in names:
                found.setdefault(stage, set()).add(reader)
    return found


def effective_reads(manifest: dict) -> dict[str, dict[str, set[str]]]:
    """``stages_read`` for every registered figure and table, less ``IMPORTED_NOT_READ``."""
    reads = {}
    for consumer in _consumers():
        found = stages_read(consumer, manifest)
        for stage in IMPORTED_NOT_READ.get(consumer, {}):
            found.pop(stage, None)
        reads[consumer] = found
    return reads


def downstream_problems(manifest: dict) -> list[str]:
    """Where the manifest's ``downstream`` lists disagree with what the code reads.

    Checked statically (``stages_read``): every ``downstream`` path exists and every
    ``read_through`` reader is defined; a registered figure or table that reads a stage's
    output is listed in that stage's ``downstream`` (a bare ``figures`` / ``tables`` lists
    all of them), and one that is listed reads it; every figure and table reads at least one
    stage; every ``IMPORTED_NOT_READ`` exemption still names a stage the code reaches.

    Returns:
        One message per disagreement.
    """
    stages = manifest.get("analysis_outputs", {})
    problems: list[str] = []
    for stage, spec in stages.items():
        for path in spec.get("downstream", []):
            if path not in ("figures", "tables") and not (ROOT / path).exists():
                problems.append(f"stage {stage}: downstream '{path}' does not exist")
        for reader in spec.get("read_through", []):
            module, _, name = reader.rpartition(".")
            parsed = _parse_module(module)
            if parsed is None or name not in parsed.defs:
                problems.append(f"stage {stage}: read_through '{reader}' is not defined")
    consumers = _consumers()
    raw = {consumer: stages_read(consumer, manifest) for consumer in consumers}
    reads = effective_reads(manifest)
    for consumer, label in consumers.items():
        for stage, spec in stages.items():
            listed = _listed_downstream(spec, consumer)
            if stage in reads[consumer] and not listed:
                evidence = ", ".join(sorted(reads[consumer][stage]))
                problems.append(
                    f"{consumer} ({label}) reads {evidence} of stage {stage}, "
                    "which does not list it downstream"
                )
            elif listed and stage not in reads[consumer]:
                problems.append(
                    f"{consumer} ({label}) is listed downstream of stage {stage} "
                    "but reads none of its outputs"
                )
    for consumer, label in consumers.items():
        if not reads[consumer]:
            problems.append(f"{consumer} ({label}) is downstream of no analysis_outputs stage")
    for consumer, exempt in IMPORTED_NOT_READ.items():
        for stage in exempt:
            if stage not in raw.get(consumer, {}):
                problems.append(
                    f"IMPORTED_NOT_READ['{consumer}']['{stage}'] names a stage the code does not reach"
                )
    return problems
