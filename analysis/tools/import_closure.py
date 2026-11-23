"""The repo files a script reaches by following Python imports.

``closure`` starts from root files and follows every ``import`` and ``from ... import`` in
them, function-level imports included, to the modules that live in the repository. The
``__init__.py`` of each package containing a reached file is part of the closure, because
importing a module imports its packages. Third-party and standard-library modules resolve
to nothing and are not followed. A root that is not a Python file (an R script) is kept and
not walked.
"""

from __future__ import annotations

import ast
from pathlib import Path

from analysis.output_layout import ROOT


def _module_paths(module: str, base: Path) -> list[Path]:
    parts = module.split(".")
    return [base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")]


def _resolve(module: str, search: list[Path]) -> Path | None:
    """The file a dotted module name maps to under the first search directory that has it.

    Args:
        module: Dotted module name.
        search: Directories the name is resolved against, in order.

    Returns:
        The module's ``.py`` or package ``__init__.py``; None for a module outside the repo.
    """
    for base in search:
        for candidate in _module_paths(module, base):
            if candidate.is_file():
                return candidate
    return None


def _packages_of(path: Path, root: Path) -> set[Path]:
    """The ``__init__.py`` of every package containing a file, up to the repo root."""
    inits = set()
    parent = path.parent
    while parent != root and root in parent.parents:
        init = parent / "__init__.py"
        if init.is_file():
            inits.add(init)
        parent = parent.parent
    return inits


def _imports(path: Path, root: Path) -> set[str]:
    """Module names a Python file imports, including inside functions."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                prefix = ".".join(base.relative_to(root).parts)
                if node.module:
                    prefix = f"{prefix}.{node.module}"
                names.add(prefix)
                names.update(f"{prefix}.{alias.name}" for alias in node.names)
            elif node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def closure(roots: set[str], root: Path = ROOT) -> set[str]:
    """Every repo file reachable from the roots by following Python imports.

    Absolute names resolve against the repository root and against the directory of each
    Python root, which is where a script run by path finds its sibling modules.

    Args:
        roots: Repo-relative root file paths; non-Python roots are kept but not walked.
        root: The repository root.

    Returns:
        The roots plus every repo file they import, directly or transitively, with the
        ``__init__.py`` of each containing package.
    """
    search = [root] + sorted({(root / rel).parent for rel in roots if rel.endswith(".py")})
    reached: set[str] = set()
    queue = list(roots)
    while queue:
        rel = queue.pop()
        if rel in reached:
            continue
        reached.add(rel)
        path = root / rel
        if path.suffix != ".py" or not path.is_file():
            continue
        targets = _packages_of(path, root)
        for module in _imports(path, root):
            target = _resolve(module, search)
            if target is not None:
                targets.add(target)
                targets |= _packages_of(target, root)
        queue.extend(str(target.relative_to(root)) for target in targets)
    return reached
