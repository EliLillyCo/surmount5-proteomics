"""Static figure-style audit of the public figure scripts (``audit-figures``); needs no trial data.

Each script (or the ``figures._internal`` implementation it imports as ``_impl``) must call
``init_figure_theme()`` and ``save_figure()``, never ``fig.savefig()``, and keep literal font sizes
inside the body and panel-label policy.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

from analysis.output_layout import ROOT
from analysis.tools.manifest_stages import FIGURES_DIR


@dataclass(frozen=True)
class FigureAudit:
    script: str
    has_theme: bool
    uses_save_figure: bool
    direct_savefig: bool
    panel_font_issue: str | None
    small_font_issue: str | None


def _literal_font_sizes(tree: ast.AST) -> list[float]:
    sizes: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "fontsize":
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                sizes.append(float(value.value))
    return sizes


def _wrapper_impl_path(path: Path, tree: ast.AST) -> Path | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname != "_impl":
                continue
            module = alias.name
            if module.startswith("figures._internal."):
                rel = module.replace(".", "/")
                return ROOT / f"{rel}.py"
    return None


def audit_figure_script(path: Path, *, resolve_wrapper: bool = True) -> FigureAudit:
    """Static audit for Nature-style figure conventions."""
    source = path.read_text()
    tree = ast.parse(source)
    local_panel_defs = re.findall(r"FS_PANEL_LABEL\s*=\s*([^\n#]+)", source)
    panel_font_issue = None
    if any(defn.strip() not in {"FS_PANEL", "8", "8.0"} for defn in local_panel_defs):
        panel_font_issue = f"FS_PANEL_LABEL definitions: {local_panel_defs}"

    sizes = _literal_font_sizes(tree)
    small = sorted({s for s in sizes if 0 < s < 5.0})
    large = sorted({s for s in sizes if 7.0 < s < 8.0 or s > 8.0})
    small_font_issue = None
    if small or large:
        small_font_issue = (
            f"literal sizes outside body/panel policy: small={small}, large={large}"
        )

    audit = FigureAudit(
        script=path.name,
        has_theme="init_figure_theme()" in source,
        uses_save_figure="save_figure(" in source or "save_figure_stack(" in source,
        direct_savefig="fig.savefig(" in source,
        panel_font_issue=panel_font_issue,
        small_font_issue=small_font_issue,
    )
    if not resolve_wrapper:
        return audit

    impl_path = _wrapper_impl_path(path, tree)
    if impl_path is None or not impl_path.exists():
        return audit

    impl_audit = audit_figure_script(impl_path, resolve_wrapper=False)
    return FigureAudit(
        script=path.name,
        has_theme=audit.has_theme or impl_audit.has_theme,
        uses_save_figure=audit.uses_save_figure or impl_audit.uses_save_figure,
        direct_savefig=audit.direct_savefig or impl_audit.direct_savefig,
        panel_font_issue=audit.panel_font_issue or impl_audit.panel_font_issue,
        small_font_issue=audit.small_font_issue or impl_audit.small_font_issue,
    )


def audit_figures() -> list[FigureAudit]:
    """Audit all public figure scripts."""
    audits = []
    for path in sorted(FIGURES_DIR.glob("*.py")):
        if (
            path.name.startswith("_")
            or path.name.startswith("build_")
            or path.name == "__init__.py"
        ):
            continue
        audits.append(audit_figure_script(path))
    return audits


def cmd_audit_figures(_: argparse.Namespace) -> int:
    failures = 0
    for audit in audit_figures():
        issues = []
        if not audit.has_theme:
            issues.append("missing init_figure_theme()")
        if not audit.uses_save_figure:
            issues.append("missing save_figure()")
        if audit.direct_savefig:
            issues.append("direct fig.savefig()")
        if audit.panel_font_issue:
            issues.append(audit.panel_font_issue)
        if audit.small_font_issue:
            issues.append(audit.small_font_issue)
        status = "OK" if not issues else "WARN"
        failures += bool(issues)
        print(f"{status}\t{audit.script}\t{'; '.join(issues)}")
    return 1 if failures else 0
