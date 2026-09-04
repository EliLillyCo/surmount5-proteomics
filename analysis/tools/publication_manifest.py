"""Utilities for the manuscript reproduction manifest.

This module deliberately stays lightweight: it validates the public manifest,
lists commands, performs static figure-style checks that do not require
protected clinical trial data, and records/compares figure render baselines for
safe refactoring.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "analysis" / "manifest.yaml"
FIGURES_DIR = ROOT / "figures"
DEFAULT_BASELINE = FIGURES_DIR / "figure_regression_baseline.json"
PIXI_FILE = ROOT / "pixi.toml"
PATHS_TEMPLATE = ROOT / "paths.example.yaml"
LOCAL_PATHS_FILE = ROOT / "paths.yaml"
SUPPLEMENTARY_LEGENDS = FIGURES_DIR / "configs" / "supplementary_legends.yaml"
DEFAULT_RENDER_ROOT = ROOT / "analysis" / "outputs" / "_publication_manifest"
PATHS_ENV_VAR = "SURMOUNT5_PATHS_YAML"
CORE_PATH_KEYS = ("mmrm_root", "qc_root", "pheno_root")
_PATHS_REF_RE = re.compile(r"paths\.yaml:([A-Za-z0-9_]+)")
_PIXI_RUN_RE = re.compile(r"\bpixi\s+run\s+([A-Za-z0-9][A-Za-z0-9_-]*)\b")


@dataclass(frozen=True)
class FigureAudit:
    script: str
    has_theme: bool
    uses_save_figure: bool
    direct_savefig: bool
    panel_font_issue: str | None
    small_font_issue: str | None


@dataclass(frozen=True)
class RenderResult:
    name: str
    command: str
    output_dir: Path
    returncode: int
    artifacts: dict[str, dict[str, object]]


def load_manifest(path: Path = MANIFEST) -> dict:
    """Load the manuscript manifest."""
    with path.open() as f:
        return yaml.safe_load(f)


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


def _load_supplementary_legend_keys(path: Path | None = None) -> set[str]:
    path = SUPPLEMENTARY_LEGENDS if path is None else path
    payload = _load_yaml_mapping(path)
    entries = payload.get("figures", [])
    if not isinstance(entries, list):
        raise ValueError(f"Expected a 'figures' list in {path}")
    keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("key"):
            raise ValueError(f"Each supplementary legend entry in {path} must define 'key'")
        keys.add(str(entry["key"]))
    return keys


def _new_output_dir(prefix: str) -> Path:
    DEFAULT_RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = DEFAULT_RENDER_ROOT / f"{prefix}{stamp}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = DEFAULT_RENDER_ROOT / f"{prefix}{stamp}-{counter}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


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
            if module.startswith("_internal."):
                rel = module.replace(".", "/")
                return path.parent / f"{rel}.py"
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
        uses_save_figure="save_figure(" in source,
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

    try:
        legend_keys = _load_supplementary_legend_keys()
    except (FileNotFoundError, ValueError) as exc:
        failures.append(str(exc))
    else:
        supplementary_keys = set(manifest.get("figures", {}).get("supplementary", {}))
        missing_manifest = sorted(legend_keys - supplementary_keys)
        missing_legends = sorted(supplementary_keys - legend_keys)
        if missing_manifest:
            failures.append(
                "analysis/manifest.yaml: supplementary legends missing manifest commands for "
                + ", ".join(missing_manifest)
            )
        if missing_legends:
            failures.append(
                "figures/configs/supplementary_legends.yaml: missing keys "
                + ", ".join(missing_legends)
            )

    for warning in warnings:
        print(f"WARN {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("Manifest/config checks OK")
    return 0


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(
        fmt.strip().lstrip(".").lower() for fmt in raw.split(",") if fmt.strip()
    )
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def _slugify_render_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _relative_display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _select_figure_commands(
    manifest: dict, names: Sequence[str] | None
) -> list[tuple[str, str]]:
    commands = dict(iter_figure_commands(manifest))
    if not names:
        return list(commands.items())
    missing = [name for name in names if name not in commands]
    if missing:
        raise KeyError(f"Unknown figure names: {', '.join(missing)}")
    return [(name, commands[name]) for name in names]


def _figure_environment(output_dir: Path, formats: Sequence[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["FIGURE_OUTPUT_DIR"] = str(output_dir)
    env["FIGURE_OUTPUT_FORMATS"] = ",".join(formats)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def _png_quantized_sha256(path: Path, *, shift_bits: int = 1) -> str:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixels = bytearray(rgba.tobytes())
    if shift_bits:
        for i, value in enumerate(pixels):
            pixels[i] = value >> shift_bits
    digest = hashlib.sha256()
    digest.update(pixels)
    return digest.hexdigest()


def _png_preview_sha256(path: Path, *, size: int = 256) -> str:
    with Image.open(path) as image:
        rgba = image.convert("RGBA").resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )
        pixels = rgba.tobytes()
    digest = hashlib.sha256()
    digest.update(pixels)
    return digest.hexdigest()


def _artifact_metadata(path: Path, render_root: Path) -> dict[str, object]:
    metadata: dict[str, object] = {
        "path": str(path.relative_to(render_root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".png":
        width, height = _png_dimensions(path)
        metadata["width_px"] = width
        metadata["height_px"] = height
        metadata["pixel_sha256_q1"] = _png_quantized_sha256(path)
        metadata["pixel_sha256_preview256"] = _png_preview_sha256(path)
    return metadata


def _collect_artifacts(output_dir: Path, render_root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lstrip(".").lower() or path.name
        if suffix in artifacts:
            raise RuntimeError(
                f"{output_dir} produced multiple '.{suffix}' artifacts; "
                "the regression harness expects one artifact per format."
            )
        artifacts[suffix] = _artifact_metadata(path, render_root)
    if not artifacts:
        raise RuntimeError(f"{output_dir} did not produce any files.")
    return artifacts


def _render_environment_metadata() -> dict[str, str]:
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "py_hash_seed": "0",
    }
    try:
        import matplotlib

        metadata["matplotlib"] = matplotlib.__version__
    except Exception:
        metadata["matplotlib"] = "unavailable"
    try:
        import ultraplot as uplt

        metadata["ultraplot"] = getattr(uplt, "__version__", "unknown")
    except Exception:
        metadata["ultraplot"] = "unavailable"
    return metadata


@contextlib.contextmanager
def _output_root(path: Path | None, prefix: str) -> Iterator[Path]:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    yield _new_output_dir(prefix)


def _render_figures(
    manifest: dict,
    names: Sequence[str] | None,
    out_dir: Path,
    formats: Sequence[str],
    *,
    keep_going: bool,
) -> list[RenderResult]:
    results: list[RenderResult] = []
    for name, command in _select_figure_commands(manifest, names):
        figure_dir = out_dir / _slugify_render_name(name)
        if figure_dir.exists():
            shutil.rmtree(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        print(f"RUN {name}\t{_relative_display(figure_dir)}")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise RuntimeError(f"{name}: invalid command ({exc})") from exc
        completed = subprocess.run(
            parts,
            cwd=ROOT,
            env=_figure_environment(figure_dir, formats),
            check=False,
        )
        artifacts: dict[str, dict[str, object]] = {}
        if completed.returncode == 0:
            try:
                artifacts = _collect_artifacts(figure_dir, out_dir)
            except RuntimeError as exc:
                completed = subprocess.CompletedProcess(
                    command, returncode=1, stdout=None, stderr=str(exc)
                )
        results.append(
            RenderResult(
                name=name,
                command=command,
                output_dir=figure_dir,
                returncode=completed.returncode,
                artifacts=artifacts,
            )
        )
        if completed.returncode != 0 and not keep_going:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return results


def _baseline_payload(
    results: Sequence[RenderResult], formats: Sequence[str]
) -> dict[str, object]:
    figures = {
        result.name: {
            "command": result.command,
            "output_dir": result.output_dir.name,
            "artifacts": result.artifacts,
        }
        for result in results
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _render_environment_metadata(),
        "formats": list(formats),
        "figures": figures,
    }


def _compare_figure_records(
    name: str,
    expected: dict[str, object],
    observed: dict[str, dict[str, object]],
) -> list[str]:
    issues: list[str] = []
    expected_artifacts = expected["artifacts"]
    if not isinstance(expected_artifacts, dict):
        return [f"{name}: baseline artifacts are malformed"]

    for suffix, expected_meta in expected_artifacts.items():
        observed_meta = observed.get(suffix)
        if observed_meta is None:
            issues.append(f"{name}: missing {suffix} artifact")
            continue
        if suffix == "png":
            for key in ("width_px", "height_px"):
                if observed_meta.get(key) != expected_meta.get(key):
                    issues.append(
                        f"{name}: png {key} mismatch "
                        f"(expected {expected_meta.get(key)}, got {observed_meta.get(key)})"
                    )
            if observed_meta.get("sha256") != expected_meta.get("sha256"):
                expected_q1 = expected_meta.get("pixel_sha256_q1")
                observed_q1 = observed_meta.get("pixel_sha256_q1")
                expected_preview = expected_meta.get("pixel_sha256_preview256")
                observed_preview = observed_meta.get("pixel_sha256_preview256")
                preview_match = (
                    expected_preview is not None
                    and observed_preview is not None
                    and observed_preview == expected_preview
                )
                q1_match = (
                    expected_q1 is not None
                    and observed_q1 is not None
                    and observed_q1 == expected_q1
                )
                if not q1_match and not preview_match:
                    issues.append(f"{name}: png hash mismatch")
    unexpected = sorted(set(observed) - set(expected_artifacts))
    for suffix in unexpected:
        issues.append(f"{name}: unexpected {suffix} artifact")
    return issues


def _environment_warnings(
    expected: dict[str, object], observed: dict[str, object]
) -> list[str]:
    warnings: list[str] = []
    for key in ("python", "platform", "matplotlib", "ultraplot", "py_hash_seed"):
        if expected.get(key) != observed.get(key):
            warnings.append(
                f"environment mismatch for {key}: "
                f"baseline={expected.get(key)!r}, current={observed.get(key)!r}"
            )
    return warnings


def cmd_list(_: argparse.Namespace) -> int:
    for name, command in iter_commands(load_manifest()):
        print(f"{name}\t{command}")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    return check_manifest()


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


def cmd_run(args: argparse.Namespace) -> int:
    commands = dict(iter_commands(load_manifest()))
    if args.name not in commands:
        print(f"Unknown command: {args.name}")
        return 1
    try:
        parts = shlex.split(commands[args.name])
    except ValueError as exc:
        print(f"Invalid command for {args.name}: {exc}")
        return 1
    return subprocess.call(parts, cwd=ROOT)


def cmd_render_figures(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    formats = _parse_formats(args.formats)
    with _output_root(args.out_dir, "figure-render-") as out_dir:
        results = _render_figures(
            manifest,
            args.name,
            out_dir,
            formats,
            keep_going=True,
        )
        failures = [result for result in results if result.returncode != 0]
        for result in results:
            if result.returncode == 0:
                print(
                    f"OK {result.name}\t"
                    f"{', '.join(sorted(result.artifacts))}\t"
                    f"{_relative_display(result.output_dir)}"
                )
            else:
                print(f"FAIL {result.name}\texit code {result.returncode}")
        print(f"Rendered artifacts were written to {_relative_display(out_dir)}")
        return 1 if failures else 0


def cmd_record_figure_baseline(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    formats = _parse_formats(args.formats)
    with _output_root(args.out_dir, "figure-baseline-") as out_dir:
        results = _render_figures(
            manifest,
            args.name,
            out_dir,
            formats,
            keep_going=True,
        )
        failures = [result for result in results if result.returncode != 0]
        if failures:
            for result in failures:
                print(f"FAIL {result.name}\texit code {result.returncode}")
            print("Baseline was not written because one or more figures failed.")
            return 1
        payload = _baseline_payload(results, formats)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote baseline to {_relative_display(args.baseline)}")
        print(f"Rendered artifacts kept in {_relative_display(out_dir)}")
        return 0


def cmd_compare_figure_baseline(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    baseline = json.loads(args.baseline.read_text())
    baseline_figures = baseline.get("figures", {})
    if not isinstance(baseline_figures, dict):
        print(f"Malformed baseline: {args.baseline}")
        return 1
    names = args.name or list(baseline_figures)
    missing = [name for name in names if name not in baseline_figures]
    if missing:
        print(f"Unknown baseline figures: {', '.join(missing)}")
        return 1
    formats = tuple(baseline.get("formats", [])) or _parse_formats(args.formats)
    with _output_root(args.out_dir, "figure-compare-") as out_dir:
        results = _render_figures(
            manifest,
            names,
            out_dir,
            formats,
            keep_going=True,
        )
        failures: list[str] = []
        for result in results:
            if result.returncode != 0:
                failures.append(f"{result.name}: render failed with exit code {result.returncode}")
                continue
            failures.extend(
                _compare_figure_records(
                    result.name,
                    baseline_figures[result.name],
                    result.artifacts,
                )
            )
        for warning in _environment_warnings(
            baseline.get("environment", {}), _render_environment_metadata()
        ):
            print(f"WARN {warning}")
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            print(f"Candidate artifacts kept in {_relative_display(out_dir)}")
            return 1
        print("Figure baseline matches")
        print(f"Candidate artifacts kept in {_relative_display(out_dir)}")
        return 0


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
        help="Render figures and write a PNG-hash baseline manifest for refactor checks",
    )
    p_record.add_argument(
        "--name",
        action="append",
        help="Manifest figure name to include. Defaults to all figures.",
    )
    p_record.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
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
        default=DEFAULT_BASELINE,
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
