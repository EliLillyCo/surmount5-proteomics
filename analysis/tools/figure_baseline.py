"""Figure render baselines: render the manifest's figure commands, record their hashes and compare a rerender.

``render-figures``, ``record-figure-baseline`` and ``compare-figure-baseline`` need the trial data. A
figure whose PDF has the baseline's bytes matches. Otherwise its PNG is compared: it matches on the
exact bytes, a 1-bit-quantized pixel hash or a 256 x 256 preview hash (the preview hash matches only
that resized preview, not every pixel of the full render). The baseline also records the commit it
was rendered at and a checksum of every input the figure build reads (``provenance.build_inputs``);
a comparison warns when either differs.
``_figure_environment`` is the environment every render runs in, shared with ``figures/build_figures.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from analysis.output_layout import RENDER_ROOT, ROOT
from analysis.tools.manifest_stages import _relative_display, iter_figure_commands, load_manifest
from analysis.tools.provenance import Workspace, build_inputs, git_state, snapshot


DEFAULT_RENDER_ROOT = RENDER_ROOT


@dataclass(frozen=True)
class RenderResult:
    name: str
    command: str
    output_dir: Path
    returncode: int
    artifacts: dict[str, dict[str, object]]


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


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(
        fmt.strip().lstrip(".").lower() for fmt in raw.split(",") if fmt.strip()
    )
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def _slugify_render_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


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
    import matplotlib

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "py_hash_seed": "0",
        "matplotlib": matplotlib.__version__,
    }


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
        completed = subprocess.run(
            command,
            shell=True,
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


def _commit_metadata() -> dict[str, object]:
    """The commit the figures are rendered at, and whether tracked files differ from it."""
    state = git_state(ROOT)
    return {"sha": state["sha"], "dirty": state["dirty"]}


def _input_digests(ws: Workspace | None = None) -> dict[str, str]:
    """Checksum of every input of the figure build, keyed by its manifest reference.

    A reference that resolves to several paths gets one key per path, ``<ref> [<tail>]``, the
    tail being the part of the path the matches do not share. A reference that is absent or
    unconfigured keeps its status (``missing``, ``unconfigured``) as the value. The record holds
    no mount path.
    """
    ws = ws or Workspace.default()
    entries = snapshot(build_inputs(ws.manifest, "figures"), ws)
    paths_of: dict[str, list[str]] = {}
    for entry in entries:
        paths_of.setdefault(entry["ref"], []).append(entry.get("path", ""))
    digests: dict[str, str] = {}
    for entry in entries:
        key, paths = entry["ref"], paths_of[entry["ref"]]
        if len(paths) > 1:
            key = f"{key} [{os.path.relpath(entry['path'], os.path.commonpath(paths))}]"
        digests[key] = entry.get("sha256", entry["kind"])
    return digests


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
        "git": _commit_metadata(),
        "inputs": _input_digests(),
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

    # A byte-identical PDF is the same figure; PNG bytes can differ between rasterizer builds.
    same_pdf = (
        "pdf" in expected_artifacts
        and "pdf" in observed
        and observed["pdf"].get("sha256") == expected_artifacts["pdf"].get("sha256")
    )
    for suffix, expected_meta in expected_artifacts.items():
        observed_meta = observed.get(suffix)
        if observed_meta is None:
            issues.append(f"{name}: missing {suffix} artifact")
            continue
        if suffix == "png" and not same_pdf:
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


def _provenance_warnings(baseline: dict[str, object]) -> list[str]:
    """Differences between the baseline's commit and inputs and the current ones.

    A baseline recorded before these fields existed gives one warning saying so.
    """
    if "git" not in baseline or "inputs" not in baseline:
        return ["the baseline records no commit or input checksums; re-record it"]
    warnings: list[str] = []
    recorded, now = baseline["git"], _commit_metadata()
    if recorded["sha"] != now["sha"]:
        warnings.append(f"baseline rendered at commit {recorded['sha']}, current {now['sha']}")
    if recorded["dirty"] or now["dirty"]:
        warnings.append("tracked files differed from the commit at the baseline or differ now")
    recorded_inputs, current_inputs = baseline["inputs"], _input_digests()
    for ref in sorted(set(recorded_inputs) | set(current_inputs)):
        if recorded_inputs.get(ref) != current_inputs.get(ref):
            warnings.append(f"input differs from the baseline's: {ref}")
    return warnings


def _environment_warnings(
    expected: dict[str, object], observed: dict[str, object]
) -> list[str]:
    warnings: list[str] = []
    for key in ("python", "platform", "matplotlib", "py_hash_seed"):
        if expected.get(key) != observed.get(key):
            warnings.append(
                f"environment mismatch for {key}: "
                f"baseline={expected.get(key)!r}, current={observed.get(key)!r}"
            )
    return warnings


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
        for warning in [
            *_environment_warnings(baseline.get("environment", {}), _render_environment_metadata()),
            *_provenance_warnings(baseline),
        ]:
            print(f"WARN {warning}")
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            print(f"Candidate artifacts kept in {_relative_display(out_dir)}")
            return 1
        print("Figure baseline matches")
        print(f"Candidate artifacts kept in {_relative_display(out_dir)}")
        return 0
