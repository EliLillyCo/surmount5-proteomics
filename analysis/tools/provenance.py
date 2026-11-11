"""Provenance records: what produced a stage's outputs, and fingerprints of everything it touched.

``analysis/tools/run_stage.py`` runs the stages of ``analysis/manifest.yaml`` and, after each,
writes ``analysis/outputs/provenance/<stage>.json``; the figure and table builds write the
same kind of record for themselves (``record_build``); ``analysis/tools/provenance_check.py``
compares the records with the files on disk. This module holds what they share: the stage
list built from the manifest, reference resolution, fingerprints and the record itself.

A record holds the command, the git commit and whether the tree was dirty, the sha256 of
``pixi.lock``, the Python, R and platform versions, the wall-clock duration, the peak
resident set of the stage's largest process, and a fingerprint of every input and output the
manifest declares for the stage.

Fingerprints. A file of at most ``LARGE_FILE_BYTES`` is hashed in full (sha256). A larger file
is fingerprinted by its size, its modification time and its first and last ``SAMPLE_BYTES``; the
entry says so (``method``), and a change in the unread middle that leaves size and mtime alone
goes unseen. A directory records every file under it and a sha256 over the sorted
(relative path, size, file sha256) lines, so a changed, added or removed file changes it and
the check can name the file. Hashes are of the bytes: a rerun that writes the same table in
another row order or float rounding is a different fingerprint, which the check reports.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path, PurePosixPath

from analysis.external.pinned import sha256_file
from analysis.output_layout import PROVENANCE_DIR, ROOT
from analysis.tools.manifest_stages import MANIFEST, feeding_stages, load_manifest

PATHS_PREFIX = "paths.yaml:"
LOCKFILES = ("pixi.lock",)
# Set in the environment of a stage the runner starts: the runner writes that stage's record,
# so the figure and table builds do not write their own.
RUNNER_ENV = "SURMOUNT5_STAGE_RUNNER"

# A file above this size is fingerprinted by size, mtime and its head and tail.
LARGE_FILE_BYTES = 1024**3
SAMPLE_BYTES = 64 * 1024**2
METHOD_FULL = "sha256"
METHOD_SAMPLED = "size+mtime+head+tail sha256"
# Files hashed at once.
HASH_WORKERS = 16

AFTER_THE_FACT_NOTE = (
    "Recorded after the outputs were produced by a run this tool did not see: the fingerprints "
    "describe the files as they are now. This record does not show that they came from this "
    "code, this commit or these inputs."
)

_PLACEHOLDER = re.compile(r"<([^<>\s]+)>")
_BRACES = re.compile(r"\{([^{}]*)\}")
_GLOB_CHARS = set("*?[")


@dataclass(frozen=True)
class Stage:
    """One manifest stage, or the figure or table build, as the runner sees it."""

    id: str
    kind: str  # "analysis", "figures" or "tables"
    purpose: str
    command: str
    inputs: list[str]
    outputs: list[str]
    always: bool = False  # rebuilt on every `reproduce`, never skipped
    resumes: bool = False  # a rerun reuses its own earlier outputs when its guard allows


@dataclass(frozen=True)
class Workspace:
    """Where a run happens: the repository root, the path configuration and the manifest."""

    root: Path
    cfg: dict[str, str]
    manifest: dict
    provenance_dir: Path

    @classmethod
    def default(cls) -> Workspace:
        """The repository checkout this module lives in, with its ``paths.yaml`` and manifest."""
        from paths import load_paths_config

        return cls(ROOT, load_paths_config(), load_manifest(MANIFEST), PROVENANCE_DIR)


def load_stages(manifest: dict) -> list[Stage]:
    """Every stage in run order: the analysis stages, then the figure build, then the table build.

    The figure and table builds read the outputs of the stages whose ``downstream`` names them
    (``feeding_stages``) and the files their manifest entry lists under ``inputs``; together
    those are their inputs (``build_inputs``).

    Args:
        manifest: The loaded manifest.

    Returns:
        The stages in manifest order, the builds last and marked ``always``.
    """
    stages = [
        Stage(
            id=name,
            kind="analysis",
            purpose=spec["purpose"],
            command=spec["command"],
            inputs=list(spec.get("inputs", [])),
            outputs=list(spec["outputs"]),
            resumes=bool(spec.get("resumes", False)),
        )
        for name, spec in manifest["analysis_outputs"].items()
    ]
    bundle = build_spec(manifest, "figures")
    workbook = build_spec(manifest, "tables")
    for kind, spec, outputs in (
        ("figures", bundle, list(bundle["outputs"])),
        ("tables", workbook, [workbook["output"]]),
    ):
        stages.append(
            Stage(
                id=kind,
                kind=kind,
                purpose=spec["purpose"],
                command=spec["command"],
                inputs=build_inputs(manifest, kind),
                outputs=outputs,
                always=True,
            )
        )
    return stages


def build_spec(manifest: dict, kind: str) -> dict:
    """The manifest entry of the figure build (``figures``) or the table build (``tables``)."""
    if kind == "figures":
        return manifest["figure_bundles"]["all_figures"]
    return manifest["tables"]["supplementary_workbook"]


def build_inputs(manifest: dict, kind: str, scripts: list[str] = ()) -> list[str]:
    """References a figure or table build reads, in manifest order.

    These are the outputs of the stages whose ``downstream`` names the build, then the files
    its own manifest entry lists under ``inputs`` (files read directly, such as an ADaM table).

    Args:
        manifest: The loaded manifest.
        kind: ``"figures"`` or ``"tables"``.
        scripts: For ``figures``, the ``figures/<script>`` paths built; empty means all.

    Returns:
        Each reference once.
    """
    refs = [out for _, spec in feeding_stages(manifest, kind, scripts) for out in spec["outputs"]]
    refs += build_spec(manifest, kind).get("inputs", [])
    return list(dict.fromkeys(refs))


def placeholders(command: str) -> list[str]:
    """The ``<...>`` placeholders of a manifest command, without the brackets."""
    return _PLACEHOLDER.findall(command)


# ---------------------------------------------------------------------------
# References and fingerprints
# ---------------------------------------------------------------------------


@dataclass
class Resolved:
    """One path a manifest reference resolves to, or the reason it resolves to none."""

    ref: str
    path: Path | None
    status: str | None = None  # "placeholder", "unconfigured" or "missing"; None for a path that exists
    members: list[Path] = field(default_factory=list)


def _expand_braces(text: str) -> list[str]:
    match = _BRACES.search(text)
    if match is None:
        return [text]
    head, tail = text[: match.start()], text[match.end() :]
    return [
        expanded
        for option in match.group(1).split(",")
        for expanded in _expand_braces(head + option + tail)
    ]


def resolve(ref: str, ws: Workspace) -> list[Resolved]:
    """The paths a manifest input or output reference stands for.

    ``<name>`` is a placeholder the reader fills in; ``paths.yaml:<key>[/rest]`` is the value
    of that key in the path configuration (relative values are relative to the repository root)
    followed by ``rest``; anything else is relative to the repository root, or to the home
    directory when it starts with ``~``. ``{a,b}`` alternatives and ``*?[`` wildcards expand;
    a pattern or path that matches nothing resolves to one ``missing`` entry.

    Args:
        ref: The reference.
        ws: The workspace.

    Returns:
        One entry per path, sorted.
    """
    if ref.startswith("<"):
        return [Resolved(ref, None, "placeholder")]
    if ref.startswith(PATHS_PREFIX):
        key, _, rest = ref.removeprefix(PATHS_PREFIX).partition("/")
        value = ws.cfg.get(key)
        if not value:
            return [Resolved(ref, None, "unconfigured")]
        text = str(ws.root / value / rest) if rest else str(ws.root / value)
    else:
        text = str(ws.root / os.path.expanduser(ref))
    matches: list[str] = []
    for pattern in _expand_braces(text):
        if _GLOB_CHARS & set(pattern):
            matches += sorted(glob.glob(pattern))
        elif os.path.exists(pattern):
            matches.append(pattern)
    if not matches:
        return [Resolved(ref, Path(_expand_braces(text)[0]), "missing")]
    return [Resolved(ref, Path(m)) for m in sorted(set(matches))]


def hash_file(path: Path, *, contents: bool = True) -> dict:
    """Size, mtime and sha256 of one file (``LARGE_FILE_BYTES`` and ``SAMPLE_BYTES`` apply).

    Args:
        path: The file.
        contents: False records size and mtime only.

    Returns:
        ``bytes``, ``mtime_ns`` and, when ``contents``, ``method`` and ``sha256``
        (``sample_bytes`` too for a sampled file).
    """
    stat = path.stat()
    entry: dict = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not contents:
        return entry
    if stat.st_size <= LARGE_FILE_BYTES:
        return {**entry, "method": METHOD_FULL, "sha256": sha256_file(path)}
    h = hashlib.sha256(f"{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
    with path.open("rb") as fh:
        h.update(fh.read(SAMPLE_BYTES))
        fh.seek(-SAMPLE_BYTES, os.SEEK_END)
        h.update(fh.read(SAMPLE_BYTES))
    return {**entry, "method": METHOD_SAMPLED, "sample_bytes": SAMPLE_BYTES, "sha256": h.hexdigest()}


def _directory_digest(files: list[dict]) -> str:
    listing = hashlib.sha256()
    for f in files:
        listing.update(f"{f['path']}\t{f['bytes']}\t{f.get('sha256', '')}\n".encode())
    return listing.hexdigest()


def snapshot(refs: list[str], ws: Workspace, *, contents: bool = True) -> list[dict]:
    """Fingerprint every path the references resolve to.

    Files are hashed in parallel (``HASH_WORKERS``).

    Args:
        refs: Manifest input or output references.
        ws: The workspace.
        contents: False records size and mtime only (the quick check).

    Returns:
        One entry per resolved path: ``ref``, ``path`` and ``kind`` (file, directory, missing,
        placeholder or unconfigured), plus ``relpath`` under the repository root, the file
        fields of ``hash_file``, or for a directory ``n_files``, ``bytes``, ``sha256`` and a
        ``files`` list of relative paths with their file fields.
    """
    items = [item for ref in refs for item in resolve(ref, ws)]
    for item in items:
        if item.status is None and item.path.is_dir():
            item.members = sorted(p for p in item.path.rglob("*") if p.is_file())
    to_hash = sorted(
        {m for i in items for m in i.members}
        | {i.path for i in items if i.status is None and not i.path.is_dir()}
    )
    with ThreadPoolExecutor(HASH_WORKERS) as pool:
        hashed = dict(zip(to_hash, pool.map(lambda p: hash_file(p, contents=contents), to_hash)))
    entries = []
    for item in items:
        entry: dict = {"ref": item.ref}
        if item.path is not None:
            entry["path"] = str(item.path)
            if item.path.is_relative_to(ws.root):
                entry["relpath"] = str(item.path.relative_to(ws.root))
        if item.status is not None:
            entries.append({**entry, "kind": item.status})
        elif item.path.is_dir():
            files = [{"path": str(m.relative_to(item.path)), **hashed[m]} for m in item.members]
            entry.update(kind="directory", n_files=len(files), bytes=sum(f["bytes"] for f in files))
            if contents:
                entry["sha256"] = _directory_digest(files)
            entries.append({**entry, "files": files})
        else:
            entries.append({**entry, "kind": "file", **hashed[item.path]})
    return entries


def _label(entry: dict) -> str:
    return entry.get("relpath") or entry.get("path") or entry["ref"]


def _match_key(entry: dict) -> tuple[str, str]:
    """A recorded or current match's identity, independent of where the base sits.

    A reference resolves to ``<base>/<rest>``, so the last components of the path that ``rest``
    spells out are the same under any checkout or ``paths.yaml`` root. Assumes no ``{a,b}``
    alternative in a manifest reference contains a ``/`` (the component count is then fixed).
    """
    ref = entry["ref"]
    if "path" not in entry:
        return ref, ""
    if ref.startswith(PATHS_PREFIX):
        rest = ref.removeprefix(PATHS_PREFIX).partition("/")[2]
    else:
        rest = ref.removeprefix("~/")
    n = len(PurePosixPath(rest).parts)
    return ref, "/".join(Path(entry["path"]).parts[-n:]) if n else ""


def _same_file(recorded: dict, now: dict, quick: bool) -> str | None:
    """Why two file fingerprints differ, or None when they agree."""
    if quick:
        if (recorded["bytes"], recorded["mtime_ns"]) != (now["bytes"], now["mtime_ns"]):
            return "size or modification time changed"
        return None
    if recorded["bytes"] != now["bytes"]:
        return f"content changed (size {recorded['bytes']} -> {now['bytes']})"
    if recorded["sha256"] != now["sha256"]:
        method = "" if recorded["method"] == METHOD_FULL else f" ({recorded['method']})"
        return f"content changed{method}"
    return None


def mismatches(recorded: list[dict], ws: Workspace, *, quick: bool = False) -> list[str]:
    """Where the files the references resolve to today differ from a recorded fingerprint list.

    Args:
        recorded: ``inputs`` or ``outputs`` of a record.
        ws: The workspace.
        quick: Compare size and modification time instead of hashing.

    Returns:
        One message per differing file; empty when everything matches. Placeholders and
        unconfigured paths cannot be compared and are skipped.
    """
    refs = list(dict.fromkeys(e["ref"] for e in recorded))
    now_by_path = {
        _match_key(e): e for e in snapshot(refs, ws, contents=not quick)
    }
    problems: list[str] = []
    seen: set[tuple] = set()
    for entry in recorded:
        key = _match_key(entry)
        seen.add(key)
        kind = entry["kind"]
        if kind in ("placeholder", "unconfigured"):
            continue
        now = now_by_path.get(key)
        label = _label(entry)
        if kind == "missing":
            if now is not None and now["kind"] != "missing":
                problems.append(f"{label}: now present (it was missing when recorded)")
        elif now is None or now["kind"] == "missing":
            problems.append(f"{label}: missing")
        elif now["kind"] != kind:
            problems.append(f"{label}: was a {kind}, is now a {now['kind']}")
        elif kind == "file":
            reason = _same_file(entry, now, quick)
            if reason:
                problems.append(f"{label}: {reason}")
        else:
            problems += _directory_mismatches(entry, now, label, quick)
    for key, now in now_by_path.items():
        if key not in seen and now["kind"] != "missing":
            problems.append(f"{_label(now)}: not in the record (a new match of {now['ref']})")
    return problems


def _directory_mismatches(recorded: dict, now: dict, label: str, quick: bool) -> list[str]:
    old = {f["path"]: f for f in recorded["files"]}
    new = {f["path"]: f for f in now["files"]}
    problems = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            problems.append(f"{label}/{name}: missing")
        elif name not in old:
            problems.append(f"{label}/{name}: not in the record (added)")
        else:
            reason = _same_file(old[name], new[name], quick)
            if reason:
                problems.append(f"{label}/{name}: {reason}")
    return problems


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def git_state(root: Path) -> dict:
    """Commit, branch and tracked-file changes of the checkout; nulls outside a git checkout.

    Args:
        root: The repository root.

    Returns:
        ``sha``, ``branch``, ``dirty`` and ``dirty_paths``; untracked files do not count.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout

    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=root, capture_output=True).returncode:
        return {"sha": None, "branch": None, "dirty": None, "dirty_paths": None}
    changed = [
        line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
    ]
    return {
        "sha": git("rev-parse", "HEAD").strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(changed),
        "dirty_paths": changed,
    }


@cache
def r_version() -> str:
    """The ``R version ...`` line of ``R --version`` for the R beside the running Python (the pixi environment)."""
    out = subprocess.run(
        [str(Path(sys.prefix) / "bin" / "R"), "--version"], check=True, capture_output=True, text=True
    )
    return next(line for line in out.stdout.splitlines() if line.startswith("R version"))


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def environment(root: Path) -> dict:
    """The fields of a record that describe the code and machine, not the stage."""
    return {
        "git": git_state(root),
        "locks": {name: sha256_file(root / name) for name in LOCKFILES},
        "python": platform.python_version(),
        "r": r_version(),
        "platform": platform.platform(),
    }


def write_record(payload: dict, name: str, ws: Workspace) -> Path:
    """Write ``<provenance_dir>/<name>.json``.

    Returns:
        The path written.
    """
    ws.provenance_dir.mkdir(parents=True, exist_ok=True)
    path = ws.provenance_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def stage_record(
    stage: Stage,
    ws: Workspace,
    *,
    inputs: list[dict],
    outputs: list[dict],
    command: list[str] | None,
    started: float | None,
    finished: float,
    peak_rss_mb: float | None,
    after_the_fact: bool,
    inputs_fingerprinted: str,
    missing_outputs: list[str] = (),
    name: str | None = None,
    arguments: dict | None = None,
) -> dict:
    """The record of one stage (the JSON the runner writes), without writing it."""
    payload = {
        "stage": stage.id,
        "kind": stage.kind,
        "purpose": stage.purpose,
        "command": command,
        "manifest_command": stage.command,
        "resumes": stage.resumes,
        "started": _iso(started if started is not None else finished),
        "finished": _iso(finished),
        "duration_s": round(finished - started, 3) if started is not None else None,
        "peak_rss_mb": peak_rss_mb,
        **environment(ws.root),
        "recorded_after_the_fact": after_the_fact,
        "inputs_fingerprinted": inputs_fingerprinted,
        "outputs_missing_when_recorded": list(missing_outputs),
        "inputs": inputs,
        "outputs": outputs,
    }
    if name is not None:
        payload["name"] = name
    if arguments is not None:
        payload["arguments"] = arguments
    if after_the_fact:
        payload["note"] = AFTER_THE_FACT_NOTE
    return payload


def record_build(
    kind: str,
    *,
    outputs: list[str],
    scripts: list[str],
    started: float,
    ws: Workspace | None = None,
    name: str | None = None,
    arguments: dict | None = None,
) -> Path | None:
    """Write the record of a figure or table build run on its own (not under the runner).

    The build's inputs are the outputs of the stages whose ``downstream`` names it and the
    files its manifest entry lists under ``inputs`` (``build_inputs``); they are fingerprinted
    when the build has finished, which is when they are known not to have been written by it.
    The runner writes its own record for a build it runs, so under ``RUNNER_ENV`` this does
    nothing.

    Args:
        kind: ``"figures"`` or ``"tables"``.
        outputs: References or paths of the files the build wrote.
        scripts: For ``figures``, the ``figures/<script>`` paths built.
        started: Start time (epoch seconds).
        ws: The workspace; the checkout's own when omitted.
        name: Record file stem; the kind when omitted.
        arguments: The build's arguments, kept in the record.

    Returns:
        The record's path, or None under the runner.
    """
    if os.environ.get(RUNNER_ENV):
        return None
    ws = ws or Workspace.default()
    stage = next(s for s in load_stages(ws.manifest) if s.id == kind)
    finished = time.time()
    payload = stage_record(
        stage,
        ws,
        inputs=snapshot(build_inputs(ws.manifest, kind, scripts), ws),
        outputs=snapshot(outputs, ws),
        command=[sys.executable, *sys.argv],
        started=started,
        finished=finished,
        peak_rss_mb=None,
        after_the_fact=False,
        inputs_fingerprinted="after the run",
        name=name or kind,
        arguments=arguments,
    )
    return write_record(payload, name or kind, ws)
