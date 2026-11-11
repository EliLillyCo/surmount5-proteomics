"""Run manifest stages and record what produced each output.

    pixi run stage run <id> [<id> ...]                  # the named stages, in the order given
    pixi run stage record <id> [--allow-missing REF]    # a stage run by hand
    pixi run reproduce [--force] [--dry-run]            # run_stage.py all
    pixi run paper                                      # figures and tables from existing outputs

A stage's command is the ``command`` ``analysis/manifest.yaml`` gives it, run through
``pixi run --frozen`` from the repository root with each ``<paths.yaml:key>`` filled from the
path configuration. A ``<placeholder>`` of any other kind (a checkout or a browser download
only you have) cannot be filled: run that stage by hand and record it with ``stage record``.

After each stage that succeeds the runner writes ``analysis/outputs/provenance/<stage>.json``
(``analysis/tools/provenance.py`` describes the record): the command, git commit and dirty
flag, ``pixi.lock`` digest, Python, R and platform versions, duration, peak memory and a
fingerprint of every declared input (taken before the run) and output. The record of a stage
that fails, or that does not write a declared output, is removed and nothing is written.

``all`` runs every stage in manifest order and stops at the first failure. A stage is skipped
when it is up to date: its record exists, every declared output exists, its declared inputs
match the fingerprints in the record, and no stage it reads from ran in this invocation. The figure and table builds always rebuild. ``--force`` runs every
stage. Mediation resumes from its own fingerprint-guarded output (``resumes: true`` in the
manifest); ``--force`` does not delete it, so a from-scratch mediation run starts by removing
its CSV and checkpoints.

``reproduce`` does not check stage code before skipping. Run ``pixi run provenance-check``,
rerun reported stages with ``pixi run stage run <stage>``, then run
``pixi run provenance-check`` again.

``record`` fingerprints the files as they are and says so in the record
(``recorded_after_the_fact``): it shows what the outputs were when recorded, not that this
code produced them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from analysis.tools import provenance as prov
from analysis.tools.provenance import Stage, Workspace
from analysis.tools.manifest_stages import stage_consumers

# Runs a stage's pixi command in the locked environment from any shell.
FROZEN = ("--frozen",)


def _workspace(ws: Workspace | None) -> Workspace:
    return ws or Workspace.default()


def stages_by_id(ws: Workspace) -> dict[str, Stage]:
    """The stages of the manifest by id, in run order."""
    return {s.id: s for s in prov.load_stages(ws.manifest)}


def select(ids: list[str], ws: Workspace) -> list[Stage]:
    """The stages named, in the order given.

    Raises:
        KeyError: When an id is not a stage.
    """
    stages = stages_by_id(ws)
    for stage_id in ids:
        if stage_id not in stages:
            raise KeyError(f"{stage_id!r} is not a stage; the stages are {', '.join(stages)}")
    return [stages[i] for i in ids]


def stage_argv(stage: Stage, ws: Workspace) -> list[str]:
    """The argument vector that runs a stage: its command with ``pixi run`` made ``--frozen``.

    Args:
        stage: The stage.
        ws: The workspace (the path configuration fills ``<paths.yaml:key>``).

    Returns:
        The command words.

    Raises:
        ValueError: When the command holds a placeholder that is not a configured ``paths.yaml`` key.
    """
    words = []
    for word in shlex.split(stage.command):
        for token in prov.placeholders(word):
            key = token.removeprefix(prov.PATHS_PREFIX)
            if not token.startswith(prov.PATHS_PREFIX) or not ws.cfg.get(key):
                raise ValueError(
                    f"stage {stage.id}: the command has <{token}>, which only you can fill in; "
                    f"run it by hand, then `stage record {stage.id}`"
                )
            word = word.replace(f"<{token}>", ws.cfg[key])
        words.append(word)
    for i in range(len(words) - 1):
        if words[i : i + 2] == ["pixi", "run"]:
            return [*words[: i + 2], *FROZEN, *words[i + 2 :]]
    return words


def _missing(entries: list[dict]) -> list[str]:
    return [e["ref"] for e in entries if e["kind"] in ("missing", "unconfigured")]


def outputs_present(stage: Stage, ws: Workspace) -> bool:
    """Whether every declared output resolves to an existing path."""
    return not _missing(prov.snapshot(stage.outputs, ws, contents=False))


def up_to_date(stage: Stage, ws: Workspace) -> bool:
    """Whether a stage can be skipped.

    It can when it is not an ``always`` stage, its record and every declared output exist,
    and its declared inputs are as the record fingerprinted them (``provenance.mismatches``,
    contents hashed). A record whose input list differs from the manifest's is not up to date.

    Args:
        stage: The stage.
        ws: The workspace.

    Returns:
        True when the stage can be skipped.
    """
    if stage.always:
        return False
    record_path = ws.provenance_dir / f"{stage.id}.json"
    if not record_path.exists() or not outputs_present(stage, ws):
        return False
    recorded = json.loads(record_path.read_text())["inputs"]
    if list(dict.fromkeys(e["ref"] for e in recorded)) != list(dict.fromkeys(stage.inputs)):
        return False
    return not prov.mismatches(recorded, ws)


def _run_command(argv: list[str], ws: Workspace) -> tuple[int, float]:
    """Run ``argv`` and return (exit code, peak resident set in MB of its largest process)."""
    proc = subprocess.Popen(argv, cwd=ws.root, env={**os.environ, prov.RUNNER_ENV: "1"})
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, round(usage.ru_maxrss / 1024, 1)  # Linux reports kilobytes


def run_stage(stage: Stage, ws: Workspace) -> int:
    """Run one stage and write its record.

    Args:
        stage: The stage.
        ws: The workspace.

    Returns:
        0 on success, else the command's exit code.

    Raises:
        FileNotFoundError: When the command succeeds but a declared output is absent.
    """
    argv = stage_argv(stage, ws)
    record_path = ws.provenance_dir / f"{stage.id}.json"
    record_path.unlink(missing_ok=True)
    print(f"==> {stage.id}: {' '.join(argv)}", flush=True)
    inputs = prov.snapshot(stage.inputs, ws)
    started = time.time()
    code, peak_rss_mb = _run_command(argv, ws)
    finished = time.time()
    if code:
        print(f"==> {stage.id} failed (exit {code}); no record written", file=sys.stderr, flush=True)
        return code
    outputs = prov.snapshot(stage.outputs, ws)
    absent = _missing(outputs)
    if absent:
        raise FileNotFoundError(f"stage {stage.id} exited 0 but did not write: {', '.join(absent)}")
    payload = prov.stage_record(
        stage,
        ws,
        inputs=inputs,
        outputs=outputs,
        command=argv,
        started=started,
        finished=finished,
        peak_rss_mb=peak_rss_mb,
        after_the_fact=False,
        inputs_fingerprinted="before the run",
    )
    path = prov.write_record(payload, stage.id, ws)
    print(f"==> {stage.id} done in {finished - started:.0f} s; provenance -> {_display(path, ws)}", flush=True)
    return 0


def _display(path: Path, ws: Workspace) -> Path:
    return path.relative_to(ws.root) if path.is_relative_to(ws.root) else path


def run_many(ids: list[str], *, ws: Workspace | None = None) -> int:
    """Run the named stages in the order given, stopping at the first failure.

    Args:
        ids: Stage ids; all are validated, and their commands filled, before any runs.
        ws: The workspace; the checkout's own when omitted.

    Returns:
        0 when every stage succeeded, else the exit code of the first failure.

    Raises:
        KeyError: When an id is not a stage.
        ValueError: When a command has a placeholder that cannot be filled.
    """
    ws = _workspace(ws)
    stages = select(ids, ws)
    for stage in stages:
        stage_argv(stage, ws)
    for stage in stages:
        code = run_stage(stage, ws)
        if code:
            return code
    return 0


def record_many(
    ids: list[str], *, ws: Workspace | None = None, allow_missing: list[str] = ()
) -> int:
    """Write the record of stages that were run by hand.

    Args:
        ids: Stage ids.
        ws: The workspace; the checkout's own when omitted.
        allow_missing: Declared outputs that may be absent; each is recorded as missing
            instead of raising.

    Returns:
        0.

    Raises:
        KeyError: When an id is not a stage.
        ValueError: When an ``allow_missing`` reference is no named stage's declared output.
        FileNotFoundError: When another declared output is absent.
    """
    ws = _workspace(ws)
    stages = select(ids, ws)
    declared = {ref for s in stages for ref in s.outputs}
    unknown = [ref for ref in allow_missing if ref not in declared]
    if unknown:
        raise ValueError(f"{', '.join(unknown)}: not a declared output of {', '.join(ids)}")
    for stage in stages:
        outputs = prov.snapshot(stage.outputs, ws)
        expected = [ref for ref in allow_missing if ref in stage.outputs]
        absent = [ref for ref in _missing(outputs) if ref not in expected]
        if absent:
            raise FileNotFoundError(f"stage {stage.id}: declared outputs not found: {', '.join(absent)}")
        payload = prov.stage_record(
            stage,
            ws,
            inputs=prov.snapshot(stage.inputs, ws),
            outputs=outputs,
            command=None,
            started=None,
            finished=time.time(),
            peak_rss_mb=None,
            after_the_fact=True,
            inputs_fingerprinted="after the fact",
            missing_outputs=_missing(outputs),
        )
        path = prov.write_record(payload, stage.id, ws)
        print(f"==> {stage.id}: recorded after the fact -> {_display(path, ws)}", flush=True)
    return 0


def plan(ws: Workspace, *, force: bool) -> list[tuple[Stage, bool]]:
    """Every stage in order with whether it runs: not up to date, or after a stage it reads ran.

    Args:
        ws: The workspace.
        force: Run every stage.

    Returns:
        ``(stage, runs)`` pairs.
    """
    named = stage_consumers(ws.manifest)
    ran: set[str] = set()
    outputs_of: dict[str, set[str]] = {}
    decided = []
    for stage in prov.load_stages(ws.manifest):
        reads = set(named.get(stage.id, [])) | {
            other for other, outputs in outputs_of.items() if outputs & set(stage.inputs)
        }
        runs = force or not up_to_date(stage, ws) or bool(reads & ran)
        outputs_of[stage.id] = set(stage.outputs)
        if runs:
            ran.add(stage.id)
        decided.append((stage, runs))
    return decided


def run_all(*, ws: Workspace | None = None, force: bool = False, dry_run: bool = False) -> int:
    """Every stage in manifest order, skipping the up-to-date ones unless ``force``.

    Args:
        ws: The workspace; the checkout's own when omitted.
        force: Run every stage.
        dry_run: Print what would run and run nothing.

    Returns:
        0 when every stage succeeded or was skipped, else the first failing exit code.

    Raises:
        ValueError: When a stage that would run has a placeholder that cannot be filled.
    """
    ws = _workspace(ws)
    if not force:
        print("Checking each stage's recorded inputs (contents are hashed) ...", flush=True)
    decided = plan(ws, force=force)
    for stage, runs in decided:
        if runs:
            stage_argv(stage, ws)
    for stage, runs in decided:
        if not runs:
            print(f"==> {stage.id}: up to date, skipped", flush=True)
        elif dry_run:
            print(f"==> {stage.id}: would run", flush=True)
        else:
            code = run_stage(stage, ws)
            if code:
                return code
    return 0


def main(argv: list[str] | None = None, *, ws: Workspace | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run stages in the order given, recording each")
    p_run.add_argument("stage_ids", nargs="+", metavar="id")
    p_record = sub.add_parser("record", help="record provenance for stages already run")
    p_record.add_argument("stage_ids", nargs="+", metavar="id")
    p_record.add_argument(
        "--allow-missing",
        action="append",
        default=[],
        metavar="REF",
        help="a declared output that is known to be absent; recorded as missing",
    )
    p_all = sub.add_parser("all", help="run every stage in manifest order")
    p_all.add_argument("--force", action="store_true", help="run stages that are up to date too")
    p_all.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_many(args.stage_ids, ws=ws)
    if args.command == "record":
        return record_many(args.stage_ids, ws=ws, allow_missing=args.allow_missing)
    return run_all(ws=ws, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
