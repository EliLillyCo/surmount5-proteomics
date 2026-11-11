"""Do the files still match the provenance records?

    pixi run provenance-check [--quick]

For every ``analysis/outputs/provenance/*.json``, fingerprint the stage's declared inputs and
outputs as they are now and compare them with the record (``analysis/tools/provenance.py``): a
changed, missing or added file is reported by name and makes the record stale. A stale
output means the file is not what the record describes; a stale input means the stage was
recorded against other inputs than the ones on disk now (and, when an upstream stage reran,
that this stage has to as well). Exit status 1 when any record is stale.

Every manifest stage (the figure and table builds included) with no record is reported too (a
record counts for the stage its ``stage`` field names, so ``figures.main.json`` counts for
``figures``). When any of its declared outputs exists it is a failure: those files have no record of what
produced them. When none exists it is a note: the stage has not run here. The check also exits 1
when there are no records at all.

``--quick`` compares size and modification time instead of hashing, for a look at mounted data
that would otherwise take a while to read.

A record is also stale when the stage's code changed after it was made. The code is the
stage script's import closure (``analysis/tools/import_closure.py``; for the figure and table
builds also the registered figure scripts and table modules). A file of the closure that
differs between the record's commit and HEAD makes the record stale, except a Python file
whose two versions parse to the same AST once docstrings are dropped (comment and docstring
edits); any other change, and any added, removed or non-Python closure file that differs, counts.
A record made in a dirty tree is compared with its commit, which is said in a note.

Notes that do not fail the check: the commit the record was made at differs from HEAD, the
tree was dirty then, comment-only edits to the closure, ``pixi.lock`` has changed since, or
the record was written after the fact (``stage record``) and so does not show that the code
produced the outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import shlex
import subprocess

from analysis.tools import provenance as prov
from analysis.tools.import_closure import closure
from analysis.output_layout import ROOT
from analysis.tools.manifest_stages import BUILD_FIGURES, BUILD_TABLES, registered_figures, registered_tables
from analysis.tools.provenance import Stage, Workspace

BUILD_FIGURES_REL = BUILD_FIGURES.relative_to(ROOT).as_posix()
BUILD_TABLES_REL = BUILD_TABLES.relative_to(ROOT).as_posix()


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the docstring statement of every module, class and function body.

    Args:
        tree: Parsed module; modified in place.

    Returns:
        The same tree, without docstring statements.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del node.body[0]
    return tree


def ast_equivalent(source_a: str, source_b: str) -> bool:
    """Whether two Python sources parse to the same AST once docstrings are dropped.

    Args:
        source_a: First Python source text.
        source_b: Second Python source text.

    Returns:
        True when the stripped ASTs are equal; comments never reach the AST.
    """
    dumps = [ast.dump(strip_docstrings(ast.parse(src))) for src in (source_a, source_b)]
    return dumps[0] == dumps[1]


def _git(root, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def changed_files(root, sha: str, head: str) -> set[str]:
    """Repo-relative paths that differ between two commits.

    Args:
        root: The repository root.
        sha: Earlier commit.
        head: Later commit.

    Returns:
        The paths ``git diff --name-only`` reports.
    """
    result = _git(root, "diff", "--name-only", sha, head)
    result.check_returncode()
    return {line for line in result.stdout.splitlines() if line}


def source_at(root, sha: str, path: str) -> str | None:
    """The file's text at a commit; None when the file does not exist there.

    Args:
        root: The repository root.
        sha: Commit to read from.
        path: Repo-relative file path.

    Returns:
        The file text, or None if absent at that commit.
    """
    result = _git(root, "show", f"{sha}:{path}")
    return result.stdout if result.returncode == 0 else None


def stage_roots(stage: Stage, ws: Workspace) -> set[str]:
    """The stage script, declared child scripts, and files loaded by the builds.

    The script is the first ``.py`` or ``.R`` token of the manifest command, or, after ``python -m``,
    the module's file (a package's ``__main__.py``). An analysis stage can declare child
    scripts in ``code_roots`` when subprocess calls hide them from its import closure.
    Figure and table builds also include their registered scripts and modules.

    Args:
        stage: The manifest stage.
        ws: The workspace.

    Returns:
        Repo-relative paths.
    """
    parts = shlex.split(stage.command)
    if "-m" in parts:
        rel = parts[parts.index("-m") + 1].replace(".", "/")
        script = f"{rel}/__main__.py" if (ws.root / rel / "__main__.py").is_file() else f"{rel}.py"
    else:
        script = next(part for part in parts if part.endswith((".py", ".R")))
    roots = {script, *ws.manifest.get("analysis_outputs", {}).get(stage.id, {}).get("code_roots", [])}
    if script == BUILD_FIGURES_REL:
        for group in registered_figures(ws.root / script).values():
            roots |= {f"figures/{name}" for name, _ in group}
    if script == BUILD_TABLES_REL:
        roots |= {f"tables/{row[2]}.py" for row in registered_tables(ws.root / script)}
    return roots


def code_changes(record: dict, ws: Workspace) -> tuple[list[str], list[str]]:
    """The stage's code files that differ between the record's commit and HEAD.

    Args:
        record: Parsed record; its ``git.sha`` is the commit compared.
        ws: The workspace.

    Returns:
        ``(behavior, comment_only)``. ``behavior`` has one message per file of the stage's
        import closure that differs and is not a comment- or docstring-only edit of a Python
        file (``no commit recorded`` when the record has no commit); ``comment_only`` lists
        the closure files whose edits touch only comments and docstrings. Both are empty
        when the code is unchanged.
    """
    sha = record["git"].get("sha")
    if not sha:
        return ["no commit recorded, so the code cannot be compared"], []
    head = prov.git_state(ws.root)["sha"]
    if sha == head:
        return [], []
    stage = next(s for s in prov.load_stages(ws.manifest) if s.id == record["stage"])
    files = closure(stage_roots(stage, ws), ws.root) & changed_files(ws.root, sha, head)
    behavior, comment_only = [], []
    for path in sorted(files):
        before, after = source_at(ws.root, sha, path), source_at(ws.root, head, path)
        if before is None or after is None:
            behavior.append(f"{path} {'added' if before is None else 'removed'} since {sha[:7]}")
        elif path.endswith(".py") and ast_equivalent(before, after):
            comment_only.append(path)
        else:
            behavior.append(f"{path} changed since {sha[:7]}")
    return behavior, comment_only


def notes(record: dict, ws: Workspace) -> list[str]:
    """Why a record is weaker evidence than it looks, or out of step with the checkout.

    Args:
        record: Parsed record.
        ws: The workspace.

    Returns:
        One message per note; empty when none applies.
    """
    now = prov.git_state(ws.root)
    out = []
    if record.get("recorded_after_the_fact"):
        out.append("recorded after the fact: it does not show that this code produced the outputs")
    recorded = record["git"]
    if recorded.get("dirty"):
        out.append(
            f"recorded from a dirty tree ({', '.join(recorded['dirty_paths'])}); its code is "
            f"compared with commit {str(recorded['sha'])[:7]}, not with the dirty files"
        )
    if recorded.get("sha") and recorded["sha"] != now["sha"]:
        out.append(f"recorded at {recorded['sha'][:7]}, HEAD is {now['sha'][:7]}")
    comment_only = code_changes(record, ws)[1]
    if comment_only:
        out.append(f"code changed only in comments and docstrings: {', '.join(comment_only)}")
    for name, digest in record["locks"].items():
        if digest != prov.sha256_file(ws.root / name):
            out.append(f"{name} differs from the record")
    return out


def check_record(record: dict, ws: Workspace, *, quick: bool) -> dict[str, list[str]]:
    """Mismatches of one record with the files and the code now.

    Args:
        record: Parsed record.
        ws: The workspace.
        quick: Compare size and mtime instead of hashing.

    Returns:
        ``{"outputs": [...], "inputs": [...], "code": [...]}`` messages; all empty when the
        record holds.
    """
    stage = next(s for s in prov.load_stages(ws.manifest) if s.id == record["stage"])
    recorded_refs = {entry["ref"] for entry in record["inputs"]}
    added_inputs = [
        f"{ref}: not in the record (newly declared input)"
        for ref in stage.inputs
        if ref not in recorded_refs
    ]
    return {
        "outputs": prov.mismatches(record["outputs"], ws, quick=quick),
        "inputs": prov.mismatches(record["inputs"], ws, quick=quick) + added_inputs,
        "code": code_changes(record, ws)[0],
    }


def unrecorded(
    ws: Workspace, records: list[dict]
) -> tuple[list[tuple[Stage, list[str]]], list[Stage]]:
    """Manifest stages no record names, split by whether any of their outputs exists.

    Args:
        ws: The workspace.
        records: The parsed records; each counts for the stage its ``stage`` field names.

    Returns:
        ``(with_outputs, without_outputs)``: the first pairs each stage with the declared
        output references that resolve to an existing path; the second holds the stages none
        of whose outputs exists. Both in run order.
    """
    recorded = {record["stage"] for record in records}
    with_outputs: list[tuple[Stage, list[str]]] = []
    without_outputs: list[Stage] = []
    for stage in prov.load_stages(ws.manifest):
        if stage.id in recorded:
            continue
        present = list(
            dict.fromkeys(
                e["ref"]
                for e in prov.snapshot(stage.outputs, ws, contents=False)
                if e["kind"] in ("file", "directory")
            )
        )
        if present:
            with_outputs.append((stage, present))
        else:
            without_outputs.append(stage)
    return with_outputs, without_outputs


def main(argv: list[str] | None = None, *, ws: Workspace | None = None) -> int:
    """Print one block per record and one line per unrecorded stage; exit 1 on any failure.

    A failure is a stale record, a stage with outputs but no record, or no records at all.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="compare size and mtime, no hashing")
    args = parser.parse_args(argv)
    ws = ws or Workspace.default()
    paths = sorted(ws.provenance_dir.glob("*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    with_outputs, without_outputs = unrecorded(ws, records)
    for stage, present in with_outputs:
        print(f"{stage.id}: NO RECORD, but its outputs exist ({', '.join(present)})")
    for stage in without_outputs:
        print(f"{stage.id}: note: no record and no outputs (not run here)")
    if not paths:
        print(f"no provenance records under {ws.provenance_dir}")
        return 1
    n_stale = 0
    for path, record in zip(paths, records):
        problems = check_record(record, ws, quick=args.quick)
        stale = any(problems.values())
        n_stale += stale
        print(f"{path.stem}: {'STALE' if stale else 'ok'}")
        for side, messages in problems.items():
            for message in messages:
                print(f"  {side}: {message}")
        for note in notes(record, ws):
            print(f"  note: {note}")
    if n_stale:
        print(f"{n_stale} of {len(paths)} record(s) stale")
    if with_outputs:
        print(f"{len(with_outputs)} stage(s) with outputs but no record")
    if not n_stale and not with_outputs:
        print("provenance OK")
    return 1 if n_stale or with_outputs else 0


if __name__ == "__main__":
    raise SystemExit(main())
