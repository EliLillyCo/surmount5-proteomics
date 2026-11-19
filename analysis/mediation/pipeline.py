"""Parallel mediation pipeline: per-visit linear models with a nonparametric bootstrap."""

from __future__ import annotations

import gc
import hashlib
import json
import tempfile
from pathlib import Path

import polars as pl
from tqdm import tqdm

from analysis.mediation.constants import (
    CAT_COVARIATES,
    CONTROL_VALUE,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    TREAT_VALUE,
    VISIT_COL,
)
from analysis.mediation.core import run_mediation_single_numpy
from analysis.parallel import default_chunksize, forkserver_pool, resolve_n_jobs


def _mediation_fingerprint(
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    sims: int,
    response_var: str,
    response_var_bl: str,
    protein_id_col: str,
    control_value: str,
    treat_value: str,
) -> str:
    """Stable hash of the model config a mediation checkpoint was fit under.

    The hash covers the mediator, the covariates, the number of draws, the
    response and protein ID columns and the arm values.
    """
    payload = json.dumps(
        {
            "mediator": mediator,
            "cat_covs": sorted(cat_covs),
            "cont_covs": sorted(cont_covs),
            "sims": sims,
            "response_var": response_var,
            "response_var_bl": response_var_bl,
            "protein_id_col": protein_id_col,
            "control_value": control_value,
            "treat_value": treat_value,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    """SHA-256 of a frame's content, independent of row and column order.

    Columns are taken in name order and rows sorted by every column, so two frames with
    the same rows hash equal however they were assembled. The schema, the null masks and
    the exact float bits all enter the hash.
    """
    columns = sorted(frame.columns)
    frame = frame.select(columns).sort(columns)
    digest = hashlib.sha256()
    digest.update(json.dumps([[c, str(frame.schema[c])] for c in columns]).encode())
    digest.update(str(frame.height).encode())
    for c in columns:
        series = frame[c]
        digest.update(c.encode())
        digest.update(series.is_null().to_numpy().tobytes())
        if series.dtype.is_numeric() or series.dtype == pl.Boolean:
            digest.update(series.fill_null(0).to_numpy().tobytes())
        else:
            joined = series.cast(pl.String).fill_null("").str.join("\x1f").item()
            digest.update(joined.encode())
    return digest.hexdigest()


def _marker_visit_keys(frame: pl.DataFrame) -> set[tuple]:
    """The (marker, visit) keys of a mediation output."""
    return set(frame.select(pl.col("marker").cast(pl.String), "visit").iter_rows())


def _keys_fingerprint(keys: set[tuple]) -> str:
    """SHA-256 of a sorted (marker, visit) key set."""
    return hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()


def _visit_input_fingerprint(
    data: pl.DataFrame,
    protein_id_col: str,
    visit: int,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var: str,
    response_var_bl: str,
) -> str:
    """Fingerprint of exactly the rows and columns a visit's fits consume.

    The frame is ``_prepare_visit_frame``'s (the visit's complete cases on the model
    variables), so a changed value, participant or protein changes the fingerprint and
    a reordered input does not.
    """
    return _frame_fingerprint(
        _prepare_visit_frame(
            data,
            protein_id_col,
            visit,
            mediator,
            cat_covs,
            cont_covs,
            response_var,
            response_var_bl,
        )
    )


def _visit_protein_set(
    data: pl.DataFrame,
    protein_id_col: str,
    visit: int,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var: str,
    response_var_bl: str,
) -> set:
    """The complete-case protein set a visit's run would fit today.

    Mirrors the filtering in ``run_mediation_parallel`` (visit + finite-value
    complete case) without running any mediation — cheap enough to use purely
    as a congruence probe before trusting a checkpoint.
    """
    return set(
        _prepare_visit_frame(
            data,
            protein_id_col,
            visit,
            mediator,
            cat_covs,
            cont_covs,
            response_var,
            response_var_bl,
        )[protein_id_col]
        .unique()
        .to_list()
    )


def _prepare_visit_frame(
    data: pl.DataFrame,
    protein_id_col: str,
    visit: int,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var: str,
    response_var_bl: str,
) -> pl.DataFrame:
    """Filter to one visit + complete cases on the model's variables."""
    must_be_finite = (
        [response_var, response_var_bl, mediator]
        + list(cont_covs)
        + [c for c in cat_covs if c not in {"TRT01A", "USUBJID"}]
    )
    schema = data.collect_schema()

    finite_checks = []
    for c in must_be_finite:
        finite_checks.append(pl.col(c).is_not_null())
        if schema[c].is_float():
            finite_checks.append(pl.col(c).is_not_nan())
            finite_checks.append(pl.col(c).is_infinite().not_())

    keep_cols = {protein_id_col, response_var, response_var_bl, mediator, "TRT01A", "USUBJID"}
    keep_cols.update(cont_covs)
    keep_cols.update(cat_covs)
    available = set(data.columns)
    keep_cols = sorted(keep_cols & available)

    return (
        data.select(keep_cols + [VISIT_COL])
        .filter(pl.col(VISIT_COL) == visit)
        .drop(VISIT_COL)
        .filter(pl.all_horizontal(finite_checks))
    )


def mediation_output_fingerprint(
    data: pl.DataFrame,
    annotation: pl.DataFrame,
    *,
    visits: list[int],
    mediator: str,
    cat_covariates: list[str],
    cont_covariates: list[str],
    sims: int,
    response_var: str,
    response_var_bl: str,
    protein_id_col: str,
    control_value: str,
    treat_value: str,
) -> tuple[dict, dict[int, set]]:
    """The fingerprint record of a mediation run, and each visit's complete-case protein set.

    The record is what ``run_mediation_pipeline`` compares a stored fingerprint file with:
    the model configuration, every visit's input-content fingerprint and the annotation's.
    The written marker x visit keys are added by :func:`write_output_fingerprint`.

    Args:
        data: The prepared long frame, as passed to ``run_mediation_pipeline``.
        annotation: Protein annotation, as passed to ``run_mediation_pipeline``.
        visits: Post-baseline VISITNUMs of the run.
        mediator: Column name of the mediating variable.
        cat_covariates: Categorical covariates.
        cont_covariates: Continuous covariates.
        sims: Bootstrap resamples behind every confidence interval.
        response_var: Response column.
        response_var_bl: Baseline response column.
        protein_id_col: Protein identifier column.
        control_value: Reference treatment level.
        treat_value: Active treatment level.

    Returns:
        ``(record, expected_proteins)``: the record with keys ``fingerprint``, ``inputs``
        (by visit, as strings) and ``annotation``, and the complete-case protein set per
        visit.
    """
    model_args = (mediator, cat_covariates, cont_covariates, response_var, response_var_bl)
    expected_proteins = {
        visit: _visit_protein_set(data, protein_id_col, visit, *model_args) for visit in visits
    }
    annotation_cols = [protein_id_col, "Assay"]
    annot = annotation.select([c for c in annotation_cols if c in annotation.columns]).unique()
    record = {
        "fingerprint": _mediation_fingerprint(
            mediator=mediator,
            cat_covs=cat_covariates,
            cont_covs=cont_covariates,
            sims=sims,
            response_var=response_var,
            response_var_bl=response_var_bl,
            protein_id_col=protein_id_col,
            control_value=control_value,
            treat_value=treat_value,
        ),
        "inputs": {
            str(visit): _visit_input_fingerprint(data, protein_id_col, visit, *model_args)
            for visit in visits
        },
        "annotation": _frame_fingerprint(annot),
    }
    return record, expected_proteins


def write_output_fingerprint(path: Path, record: dict, results: pl.DataFrame, **extra) -> None:
    """Write ``<csv stem>.fingerprint.json``: the run's record plus its written keys.

    Args:
        path: The fingerprint file.
        record: The first element of :func:`mediation_output_fingerprint`.
        results: The output CSV's rows (``marker`` and ``visit`` columns).
        **extra: Further fields to store; a rerun's reuse check ignores them.
    """
    keys = _keys_fingerprint(_marker_visit_keys(results))
    path.write_text(json.dumps(record | {"keys": keys} | extra, sort_keys=True))


def _run_mediation_single_mp(args):
    """Picklable wrapper for multiprocessing — reads data from parquet."""
    (
        protein,
        parquet_path,
        protein_id_col,
        visit,
        mediator,
        cat_covs,
        cont_covs,
        response_var,
        response_var_bl,
        control_value,
        treat_value,
        sims,
    ) = args
    # The parquet holds one visit's rows only (``_prepare_visit_frame``).
    protein_data = pl.scan_parquet(parquet_path).filter(pl.col(protein_id_col) == protein).collect()
    return run_mediation_single_numpy(
        protein=protein,
        data=protein_data,
        mediator=mediator,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
        response_var=response_var,
        response_var_bl=response_var_bl,
        control_value=control_value,
        treat_value=treat_value,
        sims=sims,
        visit=visit,
    )


def run_mediation_parallel(
    data: pl.DataFrame | pl.LazyFrame,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    visit: int,
    n_jobs: int = -1,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    control_value: str = CONTROL_VALUE,
    treat_value: str = TREAT_VALUE,
    sims: int = 1000,
) -> pl.DataFrame:
    """Run mediation analysis for all proteins at a single visit.

    Args:
        data: Full analysis dataset (all proteins, all visits), with ``VISITNUM``, ``TRT01A``
            and ``USUBJID``.
        mediator: Column name of the mediating variable.
        cat_covs: Categorical covariate columns.
        cont_covs: Continuous covariate columns.
        visit: Post-baseline VISITNUM to analyze.
        n_jobs: Number of parallel workers (-1 = all CPUs - 1).
        response_var: Response column.
        response_var_bl: Baseline response column.
        protein_id_col: Protein identifier column.
        control_value: Reference treatment level.
        treat_value: Active treatment level, recorded on each output row.
        sims: Number of bootstrap resamples.

    Returns:
        One row per protein with ACME, ADE, total effect and proportion mediated, each
        with its point estimate, 95% CI and p-values, plus ``visit``; empty when no
        protein could be fit.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    prepared = _prepare_visit_frame(
        data,
        protein_id_col,
        visit,
        mediator,
        cat_covs,
        cont_covs,
        response_var,
        response_var_bl,
    )

    proteins = prepared[protein_id_col].unique().to_list()
    n_jobs = resolve_n_jobs(n_jobs, len(proteins))

    print(f"  Visit {visit}: {len(proteins)} proteins, {sims} sims, {n_jobs} cores")

    chunksize = default_chunksize(len(proteins), n_jobs)
    all_rows = []
    # Workers read the visit frame from a temporary parquet, removed with its directory
    # however the run ends. Sorted by protein id so each worker's per-protein scan
    # prunes to its own row group instead of decompressing the whole file.
    with tempfile.TemporaryDirectory(prefix="mediation_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "prepared.parquet"
        prepared = prepared.sort(protein_id_col, maintain_order=True)
        prepared.write_parquet(
            tmp_path, row_group_size=max(1, prepared.height // max(len(proteins), 1))
        )
        del prepared
        gc.collect()

        tasks = (
            (
                protein,
                str(tmp_path),
                protein_id_col,
                visit,
                mediator,
                cat_covs,
                cont_covs,
                response_var,
                response_var_bl,
                control_value,
                treat_value,
                sims,
            )
            for protein in proteins
        )
        if n_jobs == 1:
            for args in tqdm(tasks, total=len(proteins), desc=f"  V{visit}", unit="prot"):
                result = _run_mediation_single_mp(args)
                if result is not None:
                    all_rows.append(result)
        else:
            # No worker initializer: every replicate draws from a Generator seeded
            # per call in bootstrap.py, so results do not depend on worker state.
            with forkserver_pool(n_jobs, maxtasksperchild=200) as pool:
                with tqdm(total=len(proteins), desc=f"  V{visit}", unit="prot") as pbar:
                    for result in pool.imap_unordered(
                        _run_mediation_single_mp, tasks, chunksize=chunksize
                    ):
                        if result is not None:
                            all_rows.append(result)
                        pbar.update(1)

    gc.collect()

    if not all_rows:
        return pl.DataFrame()
    return pl.DataFrame(all_rows)


def run_mediation_pipeline(
    data: pl.DataFrame | pl.LazyFrame,
    output_dir: Path,
    annotation: pl.DataFrame,
    mediator: str,
    visits: list[int],
    cat_covariates: list[str] | None = None,
    cont_covariates: list[str] | None = None,
    covar_label: str = "none",
    study_name: str = "STUDY",
    n_jobs: int = -1,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    control_value: str = CONTROL_VALUE,
    treat_value: str = TREAT_VALUE,
    sims: int = 1000,
) -> pl.DataFrame:
    """Complete per-visit mediation analysis pipeline.

    Runs causal mediation for every protein at each specified visit, annotates
    the results and writes one CSV per (study, covariate set, mediator). Each
    row carries the ACME, ADE and total-effect estimates with bootstrap CIs,
    the empirical bootstrap p-values and the Sobel/Wald analytic p-values; FDR
    is left to the consumers, which pool the analytic p-values per study.

    Args:
        data: Must contain ``protein_id_col``, ``response_var``, ``response_var_bl``,
            ``mediator``, ``USUBJID``, ``VISITNUM``, ``TRT01A`` and the covariates.
        output_dir: Root output directory.
        annotation: Protein annotation (``protein_id_col``, ``Assay``).
        mediator: Column name of the mediating variable.
        visits: Post-baseline VISITNUMs to analyze.
        cat_covariates: Categorical covariates; None uses ``CAT_COVARIATES``.
        cont_covariates: Continuous covariates; None means none.
        covar_label: Label of the covariate configuration, used in the output path.
        study_name: Study identifier for file names.
        n_jobs: Parallel workers (-1 = all CPUs - 1).
        response_var: Response column.
        response_var_bl: Baseline response column.
        protein_id_col: Protein identifier column.
        control_value: Reference treatment level.
        treat_value: Active treatment level, recorded on each output row.
        sims: Number of bootstrap resamples behind every confidence interval.

    Returns:
        Annotated results for every protein and visit; the existing output when one
        written by an identical run is already on disk; empty when nothing could be fit.
        Beside the CSV, ``<csv stem>.fingerprint.json`` records the model configuration,
        each visit's input-content fingerprint (``_visit_input_fingerprint``), the
        annotation's and the written marker x visit keys'.

    Raises:
        SystemExit: When an existing output or visit checkpoint has no fingerprint file,
            or its configuration, input data, annotation or marker x visit keys do not
            match today's run; it is never reused.
    """
    cat_covariates = cat_covariates or CAT_COVARIATES
    cont_covariates = cont_covariates or []

    output_dir = Path(output_dir)
    mediation_dir = output_dir / f"covar_{covar_label}" / "mediationRes"
    mediation_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    print(f"\n{'─' * 60}")
    print(f"Mediation analysis: {mediator}, visits={visits}, sims={sims}")
    print(f"{'─' * 60}")

    output_path = mediation_dir / f"{study_name}_covar_{covar_label}_mediation_{mediator}.csv"

    output_fp, expected_proteins = mediation_output_fingerprint(
        data,
        annotation,
        visits=visits,
        mediator=mediator,
        cat_covariates=cat_covariates,
        cont_covariates=cont_covariates,
        sims=sims,
        response_var=response_var,
        response_var_bl=response_var_bl,
        protein_id_col=protein_id_col,
        control_value=control_value,
        treat_value=treat_value,
    )
    fingerprint = output_fp["fingerprint"]
    input_fps = output_fp["inputs"]
    output_fp_path = output_path.with_suffix(".fingerprint.json")

    # Reuse the final output only if its stored fingerprint (model config, every visit's
    # input rows, the annotation) matches today's run and the file holds exactly the
    # marker x visit keys that run wrote, all within today's complete-case sets (guards
    # against a stale input or config and a truncated or edited file). The written keys
    # are stored in the fingerprint.
    if output_path.exists():
        existing = pl.read_csv(output_path)
        stored = json.loads(output_fp_path.read_text()) if output_fp_path.exists() else None
        expected_keys = {(m, v) for v, proteins in expected_proteins.items() for m in proteins}
        existing_keys = _marker_visit_keys(existing)
        stored_keys = (stored or {}).get("keys")
        reusable = (
            stored is not None
            and {k: stored.get(k) for k in output_fp} == output_fp
            and stored_keys == _keys_fingerprint(existing_keys)
            and existing_keys <= expected_keys
            and existing.height == len(existing_keys)
        )
        if not reusable:
            stored = stored or {}
            changed_inputs = sorted(
                k for k in input_fps if stored.get("inputs", {}).get(k) != input_fps[k]
            )
            raise SystemExit(
                f"Existing mediation output {output_path} does not match the "
                f"requested run — refusing to reuse it.\n"
                f"  fingerprint file present: {output_fp_path.exists()}\n"
                f"  config fingerprint match: {stored.get('fingerprint') == fingerprint}\n"
                f"  input data changed at visits: {changed_inputs}\n"
                f"  annotation match: {stored.get('annotation') == output_fp['annotation']}\n"
                f"  marker x visit keys match the recorded run: "
                f"{stored_keys == _keys_fingerprint(existing_keys)} "
                f"({len(existing_keys)} keys in file, {existing.height} rows, "
                f"{len(existing_keys - expected_keys)} outside the complete-case set)\n"
                "Delete the file (or point --out elsewhere) to start a fresh run."
            )
        print(f"  Output exists at {output_path}")
        return existing

    visit_results = []
    for visit in visits:
        visit_checkpoint = mediation_dir / f"_checkpoint_v{visit}.parquet"
        fp_path = visit_checkpoint.with_suffix(".fingerprint.json")
        checkpoint_fp = {"fingerprint": fingerprint, "input": input_fps[str(visit)]}
        if visit_checkpoint.exists():
            # A checkpoint substitutes for a whole visit's fits, so require the model
            # config and the input rows to match today's run, the file to hold exactly the
            # marker keys the visit wrote, each once, and all within the complete-case protein
            # set. Otherwise the stage stops and reports what differs.
            stored_fp = json.loads(fp_path.read_text()) if fp_path.exists() else {}
            ckpt_df = pl.read_parquet(visit_checkpoint)
            ckpt_keys = _marker_visit_keys(ckpt_df)
            ckpt_proteins = {m for m, _ in ckpt_keys}
            keys_digest = _keys_fingerprint(ckpt_keys)
            reusable = (
                {k: stored_fp.get(k) for k in checkpoint_fp} == checkpoint_fp
                and stored_fp.get("keys") == keys_digest
                and ckpt_df.height == len(ckpt_keys)
                and ckpt_proteins <= expected_proteins[visit]
            )
            if not reusable:
                raise SystemExit(
                    f"Checkpoint {visit_checkpoint} does not match the current "
                    f"visit {visit} run — refusing to substitute it.\n"
                    f"  config fingerprint match: {stored_fp.get('fingerprint') == fingerprint}\n"
                    f"  input fingerprint match: "
                    f"{stored_fp.get('input') == checkpoint_fp['input']}\n"
                    f"  marker keys match the recorded visit: "
                    f"{stored_fp.get('keys') == keys_digest} "
                    f"({len(ckpt_keys)} keys, {ckpt_df.height} rows)\n"
                    f"  proteins: {len(ckpt_proteins - expected_proteins[visit])} outside "
                    f"the complete-case set\n"
                    "Delete the checkpoint to refit this visit."
                )
            print(f"  Visit {visit}: loading from checkpoint")
            visit_results.append(ckpt_df)
            continue

        result = run_mediation_parallel(
            data,
            mediator=mediator,
            cat_covs=cat_covariates,
            cont_covs=cont_covariates,
            visit=visit,
            n_jobs=n_jobs,
            response_var=response_var,
            response_var_bl=response_var_bl,
            protein_id_col=protein_id_col,
            control_value=control_value,
            treat_value=treat_value,
            sims=sims,
        )
        n_expected = len(expected_proteins[visit])
        if result.height == 0:
            raise ValueError(f"visit {visit}: none of the {n_expected} proteins could be fitted")
        n_unfitted = n_expected - len({m for m, _ in _marker_visit_keys(result)})
        if n_unfitted:
            print(f"visit {visit}: {n_unfitted} of {n_expected} proteins had too few subjects to fit and have no row")
        result.write_parquet(visit_checkpoint)
        keys = _keys_fingerprint(_marker_visit_keys(result))
        fp_path.write_text(json.dumps(checkpoint_fp | {"keys": keys}))
        visit_results.append(result)

    if not visit_results:
        print("No mediation results produced.")
        return pl.DataFrame()

    results = pl.concat(visit_results, how="diagonal")

    annotation_cols = [protein_id_col, "Assay"]
    annot = annotation.select([c for c in annotation_cols if c in annotation.columns]).unique()
    results = results.join(annot.rename({protein_id_col: "marker"}), on="marker", how="left")

    results = results.sort(["visit", "marker"])
    results.write_csv(output_path)
    # Written after the CSV: an interrupted write leaves no fingerprint, and is refused.
    write_output_fingerprint(output_fp_path, output_fp, results)
    print(f"\n[ok] wrote {output_path} ({results.height} rows)")

    for visit in visits:
        ckpt = mediation_dir / f"_checkpoint_v{visit}.parquet"
        ckpt.unlink(missing_ok=True)
        ckpt.with_suffix(".fingerprint.json").unlink(missing_ok=True)

    for visit in visits:
        print(f"  Visit {visit}: {results.filter(pl.col('visit') == visit).height} proteins")

    return results
