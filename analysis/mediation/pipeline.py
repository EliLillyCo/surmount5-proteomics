"""Parallel mediation pipeline for the manuscript weight-mediation analysis."""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control
from tqdm import tqdm

from .constants import (
    ANNOTATION_COLUMNS,
    CAT_COVARIATES,
    DEFAULT_CONTROL_VALUE,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    DEFAULT_SIMS,
    DEFAULT_TREAT_COL,
    DEFAULT_TREAT_VALUE,
    DEFAULT_VISITS,
)
from .core import run_mediation_single


def _run_mediation_single_mp(args):
    """Picklable wrapper that reads one protein from parquet and runs mediation."""
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
        treat_col,
        control_value,
        treat_value,
        sims,
    ) = args
    protein_data = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col(protein_id_col) == protein)
        .collect()
    )
    if "VISITNUM" in protein_data.columns:
        protein_data = protein_data.filter(pl.col("VISITNUM") == visit)
    return run_mediation_single(
        protein=protein,
        data=protein_data,
        mediator=mediator,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
        response_var=response_var,
        response_var_bl=response_var_bl,
        treat_col=treat_col,
        control_value=control_value,
        treat_value=treat_value,
        sims=sims,
        visit=visit,
    )


def _complete_case_filter(
    data: pl.DataFrame,
    *,
    response_var: str,
    response_var_bl: str,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    treat_col: str,
) -> list[pl.Expr]:
    schema = data.schema
    must_be_finite = (
        [response_var, response_var_bl, mediator, treat_col, "USUBJID"]
        + list(cont_covs)
        + [cov for cov in cat_covs if cov not in {treat_col, "USUBJID"}]
    )
    checks: list[pl.Expr] = []
    for col in must_be_finite:
        if col not in schema:
            continue
        checks.append(pl.col(col).is_not_null())
        if schema[col].is_float():
            checks.extend(
                [
                    pl.col(col).is_not_nan(),
                    pl.col(col).is_infinite().not_(),
                ]
            )
    return checks


def run_mediation_parallel(
    data: pl.DataFrame | pl.LazyFrame,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    visit: int,
    baseline_visitnum: int = 2,
    n_jobs: int = -1,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    treat_col: str = DEFAULT_TREAT_COL,
    control_value: str = DEFAULT_CONTROL_VALUE,
    treat_value: str = DEFAULT_TREAT_VALUE,
    sims: int = DEFAULT_SIMS,
) -> pl.DataFrame:
    """Run mediation for all proteins at one visit."""
    del baseline_visitnum
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    keep_cols = {
        protein_id_col,
        response_var,
        response_var_bl,
        mediator,
        treat_col,
        "USUBJID",
        "VISITNUM",
        *cat_covs,
        *cont_covs,
    }
    prepared = (
        data.select(sorted(keep_cols & set(data.columns)))
        .filter(pl.col("VISITNUM") == visit)
        .drop("VISITNUM")
    )
    finite_checks = _complete_case_filter(
        prepared,
        response_var=response_var,
        response_var_bl=response_var_bl,
        mediator=mediator,
        cat_covs=cat_covs,
        cont_covs=cont_covs,
        treat_col=treat_col,
    )
    if finite_checks:
        prepared = prepared.filter(pl.all_horizontal(finite_checks))
    proteins = sorted(prepared.get_column(protein_id_col).unique().to_list())
    if not proteins:
        return pl.DataFrame()

    if n_jobs == -1:
        n_jobs = min(max(cpu_count() - 1, 1), len(proteins))
    else:
        n_jobs = min(max(n_jobs, 1), len(proteins))

    print(
        f"  Visit {visit}: {len(proteins)} proteins, {sims:,} bootstrap replicates, {n_jobs} workers"
    )

    with tempfile.TemporaryDirectory(prefix="mediation_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "prepared.parquet"
        prepared.write_parquet(tmp_path)
        del prepared
        gc.collect()

        def task_iter():
            for protein in proteins:
                yield (
                    protein,
                    str(tmp_path),
                    protein_id_col,
                    visit,
                    mediator,
                    cat_covs,
                    cont_covs,
                    response_var,
                    response_var_bl,
                    treat_col,
                    control_value,
                    treat_value,
                    sims,
                )

        results: list[dict[str, float | int | str]] = []
        chunksize = min(5, max(1, len(proteins) // max(n_jobs * 20, 1)))

        if n_jobs == 1:
            for args in tqdm(
                task_iter(), total=len(proteins), desc=f"  V{visit}", unit="prot"
            ):
                result = _run_mediation_single_mp(args)
                if result is not None:
                    results.append(result)
        else:
            ctx = mp.get_context("forkserver")
            with ctx.Pool(n_jobs, maxtasksperchild=50) as pool:
                with tqdm(total=len(proteins), desc=f"  V{visit}", unit="prot") as pbar:
                    for result in pool.imap_unordered(
                        _run_mediation_single_mp, task_iter(), chunksize=chunksize
                    ):
                        if result is not None:
                            results.append(result)
                        pbar.update(1)

    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results).sort("marker")


def _apply_acme_correction(results: pl.DataFrame) -> pl.DataFrame:
    corrected = []
    for visit in sorted(results.get_column("visit").unique().to_list()):
        visit_df = results.filter(pl.col("visit") == visit)
        pvals = visit_df.get_column("ACME_pvalue").to_numpy()
        acme_fdr = false_discovery_control(pvals, method="bh")
        acme_bon = np.minimum(pvals * visit_df.height, 1.0)
        corrected.append(
            visit_df.with_columns(
                pl.Series("ACME_fdr", acme_fdr),
                pl.Series("ACME_bon", acme_bon),
            )
        )
    return pl.concat(corrected, how="vertical")


def _annotation_frame(annotation: pl.DataFrame, protein_id_col: str) -> pl.DataFrame:
    available = [
        protein_id_col,
        *[col for col in ANNOTATION_COLUMNS if col in annotation.columns],
    ]
    return annotation.select(available).unique().rename({protein_id_col: "marker"})


def _checkpoint_path(
    mediation_dir: Path,
    *,
    visit: int,
    mediator: str,
    cat_covariates: list[str],
    cont_covariates: list[str],
    baseline_visitnum: int,
    response_var: str,
    response_var_bl: str,
    treat_col: str,
    control_value: str,
    treat_value: str,
    sims: int,
) -> Path:
    config = {
        "visit": visit,
        "mediator": mediator,
        "cat_covariates": cat_covariates,
        "cont_covariates": cont_covariates,
        "baseline_visitnum": baseline_visitnum,
        "response_var": response_var,
        "response_var_bl": response_var_bl,
        "treat_col": treat_col,
        "control_value": control_value,
        "treat_value": treat_value,
        "sims": sims,
    }
    digest = hashlib.sha1(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    safe_mediator = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in mediator
    )
    return mediation_dir / f"_checkpoint_{safe_mediator}_v{visit}_{digest}.parquet"


def run_mediation_pipeline(
    data: pl.DataFrame | pl.LazyFrame,
    output_dir: Path,
    annotation: pl.DataFrame,
    mediator: str,
    visits: list[int] | tuple[int, ...] = DEFAULT_VISITS,
    cat_covariates: list[str] | None = None,
    cont_covariates: list[str] | None = None,
    covar_label: str = "none",
    study_name: str = "STUDY",
    baseline_visitnum: int = 2,
    n_jobs: int = -1,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    treat_col: str = DEFAULT_TREAT_COL,
    control_value: str = DEFAULT_CONTROL_VALUE,
    treat_value: str = DEFAULT_TREAT_VALUE,
    sims: int = DEFAULT_SIMS,
) -> pl.DataFrame:
    """Run the manuscript mediation workflow across all requested visits."""
    cat_covariates = list(cat_covariates or CAT_COVARIATES)
    cont_covariates = list(cont_covariates or [])
    visits = list(visits)

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    output_dir = Path(output_dir)
    mediation_dir = output_dir / f"covar_{covar_label}" / "mediationRes"
    mediation_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        mediation_dir / f"{study_name}_covar_{covar_label}_mediation_{mediator}.csv"
    )

    print(f"\n{'─' * 60}")
    print(
        f"Mediation analysis: mediator={mediator}, visits={visits}, bootstrap={sims:,}"
    )
    print(f"{'─' * 60}")

    visit_frames: list[pl.DataFrame] = []
    checkpoint_paths: list[Path] = []
    for visit in visits:
        checkpoint = _checkpoint_path(
            mediation_dir,
            visit=visit,
            mediator=mediator,
            cat_covariates=cat_covariates,
            cont_covariates=cont_covariates,
            baseline_visitnum=baseline_visitnum,
            response_var=response_var,
            response_var_bl=response_var_bl,
            treat_col=treat_col,
            control_value=control_value,
            treat_value=treat_value,
            sims=sims,
        )
        checkpoint_paths.append(checkpoint)
        if checkpoint.exists():
            checkpoint_df = pl.read_parquet(checkpoint)
            if checkpoint_df.height > 0:
                print(f"  Visit {visit}: loading checkpoint")
                visit_frames.append(checkpoint_df)
                continue
            checkpoint.unlink()

        visit_df = run_mediation_parallel(
            data=data,
            mediator=mediator,
            cat_covs=cat_covariates,
            cont_covs=cont_covariates,
            visit=visit,
            baseline_visitnum=baseline_visitnum,
            n_jobs=n_jobs,
            response_var=response_var,
            response_var_bl=response_var_bl,
            protein_id_col=protein_id_col,
            treat_col=treat_col,
            control_value=control_value,
            treat_value=treat_value,
            sims=sims,
        )
        if visit_df.height == 0:
            print(f"  Visit {visit}: no mediation results produced")
            continue
        tmp_checkpoint = checkpoint.with_suffix(".parquet.tmp")
        visit_df.write_parquet(tmp_checkpoint)
        tmp_checkpoint.rename(checkpoint)
        visit_frames.append(visit_df)

    if not visit_frames:
        print("No mediation results produced.")
        return pl.DataFrame()

    results = pl.concat(visit_frames, how="vertical")
    results = results.join(
        _annotation_frame(annotation, protein_id_col), on="marker", how="left"
    )
    results = _apply_acme_correction(results).sort(["visit", "marker"])
    results.write_csv(output_path)

    for checkpoint in checkpoint_paths:
        checkpoint.unlink(missing_ok=True)

    print(f"\nResults ({results.height} rows) saved to {output_path}")
    for visit in visits:
        visit_df = results.filter(pl.col("visit") == visit)
        if visit_df.height == 0:
            continue
        n_sig = visit_df.filter(pl.col("ACME_fdr") < 0.05).height
        print(
            f"  Visit {visit}: {visit_df.height} proteins, {n_sig} ACME hits at FDR<0.05"
        )
    return results
