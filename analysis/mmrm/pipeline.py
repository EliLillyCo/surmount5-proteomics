"""Main MMRM pipeline orchestrator.

Coordinates data preparation, model fitting (change, AVAL, baseline),
and output generation into a single ``run_mmrm_pipeline()`` call.
"""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from .constants import (
    CAT_COVARIATES,
    CONT_COVARIATES,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
)
from .fitting import (
    run_mmrm_parallel,
    save_raw_lsmeans,
)
from .preparation import prepare_data
from .reporting import (
    generate_baseline_output,
    generate_comparison_output,
    save_pcbl_files,
    transform_to_pcbl,
)
from .rinterface import r_gc


def run_mmrm_pipeline(
    data: pl.DataFrame | pl.LazyFrame,
    output_dir: Path,
    annotation: pl.DataFrame,
    cat_covariates: list[str] = None,
    cont_covariates: list[str] = None,
    covar_label: str = "none",
    study_name: str = "STUDY",
    baseline_visitnum: int = 2,
    n_jobs: int = -1,
    visit_labels: dict[int, int] | None = None,
    chunksize: int | None = None,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    placebo_comparisons_only: bool = True,
    emmeans_weights: str = "proportional",
) -> dict[str, pl.DataFrame]:
    """Complete MMRM analysis pipeline.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Must contain: protein_id_col, response_var, response_var_bl,
        USUBJID, VISITNUM, TRT01A, plus covariates.
    output_dir : Path
        Output directory.
    annotation : pl.DataFrame
        Protein annotation with columns: protein_id_col, Assay, allQC.
    cat_covariates : list[str], optional
        Categorical covariates (default: CAT_COVARIATES).
    cont_covariates : list[str], optional
        Continuous covariates (default: CONT_COVARIATES).
    covar_label : str
        Label for covariate combination.
    study_name : str
        Study identifier for output file naming.
    baseline_visitnum : int
        Visit number to use as baseline.
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs - 1).
    visit_labels : dict[int, int], optional
        Mapping of visit numbers to display labels.
    response_var : str
        Name of response variable (e.g., 'NPX' or 'log2_RFU').
    response_var_bl : str
        Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').
    protein_id_col : str
        Name of protein identifier column (e.g., 'OlinkID' or 'SeqId').
    placebo_comparisons_only : bool
        If True, filter across-treatment comparisons to only include
        placebo comparisons.  Falls back to all comparisons if no placebo
        exists.  Default: True.
    emmeans_weights : str
        Weighting method for emmeans ("proportional" or "equal").
        Default: "proportional" (weights by cell frequencies).

    Returns
    -------
    dict
        Dictionary with lsmeans and contrasts for all three models.
    """
    cat_covariates = cat_covariates or CAT_COVARIATES
    cont_covariates = cont_covariates or CONT_COVARIATES

    output_dir = Path(output_dir)
    covar_output_dir = output_dir / f"covar_{covar_label}"
    covar_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data
    data_clean = prepare_data(
        data,
        baseline_visitnum,
        response_var,
        response_var_bl,
        protein_id_col,
    )

    # Step 2: Fit change model
    print("\n=== Change from Baseline Model ===")
    results_chg = run_mmrm_parallel(
        data_clean,
        "change",
        cat_covariates,
        cont_covariates,
        baseline_visitnum,
        n_jobs,
        visit_labels,
        chunksize,
        response_var,
        response_var_bl,
        protein_id_col,
        emmeans_weights,
    )

    # Save raw outputs
    (covar_output_dir / "rawOutput").mkdir(parents=True, exist_ok=True)

    save_raw_lsmeans(
        results_chg["lsmeans"],
        covar_output_dir
        / "rawOutput"
        / f"{study_name}_covar_{covar_label}_chg_lsmeans.csv",
    )
    results_chg["contrasts"].write_csv(
        covar_output_dir
        / "rawOutput"
        / f"{study_name}_covar_{covar_label}_chg_diff.csv"
    )

    gc.collect()
    r_gc()

    # Step 3: Fit AVAL model
    print(f"\n=== AVAL Model ({response_var} over time) ===")
    results_aval = run_mmrm_parallel(
        data_clean,
        "aval",
        cat_covariates,
        cont_covariates,
        baseline_visitnum,
        n_jobs,
        visit_labels,
        chunksize,
        response_var,
        response_var_bl,
        protein_id_col,
        emmeans_weights,
    )

    save_raw_lsmeans(
        results_aval["lsmeans"],
        covar_output_dir
        / "rawOutput"
        / f"{study_name}_covar_{covar_label}_aval_lsmeans.csv",
    )
    results_aval["contrasts"].write_csv(
        covar_output_dir
        / "rawOutput"
        / f"{study_name}_covar_{covar_label}_aval_diff.csv"
    )

    del data_clean
    gc.collect()
    r_gc()

    # Step 4: Fit baseline model
    print(f"\n=== Baseline Model (VISITNUM={baseline_visitnum}) ===")
    data_baseline = (
        (data.lazy() if isinstance(data, pl.DataFrame) else data)
        .filter(pl.col(response_var_bl).is_not_null())
        .filter(pl.col("VISITNUM") == baseline_visitnum)
        .with_columns(
            [pl.col(c).cast(pl.Utf8) for c in cat_covariates if c in CAT_COVARIATES]
        )
        .sort([protein_id_col, "TRT01A"], descending=[False, True])
    )

    results_baseline = run_mmrm_parallel(
        data_baseline,
        "baseline",
        cat_covariates,
        cont_covariates,
        baseline_visitnum,
        n_jobs,
        visit_labels,
        chunksize,
        response_var,
        response_var_bl,
        protein_id_col,
        emmeans_weights,
    )

    if len(results_baseline["lsmeans"]) > 0:
        save_raw_lsmeans(
            results_baseline["lsmeans"],
            covar_output_dir
            / "rawOutput"
            / f"{study_name}_covar_{covar_label}_baseline_lsmeans.csv",
            is_baseline=True,
        )
    if len(results_baseline["contrasts"]) > 0:
        results_baseline["contrasts"].write_csv(
            covar_output_dir
            / "rawOutput"
            / f"{study_name}_covar_{covar_label}_baseline_diff.csv"
        )

    del data_baseline
    gc.collect()
    r_gc()

    # Step 5: Generate formatted outputs
    print("\n=== Generating Output Files ===")

    generate_comparison_output(
        results_chg["contrasts"],
        results_chg["lsmeans"],
        annotation,
        covar_output_dir,
        "across",
        covar_label,
        study_name,
        protein_id_col,
        placebo_comparisons_only=placebo_comparisons_only,
    )

    generate_comparison_output(
        results_chg["contrasts"],
        results_chg["lsmeans"],
        annotation,
        covar_output_dir,
        "within",
        covar_label,
        study_name,
        protein_id_col,
    )

    # PCBL results
    print("\n=== Generating PCBL Results ===")
    pcbl_data = transform_to_pcbl(results_chg["lsmeans"])
    save_pcbl_files(
        pcbl_data,
        covar_output_dir / "finalRes" / "pcblRes",
        annotation,
        study_name,
        covar_label,
        protein_id_col,
    )

    # Baseline comparisons
    if len(results_baseline["lsmeans"]) > 0 or len(results_baseline["contrasts"]) > 0:
        generate_baseline_output(
            results_baseline["lsmeans"],
            results_baseline["contrasts"],
            annotation,
            covar_output_dir,
            covar_label,
            study_name,
            type3=results_baseline.get("type3"),
            protein_id_col=protein_id_col,
        )

    print(f"\n✓ Analysis complete! Results in: {output_dir}")

    gc.collect()
    r_gc()

    return {
        "chg_lsmeans": results_chg["lsmeans"],
        "chg_contrasts": results_chg["contrasts"],
        "aval_lsmeans": results_aval["lsmeans"],
        "aval_contrasts": results_aval["contrasts"],
        "baseline_lsmeans": results_baseline["lsmeans"],
        "baseline_contrasts": results_baseline["contrasts"],
        "pcbl": pcbl_data,
    }
