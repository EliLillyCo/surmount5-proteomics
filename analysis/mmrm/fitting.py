"""Model fitting for MMRM analysis.

Provides single-protein and parallel (multiprocessing) model fitting using
the R ``mmrm`` and ``emmeans`` packages via rpy2.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
from multiprocessing import cpu_count
from typing import Literal

import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import Formula
from scipy.stats import false_discovery_control
from tqdm import tqdm

from .constants import (
    CAT_COVARIATES,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    RAW_BASELINE_COLS,
    RAW_LSMEANS_COLS,
)
from .preparation import build_formula
from .rinterface import extract_emmeans_summary, get_r_bindings, r_gc, to_r_factors

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def apply_fdr_correction(df: pl.DataFrame) -> pl.DataFrame:
    """Apply BH FDR correction to p-values across all rows in *df*."""
    pval_col = "pVal" if "pVal" in df.columns else "pvalue"
    fdr_vals = false_discovery_control(df[pval_col].to_numpy(), method="bh")
    return df.with_columns(pl.Series("fdr", fdr_vals))


def save_raw_lsmeans(
    lsmeans_df: pl.DataFrame, output_path, is_baseline: bool = False
) -> None:
    """Save raw lsmeans output with proper column selection."""
    from pathlib import Path

    cols = RAW_BASELINE_COLS if is_baseline else RAW_LSMEANS_COLS
    lsmeans_df.select(cols).write_csv(Path(output_path))


# ============================================================================
# SINGLE-PROTEIN FITTING
# ============================================================================


def fit_single_protein(
    protein: str,
    data: pl.DataFrame | pl.LazyFrame,
    model_type: Literal["change", "baseline", "aval"],
    cat_covs: list[str],
    cont_covs: list[str],
    baseline_visitnum: int = 2,
    visit_labels: dict[int, int] | None = None,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    emmeans_weights: str = "proportional",
) -> dict[str, pl.DataFrame] | None:
    """Fit MMRM model for a single protein.

    Args:
        protein: Protein identifier.
        data: Pre-filtered data for this protein only.
        model_type: Type of model to fit ('change', 'baseline', or 'aval').
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        baseline_visitnum: Visit number to use as baseline.
        visit_labels: Optional mapping of visit numbers to labels.
        response_var: Name of response variable (e.g., 'NPX' or 'log2_RFU').
        response_var_bl: Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').
        emmeans_weights: Weighting method for emmeans ("proportional" or "equal").
            Default is "proportional" (weights by cell frequencies).
    """
    from rpy2.robjects import pandas2ri

    model = None
    emm = None

    try:
        protein_data = data
        r_bindings = get_r_bindings()
        r_lm = r_bindings["lm"]
        mmrm_r = r_bindings["mmrm"]
        emmeans_r = r_bindings["emmeans"]

        # Clean treatment names BEFORE passing to R
        protein_data = protein_data.with_columns(
            pl.col("TRT01A").str.replace_all(" ", "")
        )

        protein_data_pd = protein_data.rechunk().to_pandas()
        del protein_data

        r_data = to_r_factors(protein_data_pd, cat_covs)
        del protein_data_pd

        if model_type == "baseline":
            other_cats = [c for c in cat_covs if c not in {"VISITNUM", "USUBJID"}]
            formula = (
                f"{response_var_bl} ~ {' + '.join(cont_covs + other_cats + ['TRT01A'])}"
            )

            model = r_lm(Formula(formula), data=r_data)

            emm = emmeans_r.emmeans(
                model, specs=ro.StrVector(["TRT01A"]), weights=emmeans_weights
            )
            lsmeans, contrasts = extract_emmeans_summary(
                emm, protein, is_baseline=True, visit_labels=visit_labels
            )

            # Get Type III test for treatment effect
            joint_test = emmeans_r.joint_tests(model)
            with pandas2ri.converter.context():
                joint_test_pd = pandas2ri.rpy2py(joint_test)
                joint_test_df = pl.from_pandas(joint_test_pd)

            type3_test = (
                joint_test_df.filter(pl.col("model term") == "TRT01A")
                .select(["F.ratio", "df1", "df2", "p.value"])
                .rename({"F.ratio": "F_statistic", "p.value": "pVal"})
                .with_columns(pl.lit(protein).alias("marker"))
            )

            del joint_test, joint_test_pd, joint_test_df

        else:
            # AVAL or change model
            resp_var = response_var if model_type == "aval" else f"{response_var}_DIFF"
            formula = build_formula(
                resp_var, cat_covs, cont_covs, model_type, response_var_bl
            )
            model = mmrm_r.mmrm(formula=Formula(formula), data=r_data)
            emm = emmeans_r.emmeans(
                model,
                specs=ro.StrVector(["TRT01A", "VISITNUM"]),
                weights=emmeans_weights,
            )
            lsmeans, contrasts = extract_emmeans_summary(
                emm, protein, is_baseline=False, visit_labels=visit_labels
            )
            type3_test = None

        result = {"lsmeans": lsmeans, "contrasts": contrasts}
        if type3_test is not None:
            result["type3"] = type3_test
        return result

    except Exception as e:
        print(f"Error fitting {protein} ({model_type} model): {e}")
        return None
    finally:
        try:
            del model, emm
        except Exception:
            pass
        try:
            del protein_data, protein_data_pd, r_data
        except Exception:
            pass
        gc.collect()


def _fit_protein_with_data(args):
    """Wrapper function for multiprocessing — must be at module level for pickling."""
    (
        protein,
        protein_data,
        model_type,
        cat_covs,
        cont_covs,
        baseline_visitnum,
        visit_labels,
        response_var,
        response_var_bl,
        emmeans_weights,
    ) = args
    return fit_single_protein(
        protein,
        protein_data,
        model_type,
        cat_covs,
        cont_covs,
        baseline_visitnum,
        visit_labels,
        response_var,
        response_var_bl,
        emmeans_weights,
    )


# ============================================================================
# PARALLEL FITTING
# ============================================================================


def run_mmrm_parallel(
    data: pl.DataFrame | pl.LazyFrame,
    model_type: Literal["change", "baseline", "aval"],
    cat_covs: list[str],
    cont_covs: list[str],
    baseline_visitnum: int = 2,
    n_jobs: int = -1,
    visit_labels: dict[int, int] | None = None,
    chunksize: int | None = None,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
    emmeans_weights: str = "proportional",
) -> dict[str, pl.DataFrame]:
    """Run MMRM for all proteins in parallel.

    Args:
        data: Input data.
        model_type: Type of model ('change', 'baseline', or 'aval').
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        baseline_visitnum: Visit number to use as baseline.
        n_jobs: Number of parallel jobs (-1 = all CPUs - 1).
        visit_labels: Optional mapping of visit numbers to labels.
        chunksize: Number of proteins per chunk.
        response_var: Name of response variable (e.g., 'NPX' or 'log2_RFU').
        response_var_bl: Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').
        protein_id_col: Name of protein identifier column (e.g., 'OlinkID' or 'SeqId').
        emmeans_weights: Weighting method for emmeans ("proportional" or "equal").
            Default is "proportional".
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    proteins = sorted(data[protein_id_col].unique().to_list())

    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
    print(
        f"Fitting {len(proteins)} proteins using {n_jobs} cores ({model_type} model)..."
    )

    def protein_data_generator():
        for protein in proteins:
            yield (
                protein,
                data.filter(pl.col(protein_id_col) == protein),
                model_type,
                cat_covs,
                cont_covs,
                baseline_visitnum,
                visit_labels,
                response_var,
                response_var_bl,
                emmeans_weights,
            )

    all_lsmeans = []
    all_contrasts = []
    all_type3 = []
    chunk_counter = 0

    if chunksize is None:
        chunksize = min(5, max(1, len(proteins) // (n_jobs * 20)))

    # Use forkserver context to avoid Polars rayon thread pool deadlock
    # after fork (Polars operations hang in forked children because the
    # rayon thread pool state is corrupted). Increased maxtasksperchild
    # to amortize R package loading overhead in fresh workers.
    ctx = mp.get_context("forkserver")
    with ctx.Pool(n_jobs, maxtasksperchild=50) as pool:
        with tqdm(total=len(proteins), desc="Fitting models", unit="protein") as pbar:
            for result in pool.imap_unordered(
                _fit_protein_with_data,
                protein_data_generator(),
                chunksize=chunksize,
            ):
                if result is not None:
                    all_lsmeans.append(result["lsmeans"])
                    all_contrasts.append(result["contrasts"])
                    if "type3" in result:
                        all_type3.append(result["type3"])
                pbar.update(1)
                chunk_counter += 1

                if chunk_counter >= chunksize:
                    temp_lsmeans = pl.concat(all_lsmeans)
                    temp_contrasts = pl.concat(all_contrasts)
                    temp_type3 = pl.concat(all_type3) if all_type3 else None
                    del all_lsmeans, all_contrasts, all_type3
                    all_lsmeans = [temp_lsmeans]
                    all_contrasts = [temp_contrasts]
                    all_type3 = [temp_type3] if temp_type3 is not None else []
                    del temp_lsmeans, temp_contrasts
                    if temp_type3 is not None:
                        del temp_type3
                    chunk_counter = 0
                    gc.collect()
                    r_gc(full=ro.BoolVector([True]))

    lsmeans = pl.concat(all_lsmeans) if all_lsmeans else pl.DataFrame()
    contrasts = pl.concat(all_contrasts) if all_contrasts else pl.DataFrame()
    type3 = pl.concat(all_type3) if all_type3 else pl.DataFrame()

    del all_lsmeans, all_contrasts, all_type3, data
    gc.collect()
    r_gc()

    result = {"lsmeans": lsmeans, "contrasts": contrasts}
    if len(type3) > 0:
        result["type3"] = type3
    return result
