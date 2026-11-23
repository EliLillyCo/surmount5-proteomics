"""Model fitting for MMRM analysis.

Provides single-protein and parallel (multiprocessing) model fitting using
the R ``mmrm`` and ``emmeans`` packages via rpy2.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal

import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import Formula
from tqdm import tqdm

from analysis.parallel import default_chunksize, forkserver_pool, resolve_n_jobs
from analysis.rinterface import get_r_bindings, r_gc, to_r_factors

from .constants import (
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    MMRM_CONTROL_ARGS,
    RAW_BASELINE_COLS,
    RAW_LSMEANS_COLS,
)
from .preparation import build_formula
from .rinterface import extract_emmeans_summary

# emmeans cell weights: proportional to the cell frequencies.
EMMEANS_WEIGHTS = "proportional"

# Identifier columns that give a fit-result frame one canonical row order. Workers
# finish in load-balance order, so the pooled frames are sorted by these.
_ORDER_KEYS = ("marker", "trt", "time", "_trt", "_time", "term")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _canonical_order(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a fit-result frame by whichever identifier columns it carries."""
    keys = [c for c in _ORDER_KEYS if c in df.columns]
    return df.sort(keys) if keys else df


def save_raw_lsmeans(
    lsmeans_df: pl.DataFrame, output_path, is_baseline: bool = False
) -> None:
    """Save raw lsmeans output with proper column selection."""
    cols = RAW_BASELINE_COLS if is_baseline else RAW_LSMEANS_COLS
    lsmeans_df.select(cols).write_csv(Path(output_path))


# ============================================================================
# SINGLE-PROTEIN FITTING
# ============================================================================


def fit_single_protein(
    protein: str,
    data: pl.DataFrame | pl.LazyFrame,
    model_type: Literal["change", "baseline"],
    cat_covs: list[str],
    cont_covs: list[str],
    control_trt: str,
    reference_visit: int | str,
    visit_labels: dict[int, int] | None = None,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
) -> dict[str, pl.DataFrame]:
    """Fit MMRM model for a single protein.

    A failed fit raises an error naming the protein.

    Args:
        protein: Protein identifier.
        data: Pre-filtered data for this protein only.
        model_type: Type of model to fit ('change' or 'baseline').
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        control_trt: TRT01A level that is the factor reference (the comparator arm, with
            spaces removed as the treatment names are cleaned before R). Treatment
            contrasts are then "other arm - control".
        reference_visit: VISITNUM level that is the time-factor reference (the earliest
            post-baseline visit). Within-arm contrasts are then "later - earlier".
        visit_labels: Optional mapping of visit numbers to labels.
        response_var: Name of response variable (e.g., 'NPX' or 'log2_RFU').
        response_var_bl: Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').

    Returns:
        ``{"lsmeans", "contrasts"}`` for the protein.

    Raises:
        RuntimeError: When the model fit or the emmeans extraction fails.
    """
    model = None
    emm = None
    r_data = None

    try:
        r_bindings = get_r_bindings()
        r_lm = r_bindings["lm"]
        mmrm_r = r_bindings["mmrm"]
        emmeans_r = r_bindings["emmeans"]

        # Clean treatment names BEFORE passing to R
        protein_data = data.with_columns(pl.col("TRT01A").str.replace_all(" ", ""))

        protein_data_pd = protein_data.rechunk().to_pandas()
        del protein_data

        # Sorted levels with the reference first: factor coding and contrast direction
        # are independent of row order.
        factors: dict[str, str | None] = {"TRT01A": control_trt}
        if model_type != "baseline":
            factors["VISITNUM"] = str(reference_visit)
            factors["USUBJID"] = None
        for col in cat_covs:
            factors.setdefault(col, None)
        r_data = to_r_factors(protein_data_pd, factors)
        del protein_data_pd

        if model_type == "baseline":
            other_cats = [c for c in cat_covs if c not in {"VISITNUM", "USUBJID"}]
            formula = (
                f"{response_var_bl} ~ {' + '.join(cont_covs + other_cats + ['TRT01A'])}"
            )

            model = r_lm(Formula(formula), data=r_data)

            emm = emmeans_r.emmeans(
                model, specs=ro.StrVector(["TRT01A"]), weights=EMMEANS_WEIGHTS
            )
            lsmeans, contrasts = extract_emmeans_summary(
                emm, protein, is_baseline=True, visit_labels=visit_labels
            )

        else:
            # Change model
            formula = build_formula(
                f"{response_var}_DIFF", cat_covs, cont_covs, response_var_bl
            )
            model = mmrm_r.mmrm(
                formula=Formula(formula),
                data=r_data,
                control=mmrm_r.mmrm_control(**MMRM_CONTROL_ARGS),
            )
            emm = emmeans_r.emmeans(
                model,
                specs=ro.StrVector(["TRT01A", "VISITNUM"]),
                weights=EMMEANS_WEIGHTS,
            )
            lsmeans, contrasts = extract_emmeans_summary(
                emm, protein, is_baseline=False, visit_labels=visit_labels
            )

        return {"lsmeans": lsmeans, "contrasts": contrasts}

    except Exception as exc:
        raise RuntimeError(f"MMRM fit failed for {protein} ({model_type} model): {exc}") from exc
    finally:
        del model, emm, r_data
        gc.collect()
        # The R heap for this protein's objects lives in THIS worker process.
        r_gc()


def _fit_protein_with_data(args):
    """Wrapper function for multiprocessing — must be at module level for pickling."""
    (
        protein,
        protein_data,
        model_type,
        cat_covs,
        cont_covs,
        control_trt,
        reference_visit,
        visit_labels,
        response_var,
        response_var_bl,
    ) = args
    return fit_single_protein(
        protein,
        protein_data,
        model_type,
        cat_covs,
        cont_covs,
        control_trt,
        reference_visit,
        visit_labels,
        response_var,
        response_var_bl,
    )


# ============================================================================
# PARALLEL FITTING
# ============================================================================


def run_mmrm_parallel(
    data: pl.DataFrame | pl.LazyFrame,
    model_type: Literal["change", "baseline"],
    cat_covs: list[str],
    cont_covs: list[str],
    control_trt: str,
    n_jobs: int = -1,
    visit_labels: dict[int, int] | None = None,
    chunksize: int | None = None,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
) -> dict[str, pl.DataFrame]:
    """Run MMRM for all proteins in parallel.

    The time-factor reference is the earliest VISITNUM present in ``data``, so the
    change model (post-baseline rows only) contrasts later minus earlier.

    Args:
        data: Input data.
        model_type: Type of model ('change' or 'baseline').
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        control_trt: TRT01A reference level (comparator arm, spaces removed).
        n_jobs: Number of parallel jobs (-1 = all CPUs - 1).
        visit_labels: Optional mapping of visit numbers to labels.
        chunksize: Number of proteins per chunk.
        response_var: Name of response variable (e.g., 'NPX' or 'log2_RFU').
        response_var_bl: Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').
        protein_id_col: Name of protein identifier column (e.g., 'OlinkID' or 'SeqId').

    Returns:
        ``{"lsmeans", "contrasts"}`` over all proteins in canonical row order.

    Raises:
        RuntimeError: When any protein's fit fails; the message names the protein.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    proteins = data[protein_id_col].unique().sort().to_list()
    reference_visit = data["VISITNUM"].min()

    n_jobs = resolve_n_jobs(n_jobs)
    print(
        f"Fitting {len(proteins)} proteins using {n_jobs} cores ({model_type} model)..."
    )

    # One pass to bucket rows by protein rather than a full-frame scan per protein.
    partitions = data.partition_by(protein_id_col, as_dict=True, maintain_order=True)

    def protein_data_generator():
        for protein in proteins:
            yield (
                protein,
                partitions[(protein,)],
                model_type,
                cat_covs,
                cont_covs,
                control_trt,
                reference_visit,
                visit_labels,
                response_var,
                response_var_bl,
            )

    all_lsmeans = []
    all_contrasts = []

    if chunksize is None:
        chunksize = default_chunksize(len(proteins), n_jobs)

    # maxtasksperchild amortizes the R package load of each fresh worker.
    with forkserver_pool(n_jobs, maxtasksperchild=50) as pool:
        with tqdm(total=len(proteins), desc="Fitting models", unit="protein") as pbar:
            for result in pool.imap_unordered(
                _fit_protein_with_data,
                protein_data_generator(),
                chunksize=chunksize,
            ):
                all_lsmeans.append(result["lsmeans"])
                all_contrasts.append(result["contrasts"])
                pbar.update(1)

    # Workers finish in load-balance order; sort so two runs write identical rows.
    lsmeans = _canonical_order(pl.concat(all_lsmeans))
    contrasts = _canonical_order(pl.concat(all_contrasts))

    del all_lsmeans, all_contrasts, data
    gc.collect()

    return {"lsmeans": lsmeans, "contrasts": contrasts}
