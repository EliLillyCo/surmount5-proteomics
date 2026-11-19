"""Core single-protein per-visit causal mediation analysis."""

from __future__ import annotations

import numpy as np
import polars as pl

from analysis.mediation.bootstrap import mediate_bootstrap
from analysis.mediation.constants import (
    CONTROL_VALUE,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    TREAT_VALUE,
)
from analysis.statistics import SEED


def _build_design_matrix(
    data: pl.DataFrame,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var_bl: str,
    control_value: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Covariate design matrix and treatment indicator from a Polars DataFrame.

    Returns ``(X_common, treat)``: the intercept, baseline, continuous and
    dummy-coded categorical columns shared by every model, and the 0/1 treatment.
    """
    exclude = {"TRT01A", "USUBJID"}
    other_cats = [c for c in cat_covs if c not in exclude]

    n = data.height
    columns = [np.ones(n)]

    # Baseline response (if not already in cont_covs)
    if response_var_bl not in cont_covs:
        columns.append(data[response_var_bl].to_numpy().astype(np.float64))

    for col in cont_covs:
        columns.append(data[col].to_numpy().astype(np.float64))

    # Categorical covariates (dummy encoding, drop first level alphabetically)
    for col in other_cats:
        levels = sorted(data[col].unique().to_list())
        for level in levels[1:]:
            columns.append((data[col] == level).to_numpy().astype(np.float64))

    X_common = np.column_stack(columns)

    # Treatment: binary 0/1 (control=0, treat=1)
    treat = (data["TRT01A"] != control_value).to_numpy().astype(np.float64)

    return X_common, treat


def run_mediation_single_numpy(
    protein: str,
    data: pl.DataFrame,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    control_value: str = CONTROL_VALUE,
    treat_value: str = TREAT_VALUE,
    sims: int = 1000,
    visit: int | None = None,
) -> dict | None:
    """Causal mediation for one protein at one visit with the numpy bootstrap.

    Fits the OLS mediator, outcome and total-effect models and bootstraps the ACME,
    ADE and total effect (``bootstrap.mediate_bootstrap``).

    Args:
        protein: Protein identifier for the output row.
        data: One row per subject at the visit, with USUBJID, the response and its
            baseline, the mediator, treatment and covariates.
        mediator: Mediator column.
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        response_var: Response column.
        response_var_bl: Baseline response column.
        control_value: Reference treatment level.
        treat_value: Active treatment level, recorded on the output row.
        sims: Bootstrap resamples.
        visit: Visit recorded on the output row.

    Returns:
        The output row, or None when the finite rows are too few to fit the outcome
        model.
    """
    # The bootstrap resamples row indices, so the row order must not depend on the caller.
    data = data.sort("USUBJID")
    y = (data[response_var] - data[response_var_bl]).to_numpy().astype(np.float64)
    m = data[mediator].to_numpy().astype(np.float64)

    # Treatment levels carry no spaces, the same normalization the MMRM engine applies.
    data_clean = data.with_columns(pl.col("TRT01A").str.replace_all(" ", ""))
    control_clean = control_value.replace(" ", "")

    X_common, treat = _build_design_matrix(
        data_clean, cat_covs, cont_covs, response_var_bl, control_clean
    )

    # Drop rows with non-finite values (NaN, Inf from e.g. log2(0))
    finite_mask = np.isfinite(y) & np.isfinite(m) & np.all(np.isfinite(X_common), axis=1)
    if not finite_mask.all():
        y, m, X_common, treat = (
            y[finite_mask],
            m[finite_mask],
            X_common[finite_mask],
            treat[finite_mask],
        )

    # Too few subjects to fit the outcome model: no result for this protein.
    n_params = X_common.shape[1] + 2  # +treat +mediator
    if len(y) <= n_params:
        return None

    result = mediate_bootstrap(y=y, m=m, X_common=X_common, treat=treat, sims=sims, seed=SEED)

    return {
        "marker": protein,
        "visit": visit,
        **result,
        "treat": treat_value,
        "control": control_value,
        "mediator": mediator,
        "seed": SEED,
        "sims": sims,
    }
