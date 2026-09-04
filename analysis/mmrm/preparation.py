"""Data preparation and formula building for MMRM analysis."""

from __future__ import annotations

import polars as pl

from .constants import (
    CAT_COVARIATES,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
)


def prepare_data(
    data: pl.DataFrame | pl.LazyFrame,
    baseline_visitnum: int = 2,
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    protein_id_col: str = "OlinkID",
) -> pl.LazyFrame:
    """Prepare analysis dataset with change from baseline.

    Args:
        data: Input data.
        baseline_visitnum: Visit number to use as baseline.
        response_var: Name of response variable (e.g., 'NPX' or 'log2_RFU').
        response_var_bl: Name of baseline response variable (e.g., 'NPXBL' or 'log2_RFUBL').
        protein_id_col: Name of protein identifier column (e.g., 'OlinkID' or 'SeqId').
    """
    if isinstance(data, pl.DataFrame):
        data = data.lazy()

    diff_col = f"{response_var}_DIFF"

    prepared = (
        data.filter(
            pl.col(response_var).is_not_null() & pl.col(response_var_bl).is_not_null()
        )
        .filter(pl.col("VISITNUM") != baseline_visitnum)
        .with_columns(
            [
                (pl.col(response_var) - pl.col(response_var_bl)).alias(diff_col),
                *[pl.col(c).cast(pl.Utf8) for c in CAT_COVARIATES],
            ]
        )
    )

    return prepared.sort(
        [protein_id_col, "TRT01A", "VISITNUM"], descending=[False, True, True]
    )


def build_formula(
    response: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_type: str,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
) -> str:
    """Build MMRM formula string.

    Args:
        response: Response variable name.
        cat_covs: Categorical covariates.
        cont_covs: Continuous covariates.
        response_type: Type of response ('chg' or 'aval').
        response_var_bl: Baseline response variable name.
    """
    exclude = {"TRT01A", "VISITNUM", "USUBJID"}
    other_cats = [c for c in cat_covs if c not in exclude]
    baseline_term = [response_var_bl] if response_var_bl not in cont_covs else []

    # Time-varying covariates (PCTCHG_*) get :VISITNUM interaction
    tv_covs = [c for c in cont_covs if c.startswith("PCTCHG_")]
    ti_covs = [c for c in cont_covs if not c.startswith("PCTCHG_")]
    tv_terms = tv_covs + [f"{c}:VISITNUM" for c in tv_covs]

    fixed_terms = (
        baseline_term
        + ti_covs
        + tv_terms
        + ["TRT01A", "VISITNUM", "TRT01A:VISITNUM"]
        + other_cats
    )

    fixed_effects = " + ".join(fixed_terms)
    return f"{response} ~ {fixed_effects} + us(VISITNUM | USUBJID)"
