"""Core single-protein mediation analysis with R bootstrap CIs and Sobel p-values."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import Formula
from scipy.stats import norm

from .constants import (
    DEFAULT_CONTROL_VALUE,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    DEFAULT_SIMS,
    DEFAULT_TREAT_COL,
    DEFAULT_TREAT_VALUE,
    SEED,
)
from .rinterface import coefficient_stats, r_gc, r_lm, r_mediate, to_r_factors


def _covariate_terms(
    cat_covs: list[str],
    cont_covs: list[str],
    response_var_bl: str,
    treat_col: str,
) -> list[str]:
    exclude = {treat_col, "USUBJID"}
    other_cats = [cov for cov in cat_covs if cov not in exclude]
    baseline = [response_var_bl] if response_var_bl not in cont_covs else []
    return baseline + list(cont_covs) + other_cats


def _build_mediator_formula(
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var_bl: str,
    treat_col: str,
) -> str:
    terms = _covariate_terms(cat_covs, cont_covs, response_var_bl, treat_col) + [
        treat_col
    ]
    return f"{mediator} ~ {' + '.join(terms)}"


def _build_outcome_formula(
    response: str,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var_bl: str,
    treat_col: str,
) -> str:
    terms = (
        [mediator]
        + _covariate_terms(cat_covs, cont_covs, response_var_bl, treat_col)
        + [treat_col]
    )
    return f"{response} ~ {' + '.join(terms)}"


def _build_total_formula(
    response: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var_bl: str,
    treat_col: str,
) -> str:
    terms = _covariate_terms(cat_covs, cont_covs, response_var_bl, treat_col) + [
        treat_col
    ]
    return f"{response} ~ {' + '.join(terms)}"


def sobel_p_value(indirect_effect: float, indirect_se: float) -> float:
    """Two-sided Sobel p-value with underflow-safe tail handling."""
    if indirect_se <= 0:
        raise ValueError("indirect_se must be positive")
    z = abs(indirect_effect / indirect_se)
    log_p = math.log(2.0) + float(norm.logsf(z))
    smallest = float(np.nextafter(0.0, 1.0))
    if log_p <= math.log(smallest):
        return smallest
    return float(math.exp(log_p))


def _sobel_standard_error(a: float, b: float, sa: float, sb: float) -> float:
    variance = (b * b * sa * sa) + (a * a * sb * sb)
    if variance <= 0:
        raise ValueError("Indirect-effect variance must be positive")
    return float(math.sqrt(variance))


def _extract_result(
    *,
    med_result,
    med_fit,
    out_fit,
    total_fit,
    protein: str,
    visit: int | None,
    mediator: str,
    treat_term: str,
    treat_value: str,
    control_value: str,
    sims: int,
) -> dict[str, float | int | str]:
    acme_estimate = float(med_result.rx2("d0")[0])
    a_stats = coefficient_stats(med_fit, treat_term)
    b_stats = coefficient_stats(out_fit, mediator)
    ade_stats = coefficient_stats(out_fit, treat_term)
    total_stats = coefficient_stats(total_fit, treat_term)
    acme_se = _sobel_standard_error(
        a=a_stats["estimate"],
        b=b_stats["estimate"],
        sa=a_stats["std_error"],
        sb=b_stats["std_error"],
    )

    return {
        "marker": protein,
        "visit": visit,
        "ACME_estimate": acme_estimate,
        "ACME_ci_lower": float(med_result.rx2("d0.ci")[0]),
        "ACME_ci_upper": float(med_result.rx2("d0.ci")[1]),
        "ACME_pvalue": sobel_p_value(acme_estimate, acme_se),
        "ADE_estimate": float(med_result.rx2("z0")[0]),
        "ADE_ci_lower": float(med_result.rx2("z0.ci")[0]),
        "ADE_ci_upper": float(med_result.rx2("z0.ci")[1]),
        "ADE_pvalue": ade_stats["p_value"],
        "total_estimate": float(med_result.rx2("tau.coef")[0]),
        "total_ci_lower": float(med_result.rx2("tau.ci")[0]),
        "total_ci_upper": float(med_result.rx2("tau.ci")[1]),
        "total_pvalue": total_stats["p_value"],
        "prop_mediated": float(med_result.rx2("n0")[0]),
        "prop_mediated_ci_lower": float(med_result.rx2("n0.ci")[0]),
        "prop_mediated_ci_upper": float(med_result.rx2("n0.ci")[1]),
        "prop_mediated_pvalue": float("nan"),
        "treat": treat_value,
        "control": control_value,
        "mediator": mediator,
        "seed": SEED,
        "sims": sims,
    }


def run_mediation_single(
    protein: str,
    data: pl.DataFrame,
    mediator: str,
    cat_covs: list[str],
    cont_covs: list[str],
    response_var: str = DEFAULT_RESPONSE_VAR,
    response_var_bl: str = DEFAULT_RESPONSE_VAR_BL,
    treat_col: str = DEFAULT_TREAT_COL,
    control_value: str = DEFAULT_CONTROL_VALUE,
    treat_value: str = DEFAULT_TREAT_VALUE,
    sims: int = DEFAULT_SIMS,
    visit: int | None = None,
) -> dict[str, float | int | str] | None:
    """Run one mediation model for one protein at one visit."""
    diff_col = f"{response_var}_DIFF"
    control_clean = control_value.replace(" ", "")
    treat_clean = treat_value.replace(" ", "")

    protein_data = data.with_columns(
        pl.col(treat_col).str.replace_all(" ", ""),
        (pl.col(response_var) - pl.col(response_var_bl)).alias(diff_col),
    )
    observed_treatments = set(
        protein_data.get_column(treat_col).drop_nulls().unique().to_list()
    )
    if {control_clean, treat_clean} - observed_treatments:
        print(
            f"Skipping {protein} at visit={visit}: missing one of "
            f"{control_clean!r} or {treat_clean!r} in {treat_col}"
        )
        return None

    min_rows = (
        len(_covariate_terms(cat_covs, cont_covs, response_var_bl, treat_col)) + 4
    )
    if protein_data.height < min_rows:
        print(
            f"Skipping {protein} at visit={visit}: "
            f"{protein_data.height} rows available, need at least {min_rows}"
        )
        return None

    r_data = med_fit = out_fit = total_fit = med_result = None
    try:
        protein_pd = protein_data.to_pandas()
        r_data = to_r_factors(
            protein_pd,
            cat_covs,
            treat_col=treat_col,
            treat_ref=control_clean,
        )

        med_fit = r_lm(
            Formula(
                _build_mediator_formula(
                    mediator=mediator,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    response_var_bl=response_var_bl,
                    treat_col=treat_col,
                )
            ),
            data=r_data,
        )
        out_fit = r_lm(
            Formula(
                _build_outcome_formula(
                    response=diff_col,
                    mediator=mediator,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    response_var_bl=response_var_bl,
                    treat_col=treat_col,
                )
            ),
            data=r_data,
        )
        total_fit = r_lm(
            Formula(
                _build_total_formula(
                    response=diff_col,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    response_var_bl=response_var_bl,
                    treat_col=treat_col,
                )
            ),
            data=r_data,
        )

        ro.r["set.seed"](SEED)
        med_result = r_mediate(
            med_fit,
            out_fit,
            treat=treat_col,
            mediator=mediator,
            control_value=control_clean,
            treat_value=treat_clean,
            sims=sims,
            boot=True,
        )
        treat_term = f"{treat_col}{treat_clean}"
        return _extract_result(
            med_result=med_result,
            med_fit=med_fit,
            out_fit=out_fit,
            total_fit=total_fit,
            protein=protein,
            visit=visit,
            mediator=mediator,
            treat_term=treat_term,
            treat_value=treat_value,
            control_value=control_value,
            sims=sims,
        )
    finally:
        del med_result, total_fit, out_fit, med_fit, r_data
        r_gc()
