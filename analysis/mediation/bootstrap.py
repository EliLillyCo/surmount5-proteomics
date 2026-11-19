"""Vectorized numpy bootstrap for causal mediation with linear models.

Implements the product-of-coefficients method with nonparametric bootstrap,
equivalent to R's mediation::mediate(boot=TRUE) for lm models.

Every effect (ACME, ADE, total) is reported under TWO p-value columns:
  - `*_pvalue`          : empirical bootstrap p, R mediation::mediate
    convention (`p = min(1, 2 * min(#boot>0, #boot<0) / n_boot)`), computed
    from the same replicate arrays used for the CIs. When no replicate
    crosses zero, the p-value is the smallest positive float (p < 2/sims).
  - `*_pvalue_analytic` : Sobel/parametric (product-of-coefficients standard
    error for ACME; Wald z-tests of the linear-model coefficients, with a normal
    reference, for ADE and total).
    Floor-free, continuous — this is what genome-wide FDR pooling uses
    downstream.
"""

import math

import numpy as np
from numpy.linalg import lstsq
from scipy.stats import norm


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least squares. Returns β where y = Xβ."""
    return lstsq(X, y, rcond=None)[0]


def _ols_se(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Standard errors of OLS coefficients."""
    n, p = X.shape
    resid = y - X @ coef
    sigma2 = float(resid @ resid) / max(n - p, 1)
    xtx_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(xtx_inv) * sigma2)


def _sobel_standard_error(a: float, b: float, sa: float, sb: float) -> float:
    variance = (b * b * sa * sa) + (a * a * sb * sb)
    if variance <= 0:
        return 0.0
    return math.sqrt(variance)


def _sobel_p_value(indirect_effect: float, indirect_se: float) -> float:
    """Two-sided Sobel p-value with underflow-safe tail handling."""
    if indirect_se <= 0:
        return 1.0
    z = abs(indirect_effect / indirect_se)
    log_p = math.log(2.0) + float(norm.logsf(z))
    smallest = float(np.nextafter(0.0, 1.0))
    if log_p <= math.log(smallest):
        return smallest
    return float(math.exp(log_p))


def _empirical_p_value(estimate: float, boot: np.ndarray) -> float:
    """Two-sided empirical bootstrap p-value, R mediation::mediate convention.

    p = min(1, 2 * min(#boot > 0, #boot < 0) / n_boot); p = 1 if the point
    estimate itself is exactly 0. When no replicate crosses zero, it returns the
    smallest positive float (p < 2/n_boot).
    """
    if estimate == 0.0:
        return 1.0
    n_boot = boot.shape[0]
    n_pos = int(np.sum(boot > 0))
    n_neg = int(np.sum(boot < 0))
    p = min(1.0, 2.0 * min(n_pos, n_neg) / n_boot)
    floor = float(np.nextafter(0.0, 1.0))
    return p if p > floor else floor


def mediate_bootstrap(
    y: np.ndarray,
    m: np.ndarray,
    X_common: np.ndarray,
    treat: np.ndarray,
    sims: int = 1000,
    seed: int = 0,
    batch_size: int = 10_000,
) -> dict:
    """Nonparametric bootstrap mediation for linear models.

    Computes ACME (indirect effect), ADE (direct effect), and total effect
    using the product-of-coefficients method. Point estimates are the
    full-sample OLS fits, CIs are bootstrap percentiles, and each effect gets
    both an empirical bootstrap p-value and an analytic Sobel/parametric one
    (see module docstring).

    Args:
        y: Outcome (response minus baseline), shape (n,).
        m: Mediator, shape (n,).
        X_common: Design matrix of shared covariates including the intercept
            column, shape (n, p).
        treat: Binary treatment indicator (0 = control, 1 = treated), shape (n,).
        sims: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
        batch_size: Resample indices are drawn in chunks of this many replicates to
            bound memory.

    Returns:
        Dict keyed ``{ACME,ADE,total}_{estimate,ci_lower,ci_upper,pvalue,pvalue_analytic}``
        plus ``prop_mediated`` with its ``_ci_lower`` and ``_ci_upper``; the proportion
        mediated has a NaN ``prop_mediated_pvalue``.
    """
    n = len(y)

    # Mediator model: m ~ X_common + treat
    X_med = np.column_stack([X_common, treat])
    treat_idx_med = X_med.shape[1] - 1

    # Outcome model: y ~ X_common + treat + m
    X_out = np.column_stack([X_common, treat, m])
    treat_idx_out = X_out.shape[1] - 2
    med_idx_out = X_out.shape[1] - 1

    # Point estimates and standard errors (for Sobel p-values)
    beta_m = _ols_coef(X_med, m)
    beta_y = _ols_coef(X_out, y)
    se_m = _ols_se(X_med, m, beta_m)
    se_y = _ols_se(X_out, y, beta_y)

    a_est = float(beta_m[treat_idx_med])
    b_est = float(beta_y[med_idx_out])
    acme_est = a_est * b_est
    ade_est = float(beta_y[treat_idx_out])
    total_est = acme_est + ade_est

    acme_se = _sobel_standard_error(
        a=a_est, b=b_est, sa=float(se_m[treat_idx_med]), sb=float(se_y[med_idx_out])
    )
    acme_pvalue = _sobel_p_value(acme_est, acme_se)
    ade_pvalue = _sobel_p_value(ade_est, float(se_y[treat_idx_out]))

    # Total effect model: y ~ X_common + treat (no mediator)
    X_total = np.column_stack([X_common, treat])
    beta_total = _ols_coef(X_total, y)
    se_total = _ols_se(X_total, y, beta_total)
    total_pvalue = _sobel_p_value(float(beta_total[-1]), float(se_total[-1]))

    # Bootstrap replicates: percentile CIs and the empirical p-values
    rng = np.random.default_rng(seed)
    acme_boot = np.empty(sims)
    ade_boot = np.empty(sims)
    total_boot = np.empty(sims)

    done = 0
    while done < sims:
        chunk = min(batch_size, sims - done)
        indices = rng.integers(0, n, size=(chunk, n))

        for i in range(chunk):
            idx = indices[i]
            # X_med is X_out minus its last (mediator) column, so gathering
            # X_out once and slicing views for the mediator model avoids
            # regathering the same rows twice (X_med[idx] and m[idx] would
            # each re-duplicate work already done by X_out[idx]).
            X_out_i = X_out[idx]
            X_med_i = X_out_i[:, :-1]
            m_i = X_out_i[:, -1]
            bm = _ols_coef(X_med_i, m_i)
            by = _ols_coef(X_out_i, y[idx])
            acme_boot[done + i] = bm[treat_idx_med] * by[med_idx_out]
            ade_boot[done + i] = by[treat_idx_out]
            total_boot[done + i] = acme_boot[done + i] + ade_boot[done + i]

        done += chunk

    # Proportion mediated (bootstrap)
    with np.errstate(divide="ignore", invalid="ignore"):
        prop_boot = np.where(total_boot != 0, acme_boot / total_boot, 0.0)
    prop_est = acme_est / total_est if total_est != 0 else 0.0

    return {
        "ACME_estimate": acme_est,
        "ACME_ci_lower": float(np.percentile(acme_boot, 2.5)),
        "ACME_ci_upper": float(np.percentile(acme_boot, 97.5)),
        "ACME_pvalue": _empirical_p_value(acme_est, acme_boot),
        "ACME_pvalue_analytic": acme_pvalue,
        "ADE_estimate": ade_est,
        "ADE_ci_lower": float(np.percentile(ade_boot, 2.5)),
        "ADE_ci_upper": float(np.percentile(ade_boot, 97.5)),
        "ADE_pvalue": _empirical_p_value(ade_est, ade_boot),
        "ADE_pvalue_analytic": ade_pvalue,
        "total_estimate": total_est,
        "total_ci_lower": float(np.percentile(total_boot, 2.5)),
        "total_ci_upper": float(np.percentile(total_boot, 97.5)),
        "total_pvalue": _empirical_p_value(total_est, total_boot),
        "total_pvalue_analytic": total_pvalue,
        "prop_mediated": prop_est,
        "prop_mediated_ci_lower": float(np.percentile(prop_boot, 2.5)),
        "prop_mediated_ci_upper": float(np.percentile(prop_boot, 97.5)),
        "prop_mediated_pvalue": float("nan"),
    }
