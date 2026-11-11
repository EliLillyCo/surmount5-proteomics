"""Six-way mediation class rule shared by the figures and tables.

Every univariate mediation consumer classifies a protein from three per-study BH-FDR
families (total, indirect, direct) and the signs of the indirect and direct estimates:

- ``ns``: the total effect does not clear ``alpha``; the decomposition is not read.
- ``partial``: indirect and direct both significant, same sign.
- ``inconsistent``: indirect and direct both significant, opposite signs.
- ``fully_mediated``: indirect significant only.
- ``direct_only``: direct significant only.
- ``unresolved``: total significant but neither component clears ``alpha``.

``classify_sobel`` applies the rule to a univariate mediation frame on the analytic
Sobel p-values; ``mediation_class_expr`` is the rule itself, parameterized by column
names so a frame with other effect names can reuse it.
"""

from __future__ import annotations

import polars as pl

from analysis.statistics import FDR_THRESHOLD, bh_fdr

NS_CLASS = "ns"
# Display and tabulation order of the five classes a total-effect-significant protein
# can take.
MEDIATION_CLASSES: tuple[str, ...] = (
    "fully_mediated",
    "partial",
    "direct_only",
    "inconsistent",
    "unresolved",
)


def mediation_class_expr(
    *,
    total_fdr: str,
    indirect_fdr: str,
    direct_fdr: str,
    indirect_est: str,
    direct_est: str,
    alpha: float = FDR_THRESHOLD,
) -> pl.Expr:
    """The six-way class as a Polars expression over the named FDR and estimate columns.

    Args:
        total_fdr: Column holding the total-effect FDR.
        indirect_fdr: Column holding the indirect-effect (ACME) FDR.
        direct_fdr: Column holding the direct-effect (ADE) FDR.
        indirect_est: Column holding the signed indirect-effect estimate.
        direct_est: Column holding the signed direct-effect estimate.
        alpha: FDR threshold; an effect is significant when its FDR is below it.

    Returns:
        Expression aliased ``mclass`` taking ``NS_CLASS`` or a ``MEDIATION_CLASSES`` value.
    """
    ind_sig = pl.col(indirect_fdr) < alpha
    dir_sig = pl.col(direct_fdr) < alpha
    same_sign = (pl.col(indirect_est) * pl.col(direct_est)) > 0
    return (
        pl.when(pl.col(total_fdr) >= alpha)
        .then(pl.lit(NS_CLASS))
        .when(ind_sig & dir_sig & same_sign)
        .then(pl.lit("partial"))
        .when(ind_sig & dir_sig)
        .then(pl.lit("inconsistent"))
        .when(ind_sig)
        .then(pl.lit("fully_mediated"))
        .when(dir_sig)
        .then(pl.lit("direct_only"))
        .otherwise(pl.lit("unresolved"))
        .alias("mclass")
    )


def classify_sobel(df: pl.DataFrame) -> pl.DataFrame:
    """Append unconditional BH-FDR of the analytic Sobel p-values and the ``mclass`` call.

    ``acme_fdr``/``ade_fdr``/``total_fdr`` are each one BH family over every row of
    ``df`` (one study, one visit); ``mclass`` is gated on ``total_fdr`` afterward, so
    the pool is never conditioned on the total effect.

    Args:
        df: One study and visit's mediation frame with ``ACME_pvalue_analytic``,
            ``ADE_pvalue_analytic``, ``total_pvalue_analytic``, ``ACME_estimate`` and
            ``ADE_estimate``.

    Returns:
        ``df`` with ``acme_fdr``, ``ade_fdr``, ``total_fdr`` and ``mclass`` appended.
    """
    df = df.with_columns(
        pl.Series("acme_fdr", bh_fdr(df["ACME_pvalue_analytic"].to_numpy())),
        pl.Series("ade_fdr", bh_fdr(df["ADE_pvalue_analytic"].to_numpy())),
        pl.Series("total_fdr", bh_fdr(df["total_pvalue_analytic"].to_numpy())),
    )
    return df.with_columns(
        mediation_class_expr(
            total_fdr="total_fdr",
            indirect_fdr="acme_fdr",
            direct_fdr="ade_fdr",
            indirect_est="ACME_estimate",
            direct_est="ADE_estimate",
            alpha=FDR_THRESHOLD,
        )
    )
