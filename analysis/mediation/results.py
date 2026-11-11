"""Readers of the univariate mediation results shared by the figures and tables.

The mediation stage writes one CSV per platform under that platform's MMRM directory;
``load_platform`` reads one visit of it and applies the six-way class rule, so every
consumer classifies the same proteins the same way.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.mediation.classify import classify_sobel
from analysis.mediation.constants import COVAR_LABEL, MEDIATOR, STUDY_NAME
from paths import mmrm_dir


def mediation_csv_path(platform: str, mediator: str = MEDIATOR) -> Path:
    """The per-platform univariate mediation CSV under the platform's MMRM directory.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        mediator: Mediator named in the file.

    Returns:
        ``<mmrm_dir>/mediationRes/surmount5_covar_AGE_SEX_mediation_<mediator>.csv``.
    """
    return (
        mmrm_dir(platform)
        / "mediationRes"
        / f"{STUDY_NAME}_covar_{COVAR_LABEL}_mediation_{mediator}.csv"
    )


def load_platform(platform: str, visit: int, mediator: str = MEDIATOR) -> pl.DataFrame:
    """Load a platform's mediation CSV for one visit, apply Sobel FDR, classify.

    Effect scale is log2 NPX (Olink) or log2 RFU (SomaScan) change. Sign convention
    comes from the CSV's ``treat``/``control`` columns (treat = TZP, control = SEMA):
    positive = higher on tirzepatide than on semaglutide.

    FDR pool: ``acme_fdr``/``ade_fdr``/``total_fdr`` are each an unconditional BH
    across all proteins of this platform and visit, on the analytic Sobel/Wald
    p-values; the total-FDR gate is applied only afterward, in the ``mclass``
    assignment, never before pooling.

    ``mclass`` is one of: ns (total effect not significant), fully_mediated,
    direct_only, partial, inconsistent, or unresolved (total effect significant but
    neither ACME nor ADE individually clears FDR).

    Args:
        platform: ``"olink"`` or ``"soma"``.
        visit: VISITNUM to keep; the CSV holds every
            visit that was run.
        mediator: Mediator column of the CSV file name.

    Returns:
        One row per protein: the bootstrap result columns (``marker``, ``visit``,
        ``total_*``, ``ACME_*``, ``ADE_*``, ``prop_mediated*``, ``treat``, ``control``,
        ``mediator``, ``seed``, ``sims``, ``Assay``) plus ``acme_fdr``,
        ``ade_fdr``, ``total_fdr`` and ``mclass``.
    """
    df = pl.read_csv(mediation_csv_path(platform, mediator)).filter(pl.col("visit") == visit)
    return classify_sobel(df)
