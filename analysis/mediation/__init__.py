"""Causal mediation of the treatment effect through percent weight change.

``run_mediation_single_numpy`` is the engine: per protein and visit, OLS mediator,
outcome and total-effect models with a nonparametric bootstrap for the ACME and ADE
intervals and Sobel p-values (``bootstrap.py``). ``run_mediation_pipeline`` runs it
genome-wide in a worker pool and writes one CSV per platform. It gives the same estimates
as R's ``mediation::mediate``. ``classify`` holds the six-way class rule every consumer applies,
and ``results`` reads a platform's CSV and applies it.
"""

from analysis.mediation.constants import (
    CAT_COVARIATES,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
)
from analysis.mediation.core import run_mediation_single_numpy
from analysis.mediation.pipeline import run_mediation_pipeline

__all__ = [
    "run_mediation_single_numpy",
    "run_mediation_pipeline",
    "CAT_COVARIATES",
    "DEFAULT_RESPONSE_VAR",
    "DEFAULT_RESPONSE_VAR_BL",
]
