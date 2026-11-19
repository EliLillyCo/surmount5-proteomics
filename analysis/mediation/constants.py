"""Constants for the mediation package.

The arm labels, visit numbers, output stem and covariate label are the design constants of
``analysis.design``; this module re-exports them so the engine and the runner share one
source.
"""

from analysis.design import CAT_COVARIATES as CAT_COVARIATES
from analysis.design import COVAR_LABEL as COVAR_LABEL
from analysis.design import CONTROL_VALUE as CONTROL_VALUE
from analysis.design import POST_BASELINE_VISITS as POST_BASELINE_VISITS
from analysis.design import STUDY_STEM
from analysis.design import TREAT_VALUE as TREAT_VALUE

# Visit column used by the mediation engine.
VISIT_COL = "VISITNUM"
# Continuous covariates; the engine always adds the baseline response itself.
CONT_COVARIATES: tuple[str, ...] = ("AGE", "WGTBL")
# Mediator: percent weight change from baseline.
MEDIATOR = "PCTCHG_WGT"
# Bootstrap draws behind every confidence interval; the Table S17/S18 footnote prints it.
DEFAULT_SIMS = 100_000
DEFAULT_RESPONSE_VAR = "NPX"
DEFAULT_RESPONSE_VAR_BL = "NPXBL"
# Output stem of the mediation files (``<STUDY_NAME>_covar_<COVAR_LABEL>_mediation_<mediator>.csv``).
STUDY_NAME = STUDY_STEM
