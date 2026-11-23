"""Constants and column definitions for the MMRM pipeline."""

from __future__ import annotations

from analysis.design import CAT_COVARIATES as CAT_COVARIATES

# ============================================================================
# MODEL CONTROL
# ============================================================================

# Arguments of ``mmrm::mmrm_control`` shared by every MMRM fit:
# Kenward-Roger degrees of freedom with the linear Kenward-Roger coefficient covariance,
# which matches SAS PROC MIXED DDFM=KR for the unstructured covariance. Passed as
# ``control=`` at each ``mmrm()`` call.
MMRM_CONTROL_ARGS = {"method": "Kenward-Roger", "vcov": "Kenward-Roger-Linear"}


def mmrm_control_r_call() -> str:
    """Render ``MMRM_CONTROL_ARGS`` as an R ``mmrm_control(...)`` call for R source strings.

    Returns:
        The call text, e.g. ``mmrm_control(method = "Kenward-Roger", vcov = "...")``.
    """
    args = ", ".join(f'{k} = "{v}"' for k, v in MMRM_CONTROL_ARGS.items())
    return f"mmrm_control({args})"


# ============================================================================
# COVARIATES
# ============================================================================

CONT_COVARIATES: list[str] = ["AGE"]

# ============================================================================
# RESPONSE VARIABLES
# ============================================================================

DEFAULT_RESPONSE_VAR = "NPX"
DEFAULT_RESPONSE_VAR_BL = "NPXBL"


# ============================================================================
# OUTPUT COLUMN SETS
# ============================================================================

# Common columns across all outputs
BASE_METADATA_COLS = ["marker", "Assay"]
STATS_COLS = ["pVal", "fdr"]

# Common comparison columns (for across/within treatment), from the two per-arm LS-means:
# LSM = 2^LSmean, FC = LSM1 / LSM2. SE-FC is the independence approximation
# FC * sqrt((SE-LSM1 / LSM1)^2 + (SE-LSM2 / LSM2)^2) from the two LS-means' standard errors; it
# is not the standard error of the contrast (that is SE in CONTRAST_COLS). The final files write
# FC and SE-FC; the LS-means they come from are not written.
COMPARISON_COLS = ["LSM1", "SE-LSM1", "LSM2", "SE-LSM2", "FC", "SE-FC"]
# The emmeans contrast itself, on the model (log2) scale: estimate (log2 FC), its own standard
# error SE and its Kenward-Roger df. pVal is this contrast's p-value.
CONTRAST_COLS = ["estimate", "SE", "df"]

# PCBL-specific column components
PCBL_GROUPING_COLS = ["trt", "time"]
PCBL_ESTIMATE_COLS = ["LSM", "PCBL", "SE_PCBL", "CV_PCBL"]
PCBL_INTERVAL_COLS = [
    "PCBL_upper_1SE",
    "PCBL_lower_1SE",
    "PCBL_upper_CI",
    "PCBL_lower_CI",
]

# Complete PCBL output columns
PCBL_OUTPUT_COLS = (
    BASE_METADATA_COLS
    + PCBL_GROUPING_COLS
    + PCBL_ESTIMATE_COLS
    + PCBL_INTERVAL_COLS
    + STATS_COLS
)

# Between-treatment comparison columns
ACROSS_TRT_COLS = (
    BASE_METADATA_COLS
    + ["trt1", "trt2", "time"]
    + COMPARISON_COLS
    + STATS_COLS
    + CONTRAST_COLS
)

# Within-treatment comparison columns
WITHIN_TRT_COLS = (
    BASE_METADATA_COLS
    + ["trt", "time1", "time2"]
    + COMPARISON_COLS
    + STATS_COLS
    + CONTRAST_COLS
)

# Raw output column definitions
RAW_LSMEANS_COLS = [
    "marker",
    "time",
    "trt",
    "LSmean",
    "StdErr",
    "DF",
    "tValue",
    "Probt",
    "Lower",
    "Upper",
    "pVal",
]
# Baseline uses same columns but without time
RAW_BASELINE_COLS = [c for c in RAW_LSMEANS_COLS if c != "time"]
