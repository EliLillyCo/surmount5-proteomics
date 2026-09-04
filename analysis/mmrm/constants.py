"""Constants and column definitions for the MMRM pipeline."""

from __future__ import annotations

# ============================================================================
# COVARIATES
# ============================================================================

CAT_COVARIATES: list[str] = ["SEX"]
CONT_COVARIATES: list[str] = ["AGE"]

# ============================================================================
# RESPONSE VARIABLES
# ============================================================================

DEFAULT_RESPONSE_VAR = "NPX"
DEFAULT_RESPONSE_VAR_BL = "NPXBL"


def get_response_vars(response_var: str) -> dict:
    """Get response variable names for different model types."""
    return {"chg": f"{response_var}_DIFF", "aval": response_var}


# ============================================================================
# OUTPUT COLUMN SETS
# ============================================================================

# Common columns across all outputs
BASE_METADATA_COLS = ["marker", "Assay", "allQC"]
STATS_COLS = ["pVal", "fdr"]

# Common comparison columns (for across/within treatment)
COMPARISON_COLS = ["LSM1", "SE-LSM1", "LSM2", "SE-LSM2", "FC", "SE-FC"]

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

# Across-treatment comparison columns
ACROSS_TRT_COLS = (
    BASE_METADATA_COLS + ["trt1", "trt2", "time"] + COMPARISON_COLS + STATS_COLS
)

# Within-treatment comparison columns
WITHIN_TRT_COLS = (
    BASE_METADATA_COLS + ["trt", "time1", "time2"] + COMPARISON_COLS + STATS_COLS
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
