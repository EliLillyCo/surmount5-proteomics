"""SURMOUNT-5 design constants: treatment arms, platforms, proteomics visits, output stem, covariate label.

The one home of these values for the pipelines, figures, tables and tests. It imports nothing
(no mount configuration, no third-party package), so a figure, a table or a test can read an
arm label without loading ``paths.yaml``. The false-discovery threshold and the seed live in
``analysis.statistics``, which is config-free for the same reason.
"""

from __future__ import annotations

# Output stem of every QC run (``surmount5_olink``), MMRM tree and mediation file.
STUDY_STEM = "surmount5"

# Treatment arms. ``RAW_*`` are the ADaM ``TRT01A`` values; the MMRM engine strips their
# spaces, so the model outputs, the QC breakdown and the result file names carry the
# space-free form. The short keys (``TZP``, ``SEMA``) index ``ARMS`` and the display labels.
RAW_TREAT_VALUE = "TZP 15mg or MTD"
RAW_CONTROL_VALUE = "SEMA 2.4mg or MTD"
TREAT_VALUE = "TZP15mgorMTD"
CONTROL_VALUE = "SEMA2.4mgorMTD"
# Short arm key -> analysis (space-free) TRT01A value, treatment first.
ARMS: dict[str, str] = {"TZP": TREAT_VALUE, "SEMA": CONTROL_VALUE}
# Short arm key -> label printed in figure legends and table columns.
ARM_LABELS: dict[str, str] = {"TZP": "TZP", "SEMA": "SEMA"}
# Analysis TRT01A value -> full arm name (the study-design figure's trial card).
ARM_FULL_NAMES: dict[str, str] = {
    TREAT_VALUE: "Tirzepatide 15 mg or MTD",
    CONTROL_VALUE: "Semaglutide 2.4 mg or MTD",
}

# The two assay platforms, Olink first; per-platform loops run in this order.
PLATFORMS: tuple[str, ...] = ("olink", "soma")

# Proteomics visits (ADaM VISITNUM) and their weeks; baseline is week 0.
BASELINE_VISITNUM = 2
BASELINE_WEEK = 0
# Post-baseline VISITNUM -> week.
VISIT_LABELS: dict[int, int] = {8: 24, 20: 72}
VISITNUMS: tuple[int, ...] = (BASELINE_VISITNUM, *VISIT_LABELS)
POST_BASELINE_VISITS: tuple[int, ...] = tuple(VISIT_LABELS)
# Every proteomics VISITNUM -> week, and back.
VISIT_TO_WEEK: dict[int, int] = {BASELINE_VISITNUM: BASELINE_WEEK, **VISIT_LABELS}
WEEK_TO_VISIT: dict[int, int] = {week: visit for visit, week in VISIT_TO_WEEK.items()}
WEEKS: tuple[int, ...] = tuple(VISIT_TO_WEEK.values())
POST_BASELINE_WEEKS: tuple[int, ...] = tuple(VISIT_LABELS.values())
# The two post-baseline weeks by name.
EARLY_WEEK, LATE_WEEK = POST_BASELINE_WEEKS

# Covariate set of the primary MMRM and the mediation models (age and sex); ``COVAR`` names
# the output directory ``<mmrm_root>/surmount5_<platform>/covar_AGE_SEX``.
COVAR_LABEL = "AGE_SEX"
COVAR = f"covar_{COVAR_LABEL}"
# The covariates as banners print them ("age, sex"), from the label.
COVAR_PHRASE = ", ".join(word.lower() for word in COVAR_LABEL.split("_"))
# Categorical covariate of those models; the continuous ones are set per package.
CAT_COVARIATES: tuple[str, ...] = ("SEX",)
