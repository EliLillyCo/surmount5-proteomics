"""File names of the MMRM and pathway outputs, for the figures and tables.

The writers (``analysis.pipelines.run_mmrm`` through ``analysis.mmrm.reporting``, and
``analysis.pathway.cli``) name every file ``<STUDY_STEM>_<COVAR>_...`` under
``paths.mmrm_dir(platform)``; this module is the single place the readers build those
names, so a renamed output changes here only.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.design import CONTROL_VALUE, COVAR, COVAR_LABEL, STUDY_STEM, TREAT_VALUE
from paths import mmrm_dir

# The treatment contrast every between-treatment file carries: TZP − SEMA.
ACROSS_CONTRAST = f"{TREAT_VALUE}VS{CONTROL_VALUE}"

# Directory names under a covariate directory (``paths.mmrm_dir`` is the primary model's).
RAW_OUTPUT = "rawOutput"
FINAL_RES = "finalRes"
PATHWAY_RES = "pathwayRes"
_CHG_LSMEANS_SUFFIX = "_chg_lsmeans.csv"
_CHG_DIFF_SUFFIX = "_chg_diff.csv"


def _stem(study_name: str, covar_label: str) -> str:
    return f"{study_name}_covar_{covar_label}"


def _analysis_prefix(study_name: str, covar_label: str) -> str:
    return f"{_stem(study_name, covar_label)}_proteomics_olinkAnalysis"




# File names as the writers (``analysis.mmrm``) build them for any study stem and covariate label.


def chg_lsmeans_name(study_name: str, covar_label: str) -> str:
    """``rawOutput`` name of the change model's LSmeans."""
    return f"{_stem(study_name, covar_label)}{_CHG_LSMEANS_SUFFIX}"


def chg_diff_name(study_name: str, covar_label: str) -> str:
    """``rawOutput`` name of the change model's pairwise contrasts."""
    return f"{_stem(study_name, covar_label)}{_CHG_DIFF_SUFFIX}"


def baseline_lsmeans_name(study_name: str, covar_label: str) -> str:
    """``rawOutput`` name of the baseline model's LSmeans."""
    return f"{_stem(study_name, covar_label)}_baseline_lsmeans.csv"


def baseline_diff_name(study_name: str, covar_label: str) -> str:
    """``rawOutput`` name of the baseline model's contrasts."""
    return f"{_stem(study_name, covar_label)}_baseline_diff.csv"


def pcbl_name(study_name: str, covar_label: str, arm_value: str, week: int) -> str:
    """``finalRes/pcblRes`` name of one arm x visit table."""
    return f"{_analysis_prefix(study_name, covar_label)}_PCBLRes_{arm_value}@{week}_py.csv"


def pcbl_summary_name(study_name: str, covar_label: str) -> str:
    """``finalRes/pcblRes`` name of the significant-count summary."""
    return f"{_analysis_prefix(study_name, covar_label)}_PCBLRes_py.csv"


def across_treatment_name(study_name: str, covar_label: str, contrast: str, week: int) -> str:
    """``finalRes/acTrt`` name of one ``<trt1>VS<trt2>`` contrast at ``week``."""
    return f"{_analysis_prefix(study_name, covar_label)}_acrossTrts_resCmps_{contrast}@{week}_py.csv"


def across_treatment_summary_name(study_name: str, covar_label: str) -> str:
    """``finalRes/acTrt`` name of the significant-count summary."""
    return f"{_analysis_prefix(study_name, covar_label)}_acrossTrts_summary_py.csv"


def within_treatment_name(study_name: str, covar_label: str, arm_value: str, late: int, early: int) -> str:
    """``finalRes/withinTrt`` name of one arm's ``late`` - ``early`` contrast."""
    return f"{_analysis_prefix(study_name, covar_label)}_withinTrt_resCmps_{arm_value}@{late}VS{early}_py.csv"


def within_treatment_summary_name(study_name: str, covar_label: str) -> str:
    """``finalRes/withinTrt`` name of the significant-count summary."""
    return f"{_analysis_prefix(study_name, covar_label)}_withinTrt_summary_py.csv"


def baseline_lsm_res_name(study_name: str, covar_label: str) -> str:
    """``finalRes/baseline`` name of the baseline LSmeans table."""
    return f"{_analysis_prefix(study_name, covar_label)}_Baseline_LSMRes_py.csv"


def baseline_diff_res_name(study_name: str, covar_label: str) -> str:
    """``finalRes/baseline`` name of the baseline TZP - SEMA difference table."""
    return f"{_analysis_prefix(study_name, covar_label)}_Baseline_DiffRes_py.csv"


# Numeric result columns of the combined pathway CSVs need declared types when a file has
# headers but no rows.
PATHWAY_NUMERIC_SCHEMAS = {
    "camera": {
        "NOM p-val": pl.Float64,
        "FDR q-val": pl.Float64,
        "N_Genes": pl.Int64,
        "Total_Genes": pl.Int64,
        "inter_gene_cor": pl.Float64,
    },
    "ora": {
        "P-value": pl.Float64,
        "Adjusted P-value": pl.Float64,
        "Odds Ratio": pl.Float64,
        "Combined Score": pl.Float64,
    },
}


def pathway_combined_name(method: str) -> str:
    """Name of the combined pathway table of ``method`` (``camera`` or ``ora``)."""
    return f"{method}_combined.csv"


def pathway_contrast_dir(covar_dir: Path, label: str) -> Path:
    """``<covar_dir>/pathwayRes/acTrt/<label>``: where one contrast's pathway tables go."""
    return Path(covar_dir) / PATHWAY_RES / "acTrt" / label


def single_raw_csv(covar_dir: Path, suffix: str) -> Path:
    """The one ``rawOutput/*<suffix>`` file of a covariate directory; any other count raises."""
    raw = Path(covar_dir) / RAW_OUTPUT
    files = sorted(raw.glob(f"*{suffix}"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one *{suffix} in {raw}; found {len(files)}")
    return files[0]


def chg_diff_csv_in(covar_dir: Path) -> Path:
    """``chg_diff_csv`` for any covariate directory (the pathway batch discovers them)."""
    return single_raw_csv(covar_dir, _CHG_DIFF_SUFFIX)


def across_treatment_csv(platform: str, week: int) -> Path:
    """TZP − SEMA contrast at ``week`` (``finalRes/acTrt``)."""
    name = across_treatment_name(STUDY_STEM, COVAR_LABEL, ACROSS_CONTRAST, week)
    return mmrm_dir(platform) / FINAL_RES / "acTrt" / name


def pcbl_csv(platform: str, arm_value: str, week: int) -> Path:
    """Within-arm percent change from baseline at ``week`` (``finalRes/pcblRes``)."""
    return mmrm_dir(platform) / FINAL_RES / "pcblRes" / pcbl_name(STUDY_STEM, COVAR_LABEL, arm_value, week)


def within_treatment_csv(platform: str, arm_value: str, late: int, early: int) -> Path:
    """Within-arm ``late`` − ``early`` week contrast (``finalRes/withinTrt``)."""
    name = within_treatment_name(STUDY_STEM, COVAR_LABEL, arm_value, late, early)
    return mmrm_dir(platform) / FINAL_RES / "withinTrt" / name


def chg_diff_csv(platform: str) -> Path:
    """Every pairwise contrast of the change-from-baseline fit (``rawOutput``).

    Columns ``marker, time, trt, _time, _trt, diff, StdErr, df, ...``: the contrast
    estimate with its own standard error and Kenward-Roger df; the ``finalRes``
    fold-change files also carry the contrast's estimate, SE and df.
    """
    return chg_diff_csv_in(mmrm_dir(platform))


def with_contrast_stderr(
    final: pl.DataFrame, raw: pl.DataFrame, *, trt: str, _trt: str, time: int, _time: int
) -> pl.DataFrame:
    """Add the MMRM contrast ``StdErr`` and ``df`` from ``rawOutput`` to a ``finalRes`` frame.

    ``final`` is one contrast's ``finalRes`` table (one row per ``marker``); ``raw`` is
    ``chg_diff_csv``. The raw rows of ``(trt, _trt, time, _time)`` are joined 1:1 on
    ``marker``; any marker missing from, or duplicated in, either side raises.
    """
    contrast = raw.filter(
        (pl.col("trt") == trt)
        & (pl.col("_trt") == _trt)
        & (pl.col("time") == time)
        & (pl.col("_time") == _time)
    ).select("marker", "StdErr", "df")
    out = final.join(contrast, on="marker", how="left", validate="1:1")
    if out["StdErr"].null_count() or out.height != contrast.height:
        raise ValueError(
            f"finalRes ({out.height} rows) and rawOutput contrast ({contrast.height} rows) "
            f"for {trt} - {_trt} @{time}/{_time} do not match one-to-one on marker"
        )
    return out


def chg_lsmeans_csv(platform: str) -> Path:
    """Within-arm LSmeans of the change-from-baseline fit (``rawOutput``).

    Columns ``marker, time, trt, LSmean, StdErr, DF, ...``: the log2 change from baseline
    with its own standard error. ``finalRes/pcblRes`` derives ``PCBL`` and ``SE_PCBL``
    from these (``analysis.mmrm.reporting.transform_to_pcbl``).
    """
    return single_raw_csv(mmrm_dir(platform), _CHG_LSMEANS_SUFFIX)


def with_lsmean_stderr(
    final: pl.DataFrame, raw: pl.DataFrame, *, trt: str, time: int
) -> pl.DataFrame:
    """Add the change-model ``LSmean``, ``StdErr`` and ``DF`` from ``rawOutput`` to a
    ``finalRes/pcblRes`` frame.

    ``final`` is one arm x visit ``pcblRes`` table (one row per ``marker``); ``raw`` is
    ``chg_lsmeans_csv``. The raw rows of ``(trt, time)`` are joined 1:1 on ``marker``; any
    marker missing from, or duplicated in, either side raises.
    """
    lsm = raw.filter((pl.col("trt") == trt) & (pl.col("time") == time)).select(
        "marker", "LSmean", "StdErr", "DF"
    )
    out = final.join(lsm, on="marker", how="left", validate="1:1")
    if out["StdErr"].null_count() or out.height != lsm.height:
        raise ValueError(
            f"finalRes ({out.height} rows) and rawOutput LSmeans ({lsm.height} rows) "
            f"for {trt} @{time} do not match one-to-one on marker"
        )
    return out


def baseline_diff_csv(platform: str) -> Path:
    """Baseline TZP − SEMA difference (``finalRes/baseline``)."""
    return mmrm_dir(platform) / FINAL_RES / "baseline" / baseline_diff_res_name(STUDY_STEM, COVAR_LABEL)


def pathway_csv(platform: str, week: int, method: str) -> Path:
    """Combined pathway result of ``method`` (``camera`` or ``ora``) for the TZP − SEMA contrast.

    Args:
        platform: ``"olink"`` or ``"soma"``.
        week: Contrast week.
        method: ``"camera"`` or ``"ora"``.

    Returns:
        ``<mmrm_dir>/pathwayRes/acTrt/<contrast>@<week>/<method>_combined.csv``.
    """
    return pathway_contrast_dir(mmrm_dir(platform), f"{ACROSS_CONTRAST}@{week}") / pathway_combined_name(method)


def sexint_csv(platform: str) -> Path:
    """Sex x treatment interaction results, saved beside the primary MMRM directory."""
    return mmrm_dir(platform).parent / f"{COVAR}_sexint" / f"{STUDY_STEM}_sexint.csv"
