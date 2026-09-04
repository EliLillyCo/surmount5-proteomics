"""Output generation for MMRM analysis.

Produces PCBL (percent change from baseline), across-treatment, within-treatment,
cross treatment-time, and baseline comparison output files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from .constants import (
    ACROSS_TRT_COLS,
    BASE_METADATA_COLS,
    PCBL_OUTPUT_COLS,
    WITHIN_TRT_COLS,
)
from .fitting import apply_fdr_correction

# ============================================================================
# PCBL TRANSFORMATION
# ============================================================================


def transform_to_pcbl(lsmeans: pl.DataFrame) -> pl.DataFrame:
    """Transform log2 LSMeans to percent change from baseline."""
    result = lsmeans.with_columns(
        [
            ((pl.col("LSM") - 1) * 100).alias("PCBL"),
            (np.log(2) * pl.col("LSM") * pl.col("StdErr") * 100).alias("SE_PCBL"),
            ((2 ** (pl.col("LSmean") + pl.col("StdErr")) - 1) * 100).alias(
                "PCBL_upper_1SE"
            ),
            ((2 ** (pl.col("LSmean") - pl.col("StdErr")) - 1) * 100).alias(
                "PCBL_lower_1SE"
            ),
            ((2 ** pl.col("Lower") - 1) * 100).alias("PCBL_lower_CI"),
            ((2 ** pl.col("Upper") - 1) * 100).alias("PCBL_upper_CI"),
        ]
    ).with_columns((pl.col("SE_PCBL") / pl.col("PCBL").abs()).alias("CV_PCBL"))

    return result


# ============================================================================
# PCBL FILE SAVING
# ============================================================================


def save_pcbl_files(
    pcbl_data: pl.DataFrame,
    output_dir: Path,
    annotation: pl.DataFrame,
    study_name: str = "STUDY",
    covar_label: str = "none",
    protein_id_col: str = "OlinkID",
) -> pl.DataFrame:
    """Generate and save PCBL files by treatment and visit."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcbl_data = pcbl_data.join(
        annotation, left_on="marker", right_on=protein_id_col, how="left"
    )

    treatments = pcbl_data["trt"].unique().to_list()
    visits = pcbl_data["time"].unique().sort().to_list()

    summary_data = []

    for trt in treatments:
        for visit in visits:
            subset = pcbl_data.filter(
                (pl.col("trt") == trt) & (pl.col("time") == visit)
            )

            subset = apply_fdr_correction(subset)

            filename = f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_PCBLRes_{trt}@{visit}_py.csv"
            subset.select(PCBL_OUTPUT_COLS).write_csv(output_dir / filename)

            summary_data.append(
                {
                    "Treatment": trt,
                    "Visit": visit,
                    "N_pval_0.05": (subset["pVal"] < 0.05).sum(),
                    "N_fdr_0.05": (subset["fdr"] < 0.05).sum(),
                }
            )

    summary = pl.DataFrame(summary_data)
    summary.write_csv(
        output_dir
        / f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_PCBLRes_py.csv"
    )
    print(f"\n{summary}")

    return pcbl_data


# ============================================================================
# COMPARISON OUTPUT GENERATION
# ============================================================================


def _compute_fc_columns(
    comp_data: pl.DataFrame,
    lsm1: pl.DataFrame,
    lsm2: pl.DataFrame,
) -> pl.DataFrame:
    """Join LSM values and compute fold change and SE-FC."""
    return (
        comp_data.join(lsm1, on="marker", how="left")
        .join(lsm2, on="marker", how="left")
        .with_columns(
            [
                (pl.col("LSM1") / pl.col("LSM2")).alias("FC"),
                (
                    pl.col("LSM1")
                    / pl.col("LSM2")
                    * (
                        (pl.col("SE-LSM1") / pl.col("LSM1")).pow(2)
                        + (pl.col("SE-LSM2") / pl.col("LSM2")).pow(2)
                    ).sqrt()
                ).alias("SE-FC"),
            ]
        )
    )


def generate_comparison_output(
    contrasts: pl.DataFrame,
    lsmeans: pl.DataFrame,
    annotation: pl.DataFrame,
    output_dir: Path,
    comparison_type: Literal["across", "within"],
    covar_label: str = "none",
    study_name: str = "STUDY",
    protein_id_col: str = "OlinkID",
    placebo_comparisons_only: bool = True,
) -> pl.DataFrame | None:
    """Generate across-treatment or within-treatment comparison outputs."""
    if comparison_type == "across":
        subdir = "acTrt"
    else:
        subdir = "withinTrt"
    output_dir = Path(output_dir) / "finalRes" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter based on comparison type
    if comparison_type == "across":
        filtered = contrasts.filter(
            (pl.col("trt") != pl.col("_trt")) & (pl.col("time") == pl.col("_time"))
        ).rename({"trt": "trt1", "_trt": "trt2"})

        # Swap trt1/trt2 when Placebo is in trt1
        filtered = (
            filtered.with_columns(
                [
                    pl.when(pl.col("trt1").str.to_lowercase().str.contains("placebo"))
                    .then(pl.col("trt2"))
                    .otherwise(pl.col("trt1"))
                    .alias("_tmp_trt1"),
                    pl.when(pl.col("trt1").str.to_lowercase().str.contains("placebo"))
                    .then(pl.col("trt1"))
                    .otherwise(pl.col("trt2"))
                    .alias("_tmp_trt2"),
                    pl.when(pl.col("trt1").str.to_lowercase().str.contains("placebo"))
                    .then(-pl.col("diff"))
                    .otherwise(pl.col("diff"))
                    .alias("diff"),
                ]
            )
            .drop(["trt1", "trt2"])
            .rename({"_tmp_trt1": "trt1", "_tmp_trt2": "trt2"})
        )

        keep_cols = ACROSS_TRT_COLS
    else:  # within
        filtered = contrasts.filter(
            (pl.col("trt") == pl.col("_trt")) & (pl.col("time") != pl.col("_time"))
        )
        keep_cols = WITHIN_TRT_COLS

    if len(filtered) == 0:
        return None

    # Add annotation
    filtered = filtered.join(
        annotation, left_on="marker", right_on=protein_id_col, how="left"
    )

    # Filter to placebo comparisons only if requested
    if placebo_comparisons_only and comparison_type == "across":
        all_treatments = set(
            filtered["trt1"].unique().to_list()
            + filtered["trt2"].unique().to_list()
        )

        has_placebo = any("placebo" in str(trt).lower() for trt in all_treatments)

        if has_placebo:
            filtered = filtered.filter(
                pl.col("trt1").str.to_lowercase().str.contains("placebo")
                | pl.col("trt2").str.to_lowercase().str.contains("placebo")
            )

    summary_data = []

    if comparison_type == "across":
        times = filtered["time"].unique().sort().to_list()

        for t in times:
            tmp = filtered.filter(pl.col("time") == str(t))
            tmp = apply_fdr_correction(tmp)

            unique_comparisons = tmp.select(["trt1", "trt2"]).unique().to_dicts()

            for comp in unique_comparisons:
                trt1_val = comp["trt1"]
                trt2_val = comp["trt2"]

                comp_data = tmp.filter(
                    (pl.col("trt1") == trt1_val) & (pl.col("trt2") == trt2_val)
                )

                lsm_trt1 = (
                    lsmeans.filter(
                        (pl.col("trt") == trt1_val) & (pl.col("time") == str(t))
                    )
                    .select(["marker", "LSM", "SE-LSM"])
                    .rename({"LSM": "LSM1", "SE-LSM": "SE-LSM1"})
                )

                lsm_trt2 = (
                    lsmeans.filter(
                        (pl.col("trt") == trt2_val) & (pl.col("time") == str(t))
                    )
                    .select(["marker", "LSM", "SE-LSM"])
                    .rename({"LSM": "LSM2", "SE-LSM": "SE-LSM2"})
                )

                output = (
                    _compute_fc_columns(comp_data, lsm_trt1, lsm_trt2)
                    .select(
                        [
                            c
                            for c in keep_cols
                            if c in comp_data.columns or c in ["FC", "SE-FC"]
                        ]
                    )
                    .sort("marker")
                )

                trt_label = f"{trt1_val}VS{trt2_val}"
                filename = f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_acrossTrts_resCmps_{trt_label}@{t}_py.csv"
                output.write_csv(output_dir / filename)

                summary_data.append(
                    {
                        "Comparison": f"{trt_label}@{t}",
                        "N(pVal < 0.05)": (output["pVal"] < 0.05).sum(),
                        "N(fdr < 0.05)": (output["fdr"] < 0.05).sum(),
                    }
                )

        summary_file = f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_acrossTrts_summary_py.csv"

    else:  # within
        unique_comparisons = (
            filtered.select(["trt", "time", "_time"]).unique().to_dicts()
        )

        for comp in unique_comparisons:
            trt = comp["trt"]
            time1 = comp["time"]
            time2 = comp["_time"]

            comp_data = filtered.filter(
                (pl.col("trt") == trt)
                & (pl.col("time") == time1)
                & (pl.col("_time") == time2)
            )
            comp_data = apply_fdr_correction(comp_data)

            lsm_t1 = (
                lsmeans.filter((pl.col("trt") == trt) & (pl.col("time") == time1))
                .select(["marker", "LSM", "SE-LSM"])
                .rename({"LSM": "LSM1", "SE-LSM": "SE-LSM1"})
            )

            lsm_t2 = (
                lsmeans.filter((pl.col("trt") == trt) & (pl.col("time") == time2))
                .select(["marker", "LSM", "SE-LSM"])
                .rename({"LSM": "LSM2", "SE-LSM": "SE-LSM2"})
            )

            output = (
                _compute_fc_columns(comp_data, lsm_t1, lsm_t2)
                .with_columns(
                    [
                        pl.lit(time1).alias("time1"),
                        pl.lit(time2).alias("time2"),
                    ]
                )
                .select(
                    [
                        c
                        for c in keep_cols
                        if c in comp_data.columns
                        or c in ["FC", "SE-FC", "time1", "time2"]
                    ]
                )
                .sort("marker")
            )

            filename = f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_withinTrt_resCmps_{trt}@{time1}VS{time2}_py.csv"
            output.write_csv(output_dir / filename)

            summary_data.append(
                {
                    "Comparison": f"{time1} vs {time2}@{trt}",
                    "N(pVal < 0.05)": (output["pVal"] < 0.05).sum(),
                    "N(fdr < 0.05)": (output["fdr"] < 0.05).sum(),
                }
            )

        summary_file = f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_withinTrt_summary_py.csv"

    summary = pl.DataFrame(summary_data)
    summary.write_csv(output_dir / summary_file)

    return summary


# ============================================================================
# BASELINE OUTPUT GENERATION
# ============================================================================


def generate_baseline_output(
    lsmeans: pl.DataFrame,
    contrasts: pl.DataFrame,
    annotation: pl.DataFrame,
    output_dir: Path,
    covar_label: str = "none",
    study_name: str = "STUDY",
    type3: pl.DataFrame | None = None,
    protein_id_col: str = "OlinkID",
) -> None:
    """Generate baseline comparison outputs."""
    output_dir = Path(output_dir) / "finalRes" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(lsmeans) > 0:
        lsm_output = (
            lsmeans.join(
                annotation, left_on="marker", right_on=protein_id_col, how="left"
            )
            .pipe(apply_fdr_correction)
            .sort("marker")
        )
        lsm_output.write_csv(
            output_dir
            / f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_Baseline_LSMRes_py.csv"
        )

    if len(contrasts) > 0:
        diff_output = (
            contrasts.join(
                annotation, left_on="marker", right_on=protein_id_col, how="left"
            )
            .pipe(apply_fdr_correction)
            .sort("marker")
        )
        diff_output.write_csv(
            output_dir
            / f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_Baseline_DiffRes_py.csv"
        )

    if type3 is not None and len(type3) > 0:
        type3_output = (
            type3.join(
                annotation, left_on="marker", right_on=protein_id_col, how="left"
            )
            .pipe(apply_fdr_correction)
            .sort("marker")
        )
        type3_output.write_csv(
            output_dir
            / f"{study_name}_covar_{covar_label}_proteomics_olinkAnalysis_Baseline_TypeIII_py.csv"
        )
