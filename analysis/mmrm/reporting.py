"""Output generation for MMRM analysis.

Produces PCBL (percent change from baseline), between-treatment, within-treatment
and baseline comparison output files.

Contrast orientation: treatment contrasts are ``treatment - control`` (TZP - SEMA) and
within-arm time contrasts are ``later - earlier`` (week 72 - week 24); the fit fixes this
through the factor reference levels. Every ``fdr`` column is Benjamini-Hochberg
(``analysis.statistics.bh_fdr``) over the proteins of one written comparison: one
treatment-pair x visit, one arm x visit (PCBL), or one arm's within-arm time contrast.
The baseline tables are one family per file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from analysis.mmrm_outputs import (
    across_treatment_name,
    across_treatment_summary_name,
    baseline_diff_res_name,
    baseline_lsm_res_name,
    pcbl_name,
    pcbl_summary_name,
    within_treatment_name,
    within_treatment_summary_name,
)
from analysis.statistics import FDR_THRESHOLD, bh_fdr

from .constants import (
    ACROSS_TRT_COLS,
    PCBL_OUTPUT_COLS,
    WITHIN_TRT_COLS,
)

# The raw contrast columns (``rinterface``) the final tables carry, under their final names.
_CONTRAST_RENAMES = {"diff": "estimate", "StdErr": "SE"}
_DERIVED_COLS = ("FC", "SE-FC", "estimate", "SE")


def apply_fdr_correction(df: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Add ``fdr``: BH over ``pVal`` (``pvalue`` when absent), within each ``by`` group.

    Args:
        df: Frame with a ``pVal`` (or ``pvalue``) column.
        by: Columns defining one comparison family; None treats the whole frame as one.

    Returns:
        ``df`` with an ``fdr`` column.
    """
    pval_col = "pVal" if "pVal" in df.columns else "pvalue"
    if by is None:
        return df.with_columns(pl.Series("fdr", bh_fdr(df[pval_col].to_numpy())))
    return df.with_columns(
        pl.col(pval_col).map_batches(lambda s: pl.Series(bh_fdr(s.to_numpy()))).over(by).alias("fdr")
    )


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
    """Generate and save PCBL files by treatment and visit.

    Treatments and visits are taken in sorted order and each file's rows are sorted by ``marker``, so
    the files and the summary do not depend on the row order of ``pcbl_data``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcbl_data = pcbl_data.join(
        annotation, left_on="marker", right_on=protein_id_col, how="left"
    )

    treatments = pcbl_data["trt"].unique().sort().to_list()
    visits = pcbl_data["time"].unique().sort().to_list()

    summary_data = []

    for trt in treatments:
        for visit in visits:
            subset = pcbl_data.filter(
                (pl.col("trt") == trt) & (pl.col("time") == visit)
            ).sort("marker", "Assay", maintain_order=True)

            subset = apply_fdr_correction(subset)

            filename = pcbl_name(study_name, covar_label, trt, visit)
            subset.select(PCBL_OUTPUT_COLS).write_csv(output_dir / filename)

            summary_data.append(
                {
                    "Treatment": trt,
                    "Visit": visit,
                    f"N_pval_{FDR_THRESHOLD}": (subset["pVal"] < FDR_THRESHOLD).sum(),
                    f"N_fdr_{FDR_THRESHOLD}": (subset["fdr"] < FDR_THRESHOLD).sum(),
                }
            )

    summary = pl.DataFrame(summary_data)
    summary.write_csv(
        output_dir / pcbl_summary_name(study_name, covar_label)
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
    """Join the two LS-means, compute ``FC`` and ``SE-FC``, and name the contrast's own columns.

    ``FC`` is ``LSM1 / LSM2``. ``SE-FC`` is ``FC * sqrt((SE-LSM1 / LSM1)^2 + (SE-LSM2 / LSM2)^2)``,
    the independence approximation from the two LS-means, not the contrast's standard error.
    The contrast rows' own ``diff`` and ``StdErr`` (log2 scale) become ``estimate`` and ``SE``;
    ``df`` is kept as it is.

    Args:
        comp_data: The raw contrast rows of one comparison (``diff``, ``StdErr``, ``df``, ...).
        lsm1: ``marker``, ``LSM1``, ``SE-LSM1`` of the first arm or visit.
        lsm2: ``marker``, ``LSM2``, ``SE-LSM2`` of the second.

    Returns:
        ``comp_data`` with the LS-means, ``FC``, ``SE-FC``, ``estimate`` and ``SE``.
    """
    return (
        comp_data.join(lsm1, on="marker", how="left")
        .join(lsm2, on="marker", how="left")
        .with_columns(
            [
                *(pl.col(raw).alias(final) for raw, final in _CONTRAST_RENAMES.items()),
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
) -> pl.DataFrame | None:
    """Write the between-treatment or within-treatment comparison files and their summary.

    Each file is one comparison, one row per marker: the two LS-means and ``FC`` / ``SE-FC``
    (``_compute_fc_columns``), ``pVal`` and ``fdr``, and the emmeans contrast's own
    ``estimate``, ``SE`` and ``df`` on the log2 scale (``constants.CONTRAST_COLS``).

    Args:
        contrasts: Every pairwise contrast of the change model (``rinterface``): ``marker``,
            ``trt``, ``_trt``, ``time``, ``_time``, ``diff``, ``StdErr``, ``df``, ``pVal``.
        lsmeans: The change model's LS-means with ``LSM`` and ``SE-LSM``.
        annotation: Per-marker annotation joined on ``protein_id_col``.
        output_dir: The covariate output directory; files go under ``finalRes/acTrt`` or
            ``finalRes/withinTrt``.
        comparison_type: ``across`` (treatment pairs at one visit) or ``within`` (visit pairs
            in one arm).
        covar_label: Covariate label of the file names.
        study_name: Study stem of the file names.
        protein_id_col: The annotation's marker column.

    Returns:
        The summary of significant counts per comparison, or None when there is no such
        comparison.
    """
    subdir = "acTrt" if comparison_type == "across" else "withinTrt"
    output_dir = Path(output_dir) / "finalRes" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter based on comparison type
    if comparison_type == "across":
        filtered = contrasts.filter(
            (pl.col("trt") != pl.col("_trt")) & (pl.col("time") == pl.col("_time"))
        ).rename({"trt": "trt1", "_trt": "trt2"})

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

    summary_data = []

    if comparison_type == "across":
        times = filtered["time"].unique().sort().to_list()

        for t in times:
            tmp = filtered.filter(pl.col("time") == str(t))
            tmp = apply_fdr_correction(tmp, ["trt1", "trt2"])

            unique_comparisons = (
                tmp.select(["trt1", "trt2"]).unique().sort(["trt1", "trt2"]).to_dicts()
            )

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
                            if c in comp_data.columns or c in _DERIVED_COLS
                        ]
                    )
                    .sort("marker")
                )

                trt_label = f"{trt1_val}VS{trt2_val}"
                filename = across_treatment_name(study_name, covar_label, trt_label, t)
                output.write_csv(output_dir / filename)

                summary_data.append(
                    {
                        "Comparison": f"{trt_label}@{t}",
                        f"N(pVal < {FDR_THRESHOLD})": (output["pVal"] < FDR_THRESHOLD).sum(),
                        f"N(fdr < {FDR_THRESHOLD})": (output["fdr"] < FDR_THRESHOLD).sum(),
                    }
                )

        summary_file = across_treatment_summary_name(study_name, covar_label)

    else:  # within
        filtered = apply_fdr_correction(filtered, ["trt", "time", "_time"])
        unique_comparisons = (
            filtered.select(["trt", "time", "_time"]).unique().sort(["trt", "time", "_time"]).to_dicts()
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
                        or c in (*_DERIVED_COLS, "time1", "time2")
                    ]
                )
                .sort("marker")
            )

            filename = within_treatment_name(study_name, covar_label, trt, time1, time2)
            output.write_csv(output_dir / filename)

            summary_data.append(
                {
                    "Comparison": f"{time1} vs {time2}@{trt}",
                    f"N(pVal < {FDR_THRESHOLD})": (output["pVal"] < FDR_THRESHOLD).sum(),
                    f"N(fdr < {FDR_THRESHOLD})": (output["fdr"] < FDR_THRESHOLD).sum(),
                }
            )

        summary_file = within_treatment_summary_name(study_name, covar_label)

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
            .sort(["marker", "trt"])
        )
        lsm_output.write_csv(
            output_dir / baseline_lsm_res_name(study_name, covar_label)
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
            output_dir / baseline_diff_res_name(study_name, covar_label)
        )
