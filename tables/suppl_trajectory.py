"""Within-arm temporal response trajectory table.

For each platform, emits one row per (protein × arm). Each row carries:
  - Annotation (UniProt / Gene_Symbol / Assay or Target / platform ID)
  - Arm (TZP / SEMA) and Trajectory class (e.g. "Late-onset ↓", "Null")
  - PCBL vs baseline at Wk24 and Wk72 (estimate, SE, P-value, FDR)
  - Within-arm Wk72-vs-Wk24 contrast (log2FC = log2(72/24), SE, P-value, FDR)

Null-trajectory rows are included. Two sheets:
Olink, SomaScan.
"""

from __future__ import annotations

import polars as pl

from analysis.design import ARMS, EARLY_WEEK, LATE_WEEK
from analysis.mmrm_outputs import chg_diff_csv, pcbl_csv, with_contrast_stderr, within_treatment_csv

from analysis.output_layout import trajectory_path
from tables._annotations import build_marker_annotations, join_annotation


def _load_pcbl(platform: str, arm_full: str, week: int) -> pl.DataFrame:
    path = pcbl_csv(platform, arm_full, week)
    return pl.read_csv(path).select(
        "marker",
        pl.col("PCBL").alias(f"PCBL_Wk{week}"),
        pl.col("SE_PCBL").alias(f"SE_Wk{week}"),
        pl.col("pVal").alias(f"P-value_Wk{week}"),
        pl.col("fdr").alias(f"FDR_Wk{week}"),
    )


def _load_within(platform: str, arm_full: str, raw: pl.DataFrame) -> pl.DataFrame:
    """Load the within-arm week 72 - week 24 contrast. FC is linear (72/24); convert to log2.

    The SE is the contrast's own MMRM StdErr from rawOutput (log2 scale).
    """
    path = within_treatment_csv(platform, arm_full, LATE_WEEK, EARLY_WEEK)
    return (
        with_contrast_stderr(
            pl.read_csv(path), raw, trt=arm_full, _trt=arm_full, time=LATE_WEEK, _time=EARLY_WEEK
        )
        .with_columns(
            pl.col("FC").log(base=2).alias("log2FC_Wk72vsWk24"),
            pl.col("StdErr").alias("SE_Wk72vsWk24"),
        )
        .select(
            "marker",
            "log2FC_Wk72vsWk24",
            "SE_Wk72vsWk24",
            pl.col("pVal").alias("P-value_Wk72vsWk24"),
            pl.col("fdr").alias("FDR_Wk72vsWk24"),
        )
    )


def _load_trajectory(platform: str) -> pl.DataFrame:
    """Long-format trajectory table: marker, Arm, Trajectory."""
    df = pl.read_parquet(trajectory_path(platform))
    return pl.concat(
        [
            df.select(
                "marker",
                pl.lit("TZP").alias("Arm"),
                pl.col("trajectory_TZP").alias("Trajectory"),
            ),
            df.select(
                "marker",
                pl.lit("SEMA").alias("Arm"),
                pl.col("trajectory_SEMA").alias("Trajectory"),
            ),
        ]
    )


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    traj = _load_trajectory(platform)

    raw = pl.read_csv(chg_diff_csv(platform))
    arm_blocks = []
    for arm_short, arm_full in ARMS.items():
        pcbl_24 = _load_pcbl(platform, arm_full, EARLY_WEEK)
        pcbl_72 = _load_pcbl(platform, arm_full, LATE_WEEK)
        within = _load_within(platform, arm_full, raw)
        arm_data = (
            pcbl_24.join(pcbl_72, on="marker", how="full", coalesce=True)
            .join(within, on="marker", how="full", coalesce=True)
            .with_columns(pl.lit(arm_short).alias("Arm"))
        )
        arm_blocks.append(arm_data)

    combined = pl.concat(arm_blocks, how="vertical_relaxed")
    combined = combined.join(traj, on=["marker", "Arm"], how="left")

    id_col = "OlinkID" if platform == "olink" else "SeqId"
    round_cols = [
        "PCBL_Wk24",
        "SE_Wk24",
        "PCBL_Wk72",
        "SE_Wk72",
        "log2FC_Wk72vsWk24",
        "SE_Wk72vsWk24",
    ]

    # Keep TZP before SEMA, then sort by gene then marker for searchability.
    arm_order = pl.col("Arm").replace_strict(
        {"TZP": 0, "SEMA": 1}, return_dtype=pl.Int8
    )

    return (
        join_annotation(annot, combined, "trajectory")
        .with_columns(pl.col(round_cols).round(4))
        .rename({"marker": id_col})
        .select(
            id_col,
            "UniProt",
            "Gene_Symbol",
            "Assay" if platform == "olink" else "Target",
            "Arm",
            "Trajectory",
            "PCBL_Wk24",
            "SE_Wk24",
            "P-value_Wk24",
            "FDR_Wk24",
            "PCBL_Wk72",
            "SE_Wk72",
            "P-value_Wk72",
            "FDR_Wk72",
            "log2FC_Wk72vsWk24",
            "SE_Wk72vsWk24",
            "P-value_Wk72vsWk24",
            "FDR_Wk72vsWk24",
        )
        .sort(["Gene_Symbol", id_col, arm_order])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the Olink and SomaScan trajectory tables.

    Reads ``analysis/outputs/trajectory_<platform>.parquet``, the within-arm PCBL CSVs at
    weeks 24 and 72 and the within-arm week 72 vs week 24 contrast from the MMRM outputs,
    one row per protein and arm.
    """
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
