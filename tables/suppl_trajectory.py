"""Within-arm temporal response trajectory table.

For each platform, emits one row per (protein × arm). Each row carries:
  - Annotation (UniProt / Gene_Symbol / Assay or Target / platform ID)
  - Arm (TZP / SEMA) and Trajectory class (e.g. "Late-onset ↓", "Null")
  - PCBL vs baseline at Wk24 and Wk72 (estimate, SE, P-value, FDR)
  - Within-arm Wk24-vs-Wk72 contrast (log2FC, SE, P-value, FDR)

Null-trajectory rows are retained for a complete reference. Two sheets:
Olink, SomaScan.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import COVAR, mmrm_dir  # noqa: E402

from _annotations import build_marker_annotations  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
_LN2 = math.log(2)

ARMS = {"TZP": "TZP15mgorMTD", "SEMA": "SEMA2.4mgorMTD"}


def _load_pcbl(platform: str, arm_full: str, week: int) -> pl.DataFrame:
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "pcblRes"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_PCBLRes_{arm_full}@{week}_py.csv"
    )
    return pl.read_csv(path).select(
        "marker",
        pl.col("PCBL").alias(f"PCBL_Wk{week}"),
        pl.col("SE_PCBL").alias(f"SE_Wk{week}"),
        pl.col("pVal").alias(f"P-value_Wk{week}"),
        pl.col("fdr").alias(f"FDR_Wk{week}"),
    )


def _load_within(platform: str, arm_full: str) -> pl.DataFrame:
    """Load within-arm Wk24 vs Wk72 contrast. FC is linear; convert to log2."""
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "withinTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_withinTrt_resCmps_{arm_full}@24VS72_py.csv"
    )
    return (
        pl.read_csv(path)
        .with_columns(
            pl.col("FC").log(base=2).alias("log2FC_Wk24vsWk72"),
            (pl.col("SE-FC") / (pl.col("FC") * _LN2)).alias("SE_Wk24vsWk72"),
        )
        .select(
            "marker",
            "log2FC_Wk24vsWk72",
            "SE_Wk24vsWk72",
            pl.col("pVal").alias("P-value_Wk24vsWk72"),
            pl.col("fdr").alias("FDR_Wk24vsWk72"),
        )
    )


def _load_trajectory(platform: str) -> pl.DataFrame:
    """Long-format trajectory table: marker, Arm, Trajectory."""
    df = pl.read_parquet(ANALYSIS_DIR / "outputs" / f"trajectory_{platform}.parquet")
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

    arm_blocks = []
    for arm_short, arm_full in ARMS.items():
        pcbl_24 = _load_pcbl(platform, arm_full, 24)
        pcbl_72 = _load_pcbl(platform, arm_full, 72)
        within = _load_within(platform, arm_full)
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
        "log2FC_Wk24vsWk72",
        "SE_Wk24vsWk72",
    ]

    # Keep TZP before SEMA, then sort by gene then marker for searchability.
    arm_order = pl.col("Arm").replace_strict(
        {"TZP": 0, "SEMA": 1}, return_dtype=pl.Int8
    )

    return (
        annot.join(combined, on="marker", how="inner")
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
            "log2FC_Wk24vsWk72",
            "SE_Wk24vsWk72",
            "P-value_Wk24vsWk72",
            "FDR_Wk24vsWk72",
        )
        .sort(["Gene_Symbol", id_col, arm_order])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
