"""Trajectory classification for within-arm percent change from baseline.

Classifies each protein into a temporal response category based on:
- PCBL significance (FDR < 0.05) at Wk24 and Wk72
- Direction of PCBL at each timepoint
- Within-treatment 24-vs-72 fold-change significance (FDR < 0.05)

Trajectory classes:
    Progressive ↓ : sig negative at both visits, significantly more negative at 72
    Sustained ↓    : sig negative at both visits, no significant change between visits
    Transient ↓    : sig negative at 24 only
    Late-onset ↓   : sig negative at 72 only
    Late-onset ↑   : sig positive at 72 only
    Transient ↑    : sig positive at 24 only
    Sustained ↑    : sig positive at both visits, no significant change between visits
    Progressive ↑ : sig positive at both visits, significantly more positive at 72
    Reversal ↓↑    : sig negative at 24, sig positive at 72
    Reversal ↑↓    : sig positive at 24, sig negative at 72
    Null           : not significant at either visit
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SCRIPT_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _SCRIPT_DIR.parent
_OUTPUT_DIR = _ANALYSIS_DIR / "outputs"

sys.path.insert(0, str(_ANALYSIS_DIR.parent))
from paths import COVAR, mmrm_dir  # noqa: E402

FDR_THRESHOLD = 0.05


def _load_pcbl(platform: str, arm: str, visit: int) -> pl.LazyFrame:
    """Load a single PCBL results file."""
    arm_map = {"TZP": "TZP15mgorMTD", "SEMA": "SEMA2.4mgorMTD"}
    arm_label = arm_map[arm]
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "pcblRes"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_PCBLRes_{arm_label}@{visit}_py.csv"
    )
    return pl.scan_csv(path)


def _load_within_trt(platform: str, arm: str) -> pl.LazyFrame:
    """Load within-treatment 24-vs-72 comparison."""
    arm_map = {"TZP": "TZP15mgorMTD", "SEMA": "SEMA2.4mgorMTD"}
    arm_label = arm_map[arm]
    path = (
        mmrm_dir(platform)
        / "finalRes"
        / "withinTrt"
        / f"surmount5_{COVAR}_proteomics_olinkAnalysis_withinTrt_resCmps_{arm_label}@24VS72_py.csv"
    )
    return pl.scan_csv(path)


def classify_arm(platform: str, arm: str) -> pl.LazyFrame:
    """Classify trajectory for one arm. Returns marker, Assay, trajectory, PCBL values.

    No QC filter is applied; the `allQC` flag is propagated to the output so
    downstream consumers can annotate WARN analytes if needed.
    """
    pcbl_24 = _load_pcbl(platform, arm, 24)
    pcbl_72 = _load_pcbl(platform, arm, 72)
    within = _load_within_trt(platform, arm)

    combined = (
        pcbl_24.select(
            "marker",
            "Assay",
            pl.col("allQC").alias("allQC_24"),
            pl.col("PCBL").alias("PCBL_24"),
            pl.col("fdr").alias("fdr_24"),
        )
        .join(
            pcbl_72.select(
                "marker",
                pl.col("allQC").alias("allQC_72"),
                pl.col("PCBL").alias("PCBL_72"),
                pl.col("fdr").alias("fdr_72"),
            ),
            on="marker",
            how="inner",
        )
        .join(
            within.select(
                "marker",
                pl.col("allQC").alias("allQC_within"),
                pl.col("FC").alias("FC_24v72"),
                pl.col("fdr").alias("fdr_within"),
            ),
            on="marker",
            how="inner",
        )
    )

    classified = combined.with_columns(
        pl.when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") < 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") > 0)
        )
        .then(pl.lit("Reversal ↓↑"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") > 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") < 0)
        )
        .then(pl.lit("Reversal ↑↓"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") < 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") < 0)
            & (pl.col("fdr_within") < FDR_THRESHOLD)
            & (pl.col("FC_24v72") > 1)
        )
        .then(pl.lit("Progressive ↓"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") < 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") < 0)
            & ((pl.col("fdr_within") >= FDR_THRESHOLD) | (pl.col("FC_24v72") <= 1))
        )
        .then(pl.lit("Sustained ↓"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") < 0)
            & (pl.col("fdr_72") >= FDR_THRESHOLD)
        )
        .then(pl.lit("Transient ↓"))
        .when(
            (pl.col("fdr_24") >= FDR_THRESHOLD)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") < 0)
        )
        .then(pl.lit("Late-onset ↓"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") > 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") > 0)
            & (pl.col("fdr_within") < FDR_THRESHOLD)
            & (pl.col("FC_24v72") < 1)
        )
        .then(pl.lit("Progressive ↑"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") > 0)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") > 0)
            & ((pl.col("fdr_within") >= FDR_THRESHOLD) | (pl.col("FC_24v72") >= 1))
        )
        .then(pl.lit("Sustained ↑"))
        .when(
            (pl.col("fdr_24") < FDR_THRESHOLD)
            & (pl.col("PCBL_24") > 0)
            & (pl.col("fdr_72") >= FDR_THRESHOLD)
        )
        .then(pl.lit("Transient ↑"))
        .when(
            (pl.col("fdr_24") >= FDR_THRESHOLD)
            & (pl.col("fdr_72") < FDR_THRESHOLD)
            & (pl.col("PCBL_72") > 0)
        )
        .then(pl.lit("Late-onset ↑"))
        .otherwise(pl.lit("Null"))
        .alias("trajectory")
    )

    # Compose a single per-marker QC summary: PASS only if all three contributing
    # tests are PASS, else WARN.
    classified = classified.with_columns(
        pl.when(
            (pl.col("allQC_24") == "PASS")
            & (pl.col("allQC_72") == "PASS")
            & (pl.col("allQC_within") == "PASS")
        )
        .then(pl.lit("PASS"))
        .otherwise(pl.lit("WARN"))
        .alias("allQC")
    )

    return classified.select(
        "marker", "Assay", "allQC", "trajectory", "PCBL_24", "PCBL_72"
    )


def classify_platform(platform: str) -> pl.DataFrame:
    """Run trajectory classification for both arms, merge into single DataFrame."""
    platform_key = platform

    tzp = classify_arm(platform_key, "TZP").rename(
        {
            "trajectory": "trajectory_TZP",
            "PCBL_24": "PCBL_TZP_24",
            "PCBL_72": "PCBL_TZP_72",
            "allQC": "allQC_TZP",
        }
    )
    sema = classify_arm(platform_key, "SEMA").rename(
        {
            "trajectory": "trajectory_SEMA",
            "PCBL_24": "PCBL_SEMA_24",
            "PCBL_72": "PCBL_SEMA_72",
            "allQC": "allQC_SEMA",
        }
    )

    merged = tzp.join(
        sema.select(
            "marker", "trajectory_SEMA", "PCBL_SEMA_24", "PCBL_SEMA_72", "allQC_SEMA"
        ),
        on="marker",
        how="inner",
    )

    return merged.collect()


if __name__ == "__main__":
    for platform in ("olink", "soma"):
        df = classify_platform(platform)
        print(f"\n{'='*60}")
        print(f"  {platform.upper()} — Trajectory Classification (TZP-grouped)")
        print(f"{'='*60}")
        warn_n = df.filter(
            (pl.col("allQC_TZP") == "WARN") | (pl.col("allQC_SEMA") == "WARN")
        ).height
        print(f"  Total proteins: {df.height}  (of which WARN-QC: {warn_n})")

        non_null = df.filter(pl.col("trajectory_TZP") != "Null")
        print(f"  Non-null trajectories (TZP): {non_null.height}")

        print("\n  TZP trajectory counts:")
        counts = (
            df.group_by("trajectory_TZP")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        for row in counts.iter_rows(named=True):
            print(f"    {row['trajectory_TZP']:20s}  n={row['n']}")

        out_path = _OUTPUT_DIR / f"trajectory_{platform}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        print(f"\n  Saved → {out_path}")
