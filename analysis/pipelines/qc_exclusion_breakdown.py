"""Write the QC exclusion-flow breakdown artifact (``analysis.qc_breakdown`` builds it)."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from analysis.qc_breakdown import build_breakdown
from analysis.qc_contract import (
    IDENTITY,
    LONGITUDINAL,
    PLATFORMS,
    REASSIGNED,
    TECHNICAL,
)
from analysis.qc_flow import QC_BREAKDOWN_PATH


def write_breakdown() -> Path:
    """Build the exclusion breakdown, write the parquet and print a per-platform summary.

    Returns:
        Path of the breakdown parquet written.
    """
    breakdown = build_breakdown()
    QC_BREAKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    breakdown.write_parquet(QC_BREAKDOWN_PATH)
    print(f"[ok] wrote {QC_BREAKDOWN_PATH}")
    for platform in PLATFORMS:
        sub = breakdown.filter(pl.col("platform") == platform)
        inp = sub.filter(pl.col("stage") == "input")["n_samples"].item()
        tech = sub.filter(pl.col("stage") == TECHNICAL)["n_samples"].sum()
        ident = sub.filter(pl.col("stage") == IDENTITY)["n_samples"].sum()
        long = sub.filter(pl.col("stage") == LONGITUDINAL)["n_samples"].sum()
        moved = sub.filter(pl.col("stage") == REASSIGNED)["n_samples"].item()
        final = sub.filter((pl.col("stage") == "final") & pl.col("arm").is_null())
        arms = sub.filter((pl.col("stage") == "final") & pl.col("arm").is_not_null())
        arm_str = "  ".join(
            f"{r['arm']} {r['n_subjects']}" for r in arms.sort("order").iter_rows(named=True)
        )
        print(
            f"  {platform}  input {inp:,}  -{tech} technical  -{ident} identity"
            f"  -{long} longitudinal  ({moved} reassigned)"
            f"  -> {final['n_samples'].item():,} samples / {final['n_subjects'].item()} subj"
            f" ({arm_str})"
        )
    return QC_BREAKDOWN_PATH


def main() -> int:
    """Write the exclusion breakdown parquet."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    write_breakdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
