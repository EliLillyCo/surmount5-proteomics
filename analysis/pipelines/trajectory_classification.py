"""Trajectory classification of within-arm change from baseline, both arms, both platforms.

Thin CLI over ``analysis.trajectory``: reads each platform's MMRM ``rawOutput`` (within-arm
LSmean changes at weeks 24 and 72 and the week 72 - week 24 within-arm contrast) and writes
``analysis/outputs/trajectory_{olink,soma}.parquet``.
"""

from __future__ import annotations

import polars as pl

from analysis.design import ARMS, COVAR
from analysis.design import EARLY_WEEK as EARLY
from analysis.design import LATE_WEEK as LATE
from analysis.mmrm_outputs import chg_diff_csv, chg_lsmeans_csv, pcbl_csv
from analysis.output_layout import ANALYSIS_OUTPUTS, trajectory_path
from analysis.qc_contract import PLATFORMS
from analysis.trajectory import classify_platform


def load_platform_inputs(platform: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """MMRM lsmeans, contrasts and the assay annotation of one platform.

    Returns:
        ``(lsmeans, diff, annotation)`` with annotation ``marker, Assay`` read from the
        late-visit TZP PCBL table, which carries every fitted marker.
    """
    lsmeans = pl.read_csv(chg_lsmeans_csv(platform), schema_overrides={"time": pl.Int64})
    diff = pl.read_csv(
        chg_diff_csv(platform),
        schema_overrides={"time": pl.Int64, "_time": pl.Int64},
    )
    annotation = pl.read_csv(pcbl_csv(platform, ARMS["TZP"], LATE)).select("marker", "Assay")
    return lsmeans, diff, annotation


def main() -> int:
    """Classify both platforms and write the trajectory parquets."""
    ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    for platform in PLATFORMS:
        lsmeans, diff, annotation = load_platform_inputs(platform)
        df = classify_platform(lsmeans, diff, annotation, ARMS, EARLY, LATE)

        print(f"\n{'=' * 60}\n  {platform.upper()} - trajectory classification ({COVAR})\n{'=' * 60}")
        print(f"  Total proteins: {df.height}")
        for arm in ARMS:
            counts = df.group_by(f"trajectory_{arm}").len().sort("len", descending=True)
            print(f"\n  {arm} trajectory counts:")
            for label, n in counts.iter_rows():
                print(f"    {label:20s}  n={n}")

        out_path = trajectory_path(platform)
        df.write_parquet(out_path)
        print(f"\n  Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
