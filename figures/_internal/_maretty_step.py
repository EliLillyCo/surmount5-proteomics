from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
MARETTY_STEP_PATH = ROOT / "analysis" / "outputs" / "maretty_step_data.parquet"
_BUILD_CMD = "uv run python analysis/scripts/maretty_step_data.py"


@lru_cache(maxsize=1)
def load_maretty_step_frame() -> pl.DataFrame:
    """Load the canonical STEP/Maretty parquet prepared under analysis/outputs."""
    if not MARETTY_STEP_PATH.exists():
        raise FileNotFoundError(
            "Missing Maretty STEP data. Run: "
            f"{_BUILD_CMD}"
        )
    return pl.read_parquet(MARETTY_STEP_PATH)


def load_step_primary(trial: str) -> pd.DataFrame:
    """Primary STEP 1/2 t-statistics as a pandas frame for cross-study scatters."""
    return (
        load_maretty_step_frame()
        .filter((pl.col("trial") == trial) & (pl.col("model") == "primary"))
        .select(
            "SeqId",
            pl.col("test_statistic").alias("z_step"),
        )
        .drop_nulls()
        .to_pandas()
    )


def load_step_effects(gene: str) -> dict[tuple[str, str], dict | None]:
    """Best STEP effect row per (trial, model) for a requested gene.

    Ties on ``|test_statistic|`` fall back to the original within-sheet row order.
    """
    frame = load_maretty_step_frame()
    out: dict[tuple[str, str], dict | None] = {}
    for trial in ("step1", "step2"):
        for model in ("primary", "blocked"):
            df = frame.filter(
                (pl.col("trial") == trial)
                & (pl.col("model") == model)
                & (pl.col("gene_symbol") == gene)
            )
            if df.is_empty():
                out[(trial, model)] = None
                continue
            row = (
                df.with_row_index("_row_order")
                .with_columns(pl.col("test_statistic").abs().alias("_abs_test_statistic"))
                .sort(
                    by=["_abs_test_statistic", "_row_order"],
                    descending=[True, False],
                )
                .drop(["_row_order", "_abs_test_statistic"])
                .head(1)
                .to_dicts()[0]
            )
            out[(trial, model)] = row
    return out
