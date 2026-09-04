"""Olink × SomaScan concordance at baseline.

Spearman correlations between paired Olink assays and SomaScan aptamers anchored
on a shared UniProt accession, computed at baseline (Week 0). Each row is a
unique OlinkID × SeqId pair. Is_Concordant is the binary call from a two-
component Gaussian mixture model (Prob_Concordant > 0.5).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"


def build() -> list[tuple[str, pl.DataFrame]]:
    df = pl.read_parquet(ANALYSIS_DIR / "outputs" / "platform_concordance.parquet")
    out = (
        df.select(
            "OlinkID",
            "SeqId",
            pl.col("UniProt").alias("Shared_UniProt"),
            pl.col("GeneSymbol").alias("Shared_Gene_Symbol"),
            pl.col("SpearmanRho").alias("Spearman_Rho"),
            pl.col("SpearmanP").alias("Spearman_P"),
            pl.col("NSamples").alias("N"),
            pl.col("GMM_P_Concordant").alias("Prob_Concordant"),
            pl.col("Concordant").alias("Is_Concordant"),
        )
        .with_columns(pl.col(["Spearman_Rho", "Prob_Concordant"]).round(4))
        .sort(["OlinkID", "SeqId"])
    )
    return [("Concordance", out)]
