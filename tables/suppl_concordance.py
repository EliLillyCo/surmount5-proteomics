"""Olink × SomaScan concordance at baseline.

Spearman correlations between paired Olink assays and SomaScan aptamers sharing at least one
UniProt accession of the assay annotation, computed at baseline (Week 0). Each row is a
unique OlinkID × SeqId pair. Is_Concordant is the binary call from a two-
component Gaussian mixture model (Prob_Concordant > 0.5).
"""

from __future__ import annotations

import polars as pl

from analysis.external.vendor_assay_tables import read_annotation
from analysis.output_layout import PLATFORM_CONCORDANCE


def accession_genes(platform: str) -> dict[tuple[str, str], list[str]]:
    """``(assay, accession)`` -> that accession's annotated genes, one platform (``read_annotation``)."""
    long = read_annotation().filter(pl.col("Platform") == platform)
    return {
        (assay, acc): [g for g in (genes or "").split("|") if g]
        for assay, acc, genes in long.select("AssayID", "UniProt", "Gene_Symbol").iter_rows()
    }


def _shared_genes(olink: str, seq: str, shared: str, olink_acc: dict, soma_acc: dict) -> str | None:
    """The genes of each shared accession, ``;``-joined in ``shared``'s order (the separator of
    ``Shared_UniProt``); several genes of one accession (identical proteins from paralogous loci) alphabetical
    and ``|``-joined, so each accession keeps one place, as in Tables S1 and S2.

    Raises:
        KeyError: When an (assay, accession) is not in the annotation.
        ValueError: When the two sides give a shared accession different genes.
    """
    parts = []
    for acc in shared.split(";"):
        o, s = sorted(olink_acc[(olink, acc)]), sorted(soma_acc[(seq, acc)])
        if o != s:
            raise ValueError(f"{olink} x {seq}: the platforms give {acc} different genes ({o} vs {s})")
        parts.append("|".join(o))
    return ";".join(parts) or None


def concordance_sheet(
    pairs: pl.DataFrame,
    olink_acc: dict[tuple[str, str], list[str]],
    soma_acc: dict[tuple[str, str], list[str]],
) -> pl.DataFrame:
    """The table of the stage's pairs: the shared accessions and their genes, the correlation and call.

    Args:
        pairs: ``platform_concordance.parquet``.
        olink_acc: ``(OlinkID, accession)`` -> genes (``accession_genes("olink")``).
        soma_acc: ``(SeqId, accession)`` -> genes (``accession_genes("soma")``).

    Returns:
        One row per pair, sorted by OlinkID then SeqId. ``Shared_UniProt`` lists every shared accession
        (``;``-joined, sorted); ``Shared_Gene_Symbol`` their annotated genes in the same order and layout
        (``_shared_genes``), not the vendor names.
    """
    genes = [
        _shared_genes(o, q, sh, olink_acc, soma_acc)
        for o, q, sh in pairs.select("OlinkID", "SeqId", "SharedUniProt").iter_rows()
    ]
    return (
        pairs.with_columns(pl.Series("Shared_Gene_Symbol", genes, dtype=pl.String))
        .select(
            "OlinkID",
            "SeqId",
            pl.col("SharedUniProt").alias("Shared_UniProt"),
            "Shared_Gene_Symbol",
            pl.col("SpearmanRho").alias("Spearman_Rho"),
            pl.col("SpearmanP").alias("Spearman_P"),
            pl.col("NSamples").alias("N"),
            pl.col("GMM_P_Concordant").alias("Prob_Concordant"),
            pl.col("Concordant").alias("Is_Concordant"),
        )
        .with_columns(pl.col(["Spearman_Rho", "Prob_Concordant"]).round(4))
        .sort(["OlinkID", "SeqId"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return the one-sheet cross-platform concordance table.

    Reads ``analysis/outputs/platform_concordance.parquet`` (one row per OlinkID x SeqId pair
    sharing a UniProt accession, with the baseline Spearman correlation and the GMM call) and
    the assay annotation's genes per accession (``concordance_sheet``).
    """
    pairs = pl.read_parquet(PLATFORM_CONCORDANCE)
    return [("Concordance", concordance_sheet(pairs, accession_genes("olink"), accession_genes("soma")))]
