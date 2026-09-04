"""Marker and gene lookup helpers shared across manuscript figures."""

from __future__ import annotations

import polars as pl

from ._paths import ANALYSIS_DIR

_UNIPROT_MAP_PATH = ANALYSIS_DIR / "outputs" / "uniprot_map.parquet"


def load_marker_gene_pairs() -> pl.DataFrame:
    """All (marker, gene_symbol) pairs from uniprot_map — both platforms.

    Returns a unique DataFrame with columns [marker, gene_symbol].
    Heterodimer SomaScan SeqIds contribute one row per gene.
    """
    umap = pl.read_parquet(
        _UNIPROT_MAP_PATH,
        columns=[
            "gene_symbol",
            "olink_olinkids",
            "soma_seqids",
            "present_olink",
            "present_soma",
        ],
    )
    olink = (
        umap.filter(pl.col("present_olink"))
        .explode("olink_olinkids")
        .select(
            pl.col("olink_olinkids").alias("marker"),
            pl.col("gene_symbol"),
        )
        .drop_nulls()
    )
    soma = (
        umap.filter(pl.col("present_soma"))
        .explode("soma_seqids")
        .select(
            pl.col("soma_seqids").alias("marker"),
            pl.col("gene_symbol"),
        )
        .drop_nulls()
    )
    return pl.concat([olink, soma], how="vertical_relaxed").unique(maintain_order=True)


def build_gene_lookup() -> dict[str, str]:
    """marker -> single HGNC gene_symbol (first match per marker).

    Uses the authoritative gene_symbol column from uniprot_map.
    """
    lookup: dict[str, str] = {}
    for marker, gene_symbol in load_marker_gene_pairs().iter_rows():
        lookup.setdefault(marker, gene_symbol)
    return lookup


def build_marker_genes() -> dict[str, list[str]]:
    """marker -> ALL HGNC gene_symbols measured by that marker.

    Preserves heterodimer/cross-reactive relationships (e.g. a SomaScan
    SeqId measuring Integrin aVb3 maps to both ITGB3 and ITGAV).
    """
    pairs = load_marker_gene_pairs()
    out: dict[str, list[str]] = {}
    for row in pairs.iter_rows(named=True):
        out.setdefault(row["marker"], []).append(row["gene_symbol"])
    return {marker: sorted(set(genes)) for marker, genes in out.items()}


def load_marker_to_gene(platform: str) -> dict[str, str]:
    """marker -> gene_symbol for a single platform (olink or soma).

    Marker is OlinkID for Olink, SeqId for SomaScan.
    """
    if platform not in {"olink", "soma"}:
        raise ValueError("platform must be 'olink' or 'soma'")
    col = "olink_olinkids" if platform == "olink" else "soma_seqids"
    present_col = "present_olink" if platform == "olink" else "present_soma"
    umap = pl.read_parquet(
        _UNIPROT_MAP_PATH,
        columns=["gene_symbol", col, present_col],
    )
    exploded = (
        umap.filter(pl.col(present_col))
        .explode(col)
        .select(pl.col(col).alias("marker"), pl.col("gene_symbol"))
        .drop_nulls()
    )
    lookup: dict[str, str] = {}
    for marker, gene_symbol in exploded.iter_rows():
        lookup.setdefault(marker, gene_symbol)
    return lookup
