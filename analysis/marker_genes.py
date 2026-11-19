"""Marker -> HGNC gene lookups from ``analysis/outputs/uniprot_map.parquet``, for figures and tables.

A marker is an OlinkID (Olink) or a SeqId (SomaScan). The map's rows are UniProt accessions,
each listing the markers of either platform that measure it, so a heterodimer SeqId can map to
two genes. ``load_marker_gene_pairs`` keeps the map's row order, Olink rows first, which is the
order the first-match lookups (``build_gene_lookup``, ``load_marker_to_gene``) resolve a
multi-gene marker by. Set-based readers (the pathway tables' significant-gene sets) do not
depend on the order.

An accession can carry several genes, pipe-joined in the map's ``gene_symbol`` (identical proteins
from paralogous loci, such as P0DN86 = ``CGB3|CGB5|CGB8``). The pair and set readers
(``load_marker_gene_pairs``, ``build_marker_genes``) split them, one gene each, so gene-set matching
sees every gene; the label readers (``build_gene_lookup``, ``load_marker_to_gene``) keep the joined
label.
"""

from __future__ import annotations

import polars as pl

from analysis.output_layout import UNIPROT_MAP
from analysis.qc_contract import PLATFORMS

UNIPROT_MAP_PATH = UNIPROT_MAP
# Platform -> (marker list column, presence flag column) of the map.
_MARKER_COLUMNS: dict[str, tuple[str, str]] = {
    "olink": ("olink_olinkids", "present_olink"),
    "soma": ("soma_seqids", "present_soma"),
}


def _platform_pairs(umap: pl.DataFrame, platform: str) -> pl.DataFrame:
    col, present = _MARKER_COLUMNS[platform]
    return (
        umap.filter(pl.col(present))
        .explode(col)
        .select(pl.col(col).alias("marker"), pl.col("gene_symbol"))
        .drop_nulls()
    )


def _marker_labels() -> pl.DataFrame:
    """Unique ``(marker, gene_symbol)`` label pairs of both platforms, in map row order, Olink first."""
    umap = pl.read_parquet(
        UNIPROT_MAP_PATH,
        columns=["gene_symbol", *(c for cols in _MARKER_COLUMNS.values() for c in cols)],
    )
    pairs = [_platform_pairs(umap, platform) for platform in PLATFORMS]
    return pl.concat(pairs, how="vertical_relaxed").unique(maintain_order=True)


def load_marker_gene_pairs() -> pl.DataFrame:
    """Unique ``(marker, gene_symbol)`` pairs of both platforms, one gene each, in map row order, Olink first.

    Returns:
        Columns ``marker`` and ``gene_symbol``; a heterodimer SomaScan SeqId, and a marker whose
        accession carries several genes, contribute one row per gene.
    """
    return (
        _marker_labels()
        .with_columns(pl.col("gene_symbol").str.split("|"))
        .explode("gene_symbol")
        .unique(maintain_order=True)
    )


def build_gene_lookup() -> dict[str, str]:
    """Marker -> one gene label, the first in map row order, Olink first (paralog genes stay joined)."""
    lookup: dict[str, str] = {}
    for marker, gene_symbol in _marker_labels().iter_rows():
        lookup.setdefault(marker, gene_symbol)
    return lookup


def build_marker_genes() -> dict[str, list[str]]:
    """Marker -> every gene symbol it measures, sorted (a SeqId for integrin aVb3 -> ITGAV, ITGB3)."""
    out: dict[str, list[str]] = {}
    for marker, gene_symbol in load_marker_gene_pairs().iter_rows():
        out.setdefault(marker, []).append(gene_symbol)
    return {marker: sorted(set(genes)) for marker, genes in out.items()}


def load_marker_to_gene(platform: str) -> dict[str, str]:
    """Marker -> one gene label for one platform, the first in map row order (paralog genes stay joined).

    Args:
        platform: ``olink`` (OlinkIDs) or ``soma`` (SeqIds).

    Raises:
        ValueError: For any other platform.
    """
    if platform not in _MARKER_COLUMNS:
        raise ValueError("platform must be 'olink' or 'soma'")
    col, present = _MARKER_COLUMNS[platform]
    umap = pl.read_parquet(UNIPROT_MAP_PATH, columns=["gene_symbol", col, present])
    lookup: dict[str, str] = {}
    for marker, gene_symbol in _platform_pairs(umap, platform).iter_rows():
        lookup.setdefault(marker, gene_symbol)
    return lookup
