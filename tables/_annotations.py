"""Shared marker annotation helpers for supplementary tables.

Builds authoritative per-marker annotations from
``analysis/outputs/uniprot_map.parquet``. Vendor labels are paired
positionally with their marker IDs via simultaneous list explosion so
multi-marker rows retain the correct Assay/Target names.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
UNIPROT_MAP_PATH = ANALYSIS_DIR / "outputs" / "uniprot_map.parquet"


def _load_uniprot_map() -> pl.DataFrame:
    return pl.read_parquet(
        UNIPROT_MAP_PATH,
        columns=[
            "uniprot",
            "gene_symbol",
            "olink_olinkids",
            "olink_assays",
            "soma_seqids",
            "soma_targets",
            "present_olink",
            "present_soma",
        ],
    )


def _collapse_unique(source: str, alias: str) -> pl.Expr:
    return pl.col(source).drop_nulls().unique().sort().str.join("; ").alias(alias)


def _validate_parallel_lists(
    umap: pl.DataFrame,
    *,
    marker_col: str,
    label_col: str,
    present_col: str,
) -> None:
    relevant = (
        umap.filter(pl.col(present_col) & pl.col(marker_col).is_not_null())
        .with_columns(
            pl.col(marker_col).list.len().alias("_marker_len"),
            pl.col(label_col).list.len().fill_null(0).alias("_label_len"),
        )
        .filter(pl.col("_marker_len") != pl.col("_label_len"))
    )
    if relevant.is_empty():
        return

    examples = ", ".join(
        f"{row['uniprot']} ({row['_marker_len']} markers vs {row['_label_len']} labels)"
        for row in relevant.select(
            ["uniprot", "_marker_len", "_label_len"]
        ).head(3).to_dicts()
    )
    raise ValueError(
        f"Non-parallel {marker_col}/{label_col} lists in {UNIPROT_MAP_PATH}: {examples}"
    )


def _platform_marker_rows(
    umap: pl.DataFrame,
    *,
    marker_col: str,
    label_col: str,
    present_col: str,
    label_name: str,
) -> pl.DataFrame:
    _validate_parallel_lists(
        umap,
        marker_col=marker_col,
        label_col=label_col,
        present_col=present_col,
    )
    return (
        umap.filter(pl.col(present_col) & pl.col(marker_col).is_not_null())
        .explode([marker_col, label_col])
        .select(
            pl.col(marker_col).alias("marker"),
            "uniprot",
            "gene_symbol",
            pl.col(label_col).alias(label_name),
        )
        .drop_nulls(subset=["marker"])
    )


def _collapse_marker_annotations(rows: pl.DataFrame, label_name: str) -> pl.DataFrame:
    return rows.group_by("marker").agg(
        _collapse_unique("uniprot", "UniProt"),
        _collapse_unique("gene_symbol", "Gene_Symbol"),
        _collapse_unique(label_name, label_name),
    ).sort("marker")


def build_marker_annotations() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (olink_annot, soma_annot) DataFrames keyed by ``marker``.

    Each row: marker, UniProt, Gene_Symbol, Assay (Olink) or Target (Soma).
    """
    umap = _load_uniprot_map()

    olink_annot = _collapse_marker_annotations(
        _platform_marker_rows(
            umap,
            marker_col="olink_olinkids",
            label_col="olink_assays",
            present_col="present_olink",
            label_name="Assay",
        ),
        label_name="Assay",
    )
    soma_annot = _collapse_marker_annotations(
        _platform_marker_rows(
            umap,
            marker_col="soma_seqids",
            label_col="soma_targets",
            present_col="present_soma",
            label_name="Target",
        ),
        label_name="Target",
    )

    return olink_annot, soma_annot


def load_marker_gene_pairs() -> pl.DataFrame:
    """Return unique ``(marker, gene_symbol)`` pairs across both platforms."""
    umap = _load_uniprot_map()
    olink = (
        umap.filter(pl.col("present_olink"))
        .explode("olink_olinkids")
        .select(
            pl.col("olink_olinkids").alias("marker"),
            "gene_symbol",
        )
        .drop_nulls(subset=["marker", "gene_symbol"])
    )
    soma = (
        umap.filter(pl.col("present_soma"))
        .explode("soma_seqids")
        .select(
            pl.col("soma_seqids").alias("marker"),
            "gene_symbol",
        )
        .drop_nulls(subset=["marker", "gene_symbol"])
    )
    return pl.concat([olink, soma], how="vertical_relaxed").unique().sort(
        ["marker", "gene_symbol"]
    )
