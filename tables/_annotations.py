"""Shared marker annotation helpers for supplementary tables.

Builds authoritative per-marker annotations from
``analysis/outputs/uniprot_map.parquet``. Vendor labels are paired
positionally with their marker IDs via simultaneous list explosion so
multi-marker rows retain the correct Assay/Target names.
"""

from __future__ import annotations

import polars as pl

from analysis.output_layout import UNIPROT_MAP


def _load_uniprot_map() -> pl.DataFrame:
    return pl.read_parquet(
        UNIPROT_MAP,
        columns=[
            "uniprot",
            "gene_symbol",
            "hgnc_id",
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
        f"Non-parallel {marker_col}/{label_col} lists in {UNIPROT_MAP}: {examples}"
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
            "hgnc_id",
            pl.col(label_col).alias(label_name),
        )
        .drop_nulls(subset=["marker"])
    )


def _collapse_marker_annotations(rows: pl.DataFrame, label_name: str, hgnc_ids: bool = False) -> pl.DataFrame:
    """One row per marker: its accessions listed in the order of their genes (``Gene_Symbol`` sorted, ``"; "``).

    With ``hgnc_ids``, ``HGNC_ID`` lists each ``Gene_Symbol`` entry's HGNC IDs in the same order and
    the same ``"; "`` / ``|`` layout (an empty element for a gene without one).
    """
    rows = rows.sort("marker", "gene_symbol", "uniprot", nulls_last=True)
    out = rows.group_by("marker", maintain_order=True).agg(
        pl.col("uniprot").drop_nulls().unique(maintain_order=True).str.join("; ").alias("UniProt"),
        pl.col("gene_symbol").drop_nulls().unique(maintain_order=True).str.join("; ").alias("Gene_Symbol"),
        _collapse_unique(label_name, label_name),
    )
    if hgnc_ids:
        ids = (
            rows.drop_nulls("gene_symbol")
            .unique(["marker", "gene_symbol"], maintain_order=True)
            .group_by("marker", maintain_order=True)
            .agg(pl.col("hgnc_id").fill_null("").str.join("; ").alias("HGNC_ID"))
            .with_columns(pl.when(pl.col("HGNC_ID").str.replace_all(r"[;| ]", "") != "").then(pl.col("HGNC_ID")).alias("HGNC_ID"))
        )
        out = out.join(ids, on="marker", how="left", maintain_order="left")
    return out.sort("marker")


def build_marker_annotations(hgnc_ids: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (olink_annot, soma_annot) DataFrames keyed by ``marker``.

    Each row: marker, UniProt, Gene_Symbol, Assay (Olink) or Target (Soma); with ``hgnc_ids``
    also ``HGNC_ID`` (Tables S1 and S2).
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
        hgnc_ids=hgnc_ids,
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
        hgnc_ids=hgnc_ids,
    )

    return olink_annot, soma_annot


def join_annotation(annot: pl.DataFrame, rows: pl.DataFrame, table: str) -> pl.DataFrame:
    """``annot`` joined to the model ``rows`` on ``marker``.

    Args:
        annot: One platform's frame of ``build_marker_annotations``.
        rows: Model results keyed by ``marker``.
        table: Name of the table, for the error message.

    Raises:
        ValueError: When a marker of ``rows`` has no annotation row.
    """
    missing = rows.join(annot.select("marker"), on="marker", how="anti")["marker"].unique().sort().to_list()
    if missing:
        raise ValueError(f"{table}: {len(missing)} modeled assays have no marker annotation, e.g. {missing[:5]}")
    return annot.join(rows, on="marker", how="inner")

