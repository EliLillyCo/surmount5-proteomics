"""Cross-platform UniProt mapping for SURMOUNT-5 Olink + SomaScan.

Builds and caches canonical UniProt-keyed mapping tables that list ALL
Olink assays (OlinkID) and SomaScan aptamers (SeqId) targeting each
UniProt accession. The figures and tables read them to ask
"is this protein measured on both platforms?" and "what assays cover
this protein?".

Design choices
--------------
- ``uniprot_map.parquet`` has one row per UniProt accession with list-typed columns
  of all OlinkIDs / SeqIds and aptamer/assay counts. It is the map the figures and tables
  use (e.g. ``figures/_internal/`` lookups, ``tables/_annotations.py``).
- The analyzed assays and their labels (``Assay``, ``Target``, ``EntrezGeneSymbol``) come
  from the mounted QC assay summaries (``paths.py``); their accessions and genes come from the assay
  annotation (``analysis/external/vendor_assay_tables.py``: the vendor lists with the UniProt update
  layer), one row per (assay, accession), so a multi-target assay is listed under every accession.
  Every analyzed assay must be in the annotation, or the build raises.
- ``gene_symbol`` is the accession's annotated gene, the first non-null one in annotation order with
  Olink before SomaScan; identical proteins from paralogous loci carry every gene, pipe-joined
  (``CGB3|CGB5|CGB8``). ``hgnc_id`` is the same row's ``HGNC_ID``, aligned with it.
"""

from __future__ import annotations

import polars as pl

from analysis.external.vendor_assay_tables import ANNOTATION_PATH, read_annotation
from analysis.output_layout import UNIPROT_MAP
from analysis.qc_contract import PLATFORMS, assay_summary_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_PATH = UNIPROT_MAP
_PLATFORM_IDS = {"olink": "OlinkID", "soma": "SeqId"}


def _annotation_targets(platform: str) -> pl.DataFrame:
    """``<assay id>, uniprot, gene_symbol, hgnc_id`` of one platform's annotation, one row per (assay, accession)."""
    return (
        read_annotation(ANNOTATION_PATH)
        .filter(pl.col("Platform") == platform)
        .select(
            pl.col("AssayID").alias(_PLATFORM_IDS[platform]),
            pl.col("UniProt").alias("uniprot"),
            pl.col("Gene_Symbol").alias("gene_symbol"),
            pl.col("HGNC_ID").alias("hgnc_id"),
        )
    )


def _load_panel_gene_lookup() -> pl.DataFrame:
    """``uniprot, gene_symbol, hgnc_id``: each accession's first non-null annotated gene, Olink before SomaScan."""
    targets = pl.concat(
        [_annotation_targets(p).select("uniprot", "gene_symbol", "hgnc_id") for p in PLATFORMS]
    )
    return (
        targets.with_columns(pl.col("gene_symbol").is_null().alias("_blank"))
        .with_row_index("_order")
        .sort("_blank", "_order")
        .unique(subset=["uniprot"], keep="first", maintain_order=True)
        .sort("_order")
        .select("uniprot", "gene_symbol", "hgnc_id")
    )


def _with_targets(qc: pl.DataFrame, platform: str) -> pl.DataFrame:
    """The QC assay rows joined to their annotated accessions (one row per assay and accession).

    Raises:
        ValueError: When an analyzed assay is not in the annotation.
    """
    id_col = _PLATFORM_IDS[platform]
    targets = _annotation_targets(platform).select(id_col, "uniprot")
    missing = qc.join(targets, on=id_col, how="anti")
    if missing.height:
        raise ValueError(
            f"{missing.height} analyzed {platform} assays are not in the assay annotation "
            f"(stage vendor_assay_tables): {missing[id_col].head(10).to_list()}"
        )
    return qc.join(targets, on=id_col, how="inner", maintain_order="left").filter(pl.col("uniprot").is_not_null())


def _build_olink_long() -> pl.DataFrame:
    """Olink assay metadata, long form keyed by ``uniprot`` (one row per assay and accession)."""
    df = pl.read_csv(assay_summary_path("olink"), separator="\t")
    return _with_targets(df.select(["OlinkID", "Assay"]), "olink")


def _build_soma_long() -> pl.DataFrame:
    """SomaScan aptamer metadata, long form keyed by ``uniprot`` (one row per aptamer and accession)."""
    df = pl.read_csv(assay_summary_path("soma"), separator="\t")
    return _with_targets(df.select(["SeqId", "Target", "EntrezGeneSymbol"]), "soma")


def build_uniprot_map() -> pl.DataFrame:
    """Build the canonical UniProt -> {Olink, SomaScan} map.

    Saves ``analysis/outputs/uniprot_map.parquet`` and returns it (one row per
    UniProt accession with list-valued ``olink_*`` / ``soma_*`` columns, assay counts
    and platform-availability booleans).
    """
    olink_long = _build_olink_long()
    soma_long = _build_soma_long()

    # One row per uniprot with list columns + counts.
    olink_grouped = (
        olink_long.group_by("uniprot")
        .agg(
            pl.col("OlinkID").alias("olink_olinkids"),
            pl.col("Assay").alias("olink_assays"),
            pl.col("OlinkID").len().alias("n_olink_assays"),
        )
    )

    soma_grouped = (
        soma_long.group_by("uniprot")
        .agg(
            pl.col("SeqId").alias("soma_seqids"),
            pl.col("Target").alias("soma_targets"),
            pl.col("EntrezGeneSymbol").alias("soma_entrez_symbols"),
            pl.col("SeqId").len().alias("n_soma_seqids"),
        )
    )

    combined = olink_grouped.join(soma_grouped, on="uniprot", how="full", coalesce=True)

    combined = (
        combined.join(_load_panel_gene_lookup(), on="uniprot", how="left")
        .with_columns(
            pl.col("olink_olinkids").is_not_null().alias("present_olink"),
            pl.col("soma_seqids").is_not_null().alias("present_soma"),
            pl.col("n_olink_assays").fill_null(0).alias("n_olink_assays"),
            pl.col("n_soma_seqids").fill_null(0).alias("n_soma_seqids"),
        )
        .with_columns(
            (pl.col("present_olink") & pl.col("present_soma")).alias("present_both"),
        )
    )

    out = combined.select(
        "uniprot",
        "gene_symbol",
        "olink_olinkids",
        "olink_assays",
        "n_olink_assays",
        "soma_seqids",
        "soma_targets",
        "soma_entrez_symbols",
        "n_soma_seqids",
        "present_olink",
        "present_soma",
        "present_both",
        "hgnc_id",
    ).sort("uniprot")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUT_PATH)
    return out


__all__ = [
    "build_uniprot_map",
    "OUTPUT_PATH",
]
