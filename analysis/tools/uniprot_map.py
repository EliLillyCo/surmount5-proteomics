"""Cross-platform UniProt mapping for SURMOUNT-5 Olink + SomaScan.

Builds and caches canonical UniProt-keyed mapping tables that list ALL
Olink assays (OlinkID) and SomaScan aptamers (SeqId) targeting each
UniProt accession. Used by all downstream vignettes that need to ask
"is this protein measured on both platforms?" and "what assays cover
this protein?".

Design choices
--------------
- TWO mapping forms are saved:
  * ``uniprot_map_long.parquet``: one row per (uniprot, platform,
    assay_id, qc_status). Long-form, used for any-vs-any cross-platform
    concordance logic so that ALL assays/aptamers per UniProt
    are preserved.
  * ``uniprot_map_wide.parquet``: one row per UniProt accession with
    list-typed columns of all OlinkIDs / SeqIds and aptamer/assay
    counts plus convenience ``primary_olinkid`` / ``primary_seqid``
    columns (highest-quality QC, tie-broken by lexical order). Primary
    columns are *for convenience only* - downstream concordance code
    must NOT rely on them.
- ``uniprot_map.parquet`` mirrors the wide form and is the primary map
  used by the figures and tables (e.g. ``figures/_internal/`` lookups,
  ``tables/_annotations.py``).
- Olink rows sometimes carry multi-UniProt strings (semicolon-separated)
  for proteins with multiple chains / isoforms; we explode those.
- Source-of-truth mapping tables: mounted QC assay-summary TSVs resolved through
  `paths.py` (long-form NPX/RFU parquet files do not carry UniProt directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from paths import QC_OLINK_DIR, QC_SOMA_DIR, first_existing

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _TOOLS_DIR.parent
_PROJECT_ROOT = _ANALYSIS_DIR.parent

OLINK_QC_TSV = first_existing(
    QC_OLINK_DIR / "surmount5_olink.assay_qc_summary.tsv",
    QC_OLINK_DIR / "surmount5_olink.assay_qc_summary.tsv",
)
SOMA_QC_TSV = first_existing(
    QC_SOMA_DIR / "surmount5_soma.assay_qc_summary.tsv",
    QC_SOMA_DIR / "surmount5_soma.assay_qc_summary.tsv",
)

# Canonical panel references — single source of truth for UniProt → gene_symbol.
# Per-UniProt gene names (Gene_Symbol|UniProt_ID positionally aligned) are not
# carried by the project QC TSVs, which join multi-mapped UniProts into one
# string. Panel refs preserve the per-UniProt mapping.
# Download instructions: see data/README.md § "Vendor panel manifests".
_PANEL_REF_DIR = _ANALYSIS_DIR / "reference"
OLINK_PANEL_REF = _PANEL_REF_DIR / "olink_explore_ht.tsv"
SOMA_PANEL_REF = _PANEL_REF_DIR / "somascan_11k.tsv"

OUTPUT_DIR = _ANALYSIS_DIR / "outputs"
OUTPUT_PATH_WIDE = OUTPUT_DIR / "uniprot_map_wide.parquet"
OUTPUT_PATH_LONG = OUTPUT_DIR / "uniprot_map_long.parquet"
# Primary map used by the figures and tables; mirrors the wide form.
OUTPUT_PATH = OUTPUT_DIR / "uniprot_map.parquet"

# QC ranking used to select a primary assay/aptamer (convenience only).
_QC_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _load_panel_gene_lookup() -> pl.DataFrame:
    """Build canonical UniProt → gene_symbol from panel reference TSVs.

    Both panels carry pipe-aligned ``Gene_Symbol|UniProt_ID`` for multi-mapped
    assays/aptamers. When list lengths match we zip-explode positionally; when
    they don't (e.g. histone duplicates mapped to a single UniProt), we explode
    UniProts only and broadcast the row's first gene symbol.
    """

    def _pairs(path: Path) -> pl.DataFrame:
        df = pl.read_csv(path, separator="\t").with_columns(
            pl.col("Gene_Symbol").str.split("|").alias("_genes"),
            pl.col("UniProt_ID").str.split("|").alias("_uniprots"),
        )
        matched = (
            df.filter(pl.col("_genes").list.len() == pl.col("_uniprots").list.len())
            .explode(["_genes", "_uniprots"])
            .select(
                pl.col("_uniprots").alias("uniprot"),
                pl.col("_genes").alias("gene_symbol"),
            )
        )
        mismatched = (
            df.filter(pl.col("_genes").list.len() != pl.col("_uniprots").list.len())
            .with_columns(pl.col("_genes").list.first().alias("_first_gene"))
            .explode("_uniprots")
            .select(
                pl.col("_uniprots").alias("uniprot"),
                pl.col("_first_gene").alias("gene_symbol"),
            )
        )
        return (
            pl.concat([matched, mismatched])
            .filter(pl.col("uniprot").is_not_null() & (pl.col("uniprot") != ""))
            .unique(subset=["uniprot"])
        )

    return pl.concat([_pairs(OLINK_PANEL_REF), _pairs(SOMA_PANEL_REF)]).unique(
        subset=["uniprot"], keep="first"
    )


def _explode_uniprot(df: pl.DataFrame, col: str = "UniProt") -> pl.DataFrame:
    """Explode semicolon/comma-joined UniProt strings into one-row-per-accession."""
    return (
        df.with_columns(pl.col(col).str.replace_all(r"[,\s]+", ";").str.split(";"))
        .explode(col)
        .with_columns(pl.col(col).str.strip_chars().alias(col))
        .filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
        .rename({col: "uniprot"})
    )


def _build_olink_long() -> pl.DataFrame:
    """Olink assay metadata, long form keyed by ``uniprot`` (one row per assay)."""
    df = pl.read_csv(OLINK_QC_TSV, separator="\t")
    return _explode_uniprot(
        df.select(["OlinkID", "Assay", "UniProt", "Status"]),
    )


def _build_soma_long() -> pl.DataFrame:
    """SomaScan aptamer metadata, long form keyed by ``uniprot`` (one row per aptamer)."""
    df = pl.read_csv(SOMA_QC_TSV, separator="\t")
    return _explode_uniprot(
        df.select(["SeqId", "Target", "EntrezGeneSymbol", "UniProt", "Status"]),
    )


def _rank_and_pick_primary(
    df: pl.DataFrame,
    id_col: str,
    qc_col: str = "Status",
) -> dict[str, str]:
    """Map ``UniProt -> primary_id`` choosing the lowest-rank QC status.

    NOTE: This is for convenience columns only. Downstream concordance
    code MUST iterate ``uniprot_map_long`` for any-vs-any logic.
    """
    ranked = (
        df.with_columns(
            pl.col(qc_col).replace_strict(_QC_RANK, default=3).alias("_qc_rank"),
        )
        .sort(["uniprot", "_qc_rank", id_col])
        .group_by("uniprot")
        .agg(pl.col(id_col).first())
    )
    return dict(zip(ranked["uniprot"].to_list(), ranked[id_col].to_list()))


def _build_long_combined(
    olink_long: pl.DataFrame, soma_long: pl.DataFrame
) -> pl.DataFrame:
    """Combine Olink and Soma long-form tables into a single long mapping.

    Schema: ``uniprot, platform, assay_id, assay_label, qc_status, gene_symbol``
    where ``assay_id`` = OlinkID for Olink, SeqId for SomaScan.
    One row per (uniprot, platform, assay_id).
    """
    olink_part = olink_long.select(
        pl.col("uniprot"),
        pl.lit("olink").alias("platform"),
        pl.col("OlinkID").alias("assay_id"),
        pl.col("Assay").alias("assay_label"),
        pl.col("Status").alias("qc_status"),
        pl.col("Assay").alias("gene_symbol"),
    )
    soma_part = soma_long.select(
        pl.col("uniprot"),
        pl.lit("soma").alias("platform"),
        pl.col("SeqId").alias("assay_id"),
        pl.col("Target").alias("assay_label"),
        pl.col("Status").alias("qc_status"),
        pl.col("EntrezGeneSymbol").alias("gene_symbol"),
    )
    return pl.concat([olink_part, soma_part], how="vertical_relaxed").sort(
        ["uniprot", "platform", "assay_id"]
    )


def build_uniprot_map() -> pl.DataFrame:
    """Build the canonical UniProt -> {Olink, SomaScan} mappings.

    Saves both long and wide forms to ``analysis/outputs/`` and returns
    the WIDE DataFrame (one row per UniProt accession with list-valued
    ``olink_*`` / ``soma_*`` columns plus single-valued primary-id
    convenience columns and platform-availability booleans).
    """
    olink_long = _build_olink_long()
    soma_long = _build_soma_long()

    olink_primary = _rank_and_pick_primary(olink_long, "OlinkID")
    soma_primary = _rank_and_pick_primary(soma_long, "SeqId")

    # ---- Long form: one row per (uniprot, platform, assay_id) ----
    long_combined = _build_long_combined(olink_long, soma_long)

    # ---- Wide form: one row per uniprot with list columns + counts ----
    olink_grouped = (
        olink_long.group_by("uniprot")
        .agg(
            pl.col("OlinkID").alias("olink_olinkids"),
            pl.col("Assay").alias("olink_assays"),
            pl.col("Status").alias("olink_qc_statuses"),
            pl.col("OlinkID").len().alias("n_olink_assays"),
        )
        .with_columns(
            pl.col("uniprot")
            .map_elements(
                lambda u: olink_primary.get(u),
                return_dtype=pl.String,
            )
            .alias("olink_primary_olinkid"),
        )
    )

    soma_grouped = (
        soma_long.group_by("uniprot")
        .agg(
            pl.col("SeqId").alias("soma_seqids"),
            pl.col("Target").alias("soma_targets"),
            pl.col("EntrezGeneSymbol").alias("soma_entrez_symbols"),
            pl.col("Status").alias("soma_qc_statuses"),
            pl.col("SeqId").len().alias("n_soma_seqids"),
        )
        .with_columns(
            pl.col("uniprot")
            .map_elements(
                lambda u: soma_primary.get(u),
                return_dtype=pl.String,
            )
            .alias("soma_primary_seqid"),
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

    out_wide = combined.select(
        "uniprot",
        "gene_symbol",
        "olink_olinkids",
        "olink_assays",
        "olink_qc_statuses",
        "n_olink_assays",
        "olink_primary_olinkid",
        "soma_seqids",
        "soma_targets",
        "soma_entrez_symbols",
        "soma_qc_statuses",
        "n_soma_seqids",
        "soma_primary_seqid",
        "present_olink",
        "present_soma",
        "present_both",
    ).sort("uniprot")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_wide.write_parquet(OUTPUT_PATH_WIDE)
    long_combined.write_parquet(OUTPUT_PATH_LONG)
    # Primary map used by figures/tables (mirrors wide).
    out_wide.write_parquet(OUTPUT_PATH)
    return out_wide


def load_uniprot_map() -> pl.DataFrame:
    """Load the cached canonical UniProt map (WIDE form); build if missing."""
    if not OUTPUT_PATH_WIDE.exists():
        return build_uniprot_map()
    return pl.read_parquet(OUTPUT_PATH_WIDE)


def load_uniprot_map_long() -> pl.DataFrame:
    """Load the long-form UniProt map (one row per assay/aptamer); build if missing.

    Schema columns: ``uniprot, platform, assay_id, assay_label,
    qc_status, gene_symbol``. ``platform`` is one of ``"olink"`` /
    ``"soma"``; ``assay_id`` is OlinkID for Olink, SeqId for SomaScan.
    Use this for any cross-platform analysis that needs to consider
    ALL assays/aptamers per UniProt (any-vs-any concordance).
    """
    if not OUTPUT_PATH_LONG.exists():
        build_uniprot_map()
    return pl.read_parquet(OUTPUT_PATH_LONG)


def match_proteins(genes_or_uniprots: Iterable[str]) -> pl.DataFrame:
    """Look up rows by gene symbol or UniProt accession.

    Matches case-insensitively against ``gene_symbol``, exact match on
    ``uniprot``, and case-insensitive substring on Olink ``Assay`` /
    Soma ``EntrezGeneSymbol`` lists. Each input string yields zero or
    more rows; returns the union with a ``_query`` column noting the
    input that matched. Rows include ALL assay/aptamer lists (and
    counts) per UniProt.
    """
    queries = list(genes_or_uniprots)
    if not queries:
        return (
            load_uniprot_map()
            .head(0)
            .with_columns(pl.lit(None, dtype=pl.String).alias("_query"))
        )

    df = load_uniprot_map()
    rows: list[pl.DataFrame] = []
    for q in queries:
        q_str = str(q)
        q_upper = q_str.upper()
        match = df.filter(
            (pl.col("uniprot") == q_str)
            | (pl.col("gene_symbol").str.to_uppercase() == q_upper)
            | pl.col("olink_assays")
            .list.eval(pl.element().str.to_uppercase() == q_upper)
            .list.any()
            | pl.col("soma_entrez_symbols")
            .list.eval(pl.element().str.to_uppercase() == q_upper)
            .list.any()
        ).with_columns(pl.lit(q_str).alias("_query"))
        rows.append(match)
    return pl.concat(rows, how="vertical_relaxed")


__all__ = [
    "build_uniprot_map",
    "load_uniprot_map",
    "load_uniprot_map_long",
    "match_proteins",
    "OUTPUT_PATH",
    "OUTPUT_PATH_WIDE",
    "OUTPUT_PATH_LONG",
]
