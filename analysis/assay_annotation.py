"""Assay-quality annotation for Supplementary Tables S1 (Olink) and S2 (SomaScan).

Each assay gets cis-pQTL support from the 13 parsed reference catalogs (five Olink, eight
SomaScan; ``data/external/pqtl/parsed``), the same roster the two QC configs select through
``analysis.qc.pqtl_reference``. The SomaScan catalogs are deCODE, ARIC, Fenland, the CKB Atlas
(``ckb_soma``), MESA, Genes & Health, AGES and INTERVAL. The annotation also records whether the
shared protein has a cis-pQTL on the other platform, the SURMOUNT-5 baseline
cross-platform concordance (two-component GMM, ``platform_concordance.parquet``) and a
three-level tier after the framework of Eldjarn et al., Nature 2023:

* Tier 1: cis-pQTL on both platforms and concordant.
* Tier 2: cis-pQTL on at least one platform, not Tier 1.
* Tier 3: neither.

The functions are pure over frames; ``load_catalogs`` and ``load_concordance`` are the
only readers. No assay is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.qc.pqtl_reference import (
    CATALOG_NAMES,
    _drop_sign_exclusions,
    fill_underflowed_nlp,
)
from analysis.output_layout import PLATFORM_CONCORDANCE
from analysis.qc_paths import PQTL_CATALOG_DIR

NLP_FLOOR: float = 7.30103
"""-log10(5e-8): the genome-wide threshold applied to every catalog row."""

_ACCESSION_SEPARATORS = r"[;_,|]"


@dataclass(frozen=True)
class CatalogSpec:
    """One reference catalog.

    Attributes:
        key: Name in ``pqtl_reference.CATALOG_NAMES``.
        label: Name shown in the tables.
        platform: ``olink`` or ``soma``.
        match: ``native`` (catalog ``platform_id`` is the assay ID) or ``uniprot``
            (Olink Explore 3072 IDs, matched to Explore HT assays by UniProt).
    """

    key: str
    label: str
    platform: str
    match: str


CATALOGS: tuple[CatalogSpec, ...] = (
    CatalogSpec("ukbppp", "UKB-PPP", "olink", "uniprot"),
    CatalogSpec("jhs", "JHS", "olink", "uniprot"),
    CatalogSpec("mesa", "MESA", "olink", "uniprot"),
    CatalogSpec("ckb", "CKB", "olink", "uniprot"),
    CatalogSpec("gh", "Genes & Health", "olink", "native"),
    CatalogSpec("decode", "deCODE", "soma", "native"),
    CatalogSpec("aric", "ARIC", "soma", "native"),
    CatalogSpec("fenland_hg38", "Fenland", "soma", "native"),
    CatalogSpec("ckb_soma", "CKB", "soma", "native"),
    CatalogSpec("mesa_soma", "MESA", "soma", "native"),
    CatalogSpec("gh_soma", "Genes & Health", "soma", "native"),
    CatalogSpec("ages", "AGES", "soma", "native"),
    CatalogSpec("interval", "INTERVAL", "soma", "native"),
)


def _tokens(col: str) -> pl.Expr:
    """UniProt accessions of a string column, split on ``; _ , |``."""
    return (
        pl.col(col)
        .cast(pl.String)
        .str.replace_all(_ACCESSION_SEPARATORS, ",")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
    )


def prepare_catalog(raw: pl.DataFrame, file_name: str) -> pl.DataFrame:
    """Applies the reference-catalog handling of ``pqtl_reference`` to one catalog.

    Drops the ``SIGN_EXCLUSIONS`` proteins of the file and recomputes -log10(p) from
    beta/se where it is null or at least 300 (``fill_underflowed_nlp``).

    Args:
        raw: A parsed catalog with ``platform_id``, ``uniprot``, ``beta``, ``se`` and
            ``neg_log10p``.
        file_name: The catalog's parquet file name.

    Returns:
        ``platform_id``, ``uniprot`` and ``nlp`` (-log10 p, null when unrecoverable).
    """
    kept = _drop_sign_exclusions(raw, file_name)
    filled = fill_underflowed_nlp(kept)
    return filled.select("platform_id", "uniprot", pl.col("neg_log10p").alias("nlp"))


def load_catalogs(directory: Path = PQTL_CATALOG_DIR) -> dict[str, pl.DataFrame]:
    """Reads and prepares the 13 parsed catalogs, by ``CatalogSpec.key``."""
    columns = ["platform_id", "uniprot", "beta", "se", "neg_log10p"]
    out = {}
    for spec in CATALOGS:
        file_name = CATALOG_NAMES[spec.key]
        raw = pl.read_parquet(directory / file_name, columns=columns)
        out[spec.key] = prepare_catalog(raw, file_name)
    return out


def load_concordance(path: Path = PLATFORM_CONCORDANCE) -> pl.DataFrame:
    """The SURMOUNT-5 baseline Olink x SomaScan pairs with their GMM call."""
    return pl.read_parquet(
        path, columns=["OlinkID", "SeqId", "GMM_P_Concordant", "Concordant"]
    )


def _native_evidence(spec: CatalogSpec, cat: pl.DataFrame, assays: pl.DataFrame) -> pl.DataFrame:
    """(assay_id, nlp) of the assays whose own ID the catalog reports."""
    ids = pl.col("platform_id")
    if spec.platform == "soma":
        ids = ids.str.replace_all("_", "-")
    own = assays.filter(pl.col("platform") == spec.platform).select("assay_id")
    return (
        cat.select(ids.alias("assay_id"), "nlp")
        .group_by("assay_id")
        .agg(pl.col("nlp").max())
        .join(own, on="assay_id", how="inner")
    )


def _uniprot_evidence(cat: pl.DataFrame, tokens: pl.DataFrame) -> pl.DataFrame:
    """(assay_id, nlp) of the assays sharing a UniProt accession with a catalog row."""
    rows = (
        cat.select(_tokens("uniprot").alias("tok"), "nlp")
        .explode("tok")
        .drop_nulls("tok")
    )
    return (
        rows.join(tokens, on="tok", how="inner")
        .group_by("assay_id")
        .agg(pl.col("nlp").max())
    )


def _concordance_by_assay(conc: pl.DataFrame) -> pl.DataFrame:
    """Per assay: best GMM probability and whether any partner pair is concordant."""
    parts = [
        conc.group_by(id_col).agg(
            pl.col("GMM_P_Concordant").max().alias("Concordance_P"),
            pl.col("Concordant").any().alias("Concordant"),
        ).select(
            pl.lit(platform).alias("platform"),
            pl.col(id_col).alias("assay_id"),
            "Concordance_P",
            "Concordant",
        )
        for platform, id_col in (("olink", "OlinkID"), ("soma", "SeqId"))
    ]
    return pl.concat(parts)


def build_annotation(
    assays: pl.DataFrame,
    catalogs: dict[str, pl.DataFrame],
    concordance: pl.DataFrame,
    specs: tuple[CatalogSpec, ...] = CATALOGS,
) -> pl.DataFrame:
    """Annotates every assay with cis-pQTL support, concordance and tier.

    Args:
        assays: ``platform`` (olink or soma), ``assay_id`` and ``uniprot`` (accessions
            separated by ``; _ , |``), one row per assay.
        catalogs: Prepared catalogs (``prepare_catalog``) by ``CatalogSpec.key``.
        concordance: ``OlinkID``, ``SeqId``, ``GMM_P_Concordant`` and ``Concordant``,
            one row per Olink x SomaScan pair.
        specs: The catalogs to use, in display order.

    Returns:
        One row per input assay, in input order: ``platform``, ``assay_id`` and the
        nine annotation columns. ``Concordance_P`` and ``Concordant`` are null for an
        assay without a partner pair, which can therefore never be Tier 1.

    Raises:
        ValueError: If an assay is repeated.
        KeyError: If a catalog of ``specs`` is missing from ``catalogs``.
    """
    if assays.select(["platform", "assay_id"]).is_duplicated().any():
        raise ValueError("duplicate (platform, assay_id) rows in assays")
    tokens = (
        assays.select("platform", "assay_id", _tokens("uniprot").alias("tok"))
        .explode("tok")
        .drop_nulls("tok")
        .unique()
    )

    evidence: list[pl.DataFrame] = []  # per catalog: platform, assay_id, idx, nlp
    protein_cis: list[pl.DataFrame] = []  # per catalog: platform, tok
    for idx, spec in enumerate(specs):
        cat = catalogs[spec.key].filter(pl.col("nlp") >= NLP_FLOOR)
        plat_tokens = tokens.filter(pl.col("platform") == spec.platform).select("assay_id", "tok")
        if spec.match == "native":
            hit = _native_evidence(spec, cat, assays)
            prot = hit.select("assay_id").join(plat_tokens, on="assay_id").select("tok")
        else:
            hit = _uniprot_evidence(cat, plat_tokens)
            prot = cat.select(_tokens("uniprot").alias("tok")).explode("tok").drop_nulls("tok")
        evidence.append(
            hit.select(
                pl.lit(spec.platform).alias("platform"),
                "assay_id",
                pl.lit(idx, dtype=pl.Int64).alias("idx"),
                "nlp",
            )
        )
        protein_cis.append(prot.select(pl.lit(spec.platform).alias("platform"), "tok"))

    names = pl.DataFrame(
        {
            "idx": list(range(len(specs))),
            "label": [s.label for s in specs],
            "match": [f"{s.label}: {s.match.replace('uniprot', 'UniProt')}" for s in specs],
        },
        schema={"idx": pl.Int64, "label": pl.String, "match": pl.String},
    )
    ev = pl.concat(evidence).join(names, on="idx").sort("platform", "assay_id", "idx")
    cis = ev.group_by("platform", "assay_id", maintain_order=True).agg(
        pl.len().cast(pl.Int32).alias("Cis_pQTL_N_Catalogs"),
        pl.col("label").str.join("; ").alias("Cis_pQTL_Catalogs"),
        pl.col("nlp").max().alias("Cis_pQTL_Max_NegLog10P"),
        pl.col("label").sort_by(["nlp", "idx"], descending=[True, False]).first().alias("Cis_pQTL_Max_Catalog"),
        pl.col("match").str.join("; ").alias("Cis_pQTL_Match"),
    )

    prot = pl.concat(protein_cis).unique()
    other = (
        tokens.join(
            prot.with_columns(
                pl.when(pl.col("platform") == "olink").then(pl.lit("soma")).otherwise(pl.lit("olink")).alias("platform")
            ),
            on=["platform", "tok"],
            how="inner",
        )
        .select("platform", "assay_id")
        .unique()
        .with_columns(pl.lit(True).alias("Other_Platform_Cis_pQTL"))
    )

    out = (
        assays.select("platform", "assay_id")
        .join(cis, on=["platform", "assay_id"], how="left", maintain_order="left")
        .join(other, on=["platform", "assay_id"], how="left", maintain_order="left")
        .join(
            _concordance_by_assay(concordance),
            on=["platform", "assay_id"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.col("Cis_pQTL_N_Catalogs").fill_null(0).cast(pl.Int32),
            pl.col("Other_Platform_Cis_pQTL").fill_null(False),
        )
    )
    own = pl.col("Cis_pQTL_N_Catalogs") >= 1
    both = own & pl.col("Other_Platform_Cis_pQTL")
    concordant = pl.col("Concordant").fill_null(False)
    return out.with_columns(
        pl.when(both & concordant).then(1).when(own | pl.col("Other_Platform_Cis_pQTL")).then(2).otherwise(3)
        .cast(pl.Int32)
        .alias("Assay_Tier")
    ).select(
        "platform",
        "assay_id",
        "Cis_pQTL_N_Catalogs",
        "Cis_pQTL_Catalogs",
        "Cis_pQTL_Max_NegLog10P",
        "Cis_pQTL_Max_Catalog",
        "Cis_pQTL_Match",
        "Other_Platform_Cis_pQTL",
        "Concordance_P",
        "Concordant",
        "Assay_Tier",
    )

