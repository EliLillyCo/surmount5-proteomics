"""Olink Explore HT and SomaScan 11K assay-level QC for the SURMOUNT-5 substudy.

Per-assay/aptamer QC metrics — intra-assay CV and detection rates (overall and
per arm). Gene symbols, their HGNC IDs and UniProt accessions are sourced from the
canonical ``analysis/outputs/uniprot_map.parquet`` lookup so all supplementary tables use
the same marker metadata.
"""

from __future__ import annotations


import polars as pl

from analysis.assay_annotation import (
    build_annotation,
    load_catalogs,
    load_concordance,
)
from analysis.external.assay_table_annotations import (
    GH_SEER_COLUMN,
    HPA_FIELDS,
    annotate_gh_seer,
    annotate_hpa,
    gh_seer_workbook,
    read_gh_seer,
    read_hgnc,
    read_hpa,
)
from analysis.qc_contract import PLATFORM_SPECS, read_assay_summary

from tables._annotations import build_marker_annotations


# The assay-quality columns Tables S1/S2 print, from analysis.assay_annotation. The full
# annotation also carries the catalog count, the strongest P and its catalog, the match method,
# and the concordance posterior and call; the tables do not print these. Assay_Tier is printed as
# Eldjarn2023_Tier, the three tiers as Eldjarn et al. 2023 define them.
ANNOTATION_COLUMNS = [
    "Cis_pQTL_Catalogs",
    "Other_Platform_Cis_pQTL",
    "Eldjarn2023_Tier",
]
TIER_COLUMN = "Eldjarn2023_Tier"


def _build_assay_annotation(
    olink_annot: pl.DataFrame, soma_annot: pl.DataFrame
) -> pl.DataFrame:
    """Assay-quality annotation of every canonical Olink and SomaScan assay."""
    assays = pl.concat(
        [
            annot.select(
                pl.lit(platform).alias("platform"),
                pl.col("marker").alias("assay_id"),
                pl.col("UniProt").alias("uniprot"),
            )
            for platform, annot in (("olink", olink_annot), ("soma", soma_annot))
        ]
    )
    return build_annotation(assays, load_catalogs(), load_concordance()).rename({"Assay_Tier": TIER_COLUMN})


def _build_assay_qc(
    platform: str, annot: pl.DataFrame, assay_annotation: pl.DataFrame,
    hgnc: pl.DataFrame, hpa: pl.DataFrame, gh_seer: pl.DataFrame,
) -> pl.DataFrame:
    id_col = PLATFORM_SPECS[platform].protein_id
    # The per-arm detection rates come back as Detection_Rate_TZP / _SEMA.
    # UniProt accessions come from the canonical annotation.
    qc = read_assay_summary(platform).drop("UniProt", strict=False)
    panel = annot.select(
        pl.col("marker").alias(id_col),
        "Gene_Symbol",
        "HGNC_ID",
        "UniProt",
    )
    missing = qc.join(panel.select(id_col), on=id_col, how="anti")
    if missing.height:
        examples = ", ".join(
            map(str, missing.get_column(id_col).head(5).to_list())
        )
        raise ValueError(
            f"Missing canonical annotations for {id_col}: {examples}"
        )
    round_cols = [
        "Intra_CV",
        "Detection_Rate_All",
        "Detection_Rate_SEMA",
        "Detection_Rate_TZP",
    ]
    quality = assay_annotation.filter(pl.col("platform") == platform).select(
        pl.col("assay_id").alias(id_col), *ANNOTATION_COLUMNS
    )
    n_qc = qc.height
    external_assays = panel.select(pl.col(id_col).alias("assay_id"), "HGNC_ID", "UniProt")
    hpa_annotations = annotate_hpa(external_assays, hgnc, hpa).rename({"assay_id": id_col})
    ms_annotations = annotate_gh_seer(external_assays, gh_seer).rename({"assay_id": id_col})
    out = (
        qc.join(panel, on=id_col, how="left")
        .join(quality, on=id_col, how="left", validate="1:1")
        .join(hpa_annotations, on=id_col, how="left", validate="1:1")
        .join(ms_annotations, on=id_col, how="left", validate="1:1")
        .select(
            id_col,
            "Gene_Symbol",
            "HGNC_ID",
            "UniProt",
            "Intra_CV",
            "Detection_Rate_All",
            "Detection_Rate_SEMA",
            "Detection_Rate_TZP",
            *ANNOTATION_COLUMNS,
            *HPA_FIELDS.values(),
            GH_SEER_COLUMN,
        )
        .with_columns(pl.col(round_cols).round(4))
        .sort(id_col)
    )
    if out.height != n_qc or out[TIER_COLUMN].null_count():
        raise ValueError(f"{platform}: assay annotation did not cover every QC assay")
    return out


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the Olink and SomaScan assay QC tables.

    Reads each platform's QC assay summary, the canonical marker annotations
    (``analysis/outputs/uniprot_map.parquet``) and the assay-quality annotation (cis-pQTL
    catalogs and cross-platform concordance).

    Raises:
        ValueError: If a QC assay has no canonical annotation or no assay tier.
    """
    olink_annot, soma_annot = build_marker_annotations(hgnc_ids=True)
    assay_annotation = _build_assay_annotation(olink_annot, soma_annot)
    hgnc, hpa, workbook = read_hgnc(), read_hpa(), gh_seer_workbook()
    return [
        ("Olink Assay QC", _build_assay_qc("olink", olink_annot, assay_annotation, hgnc, hpa, read_gh_seer(workbook, "olink"))),
        ("SomaScan Assay QC", _build_assay_qc("soma", soma_annot, assay_annotation, hgnc, hpa, read_gh_seer(workbook, "soma"))),
    ]
