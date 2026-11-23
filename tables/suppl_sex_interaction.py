"""Sex x treatment interaction on the TZP - SEMA contrast.

Per protein and week: the TZP - SEMA difference in females and in males (descriptive),
and the interaction (female minus male) with its P-value and FDR.
Two sheets: Olink, SomaScan.
"""

from __future__ import annotations


import polars as pl

from analysis.mmrm_outputs import sexint_csv

from tables._annotations import build_marker_annotations, join_annotation

_ROUND = ["TZP−SEMA (F)", "SE (F)", "TZP−SEMA (M)", "SE (M)", "Interaction (F−M)", "Interaction SE"]


def _build_platform_table(platform: str, annot: pl.DataFrame) -> pl.DataFrame:
    id_col = "OlinkID" if platform == "olink" else "SeqId"
    label_col = "Assay" if platform == "olink" else "Target"
    rows = pl.read_csv(sexint_csv(platform)).drop("Assay", strict=False)
    return (
        join_annotation(annot, rows, "sex interaction")
        .rename({"marker": id_col})
        .select(
            id_col,
            "UniProt",
            "Gene_Symbol",
            label_col,
            pl.col("time").cast(pl.Int16).alias("Week"),
            pl.col("est_F").alias("TZP−SEMA (F)"),
            pl.col("se_F").alias("SE (F)"),
            pl.col("fdr_F_desc").alias("FDR (F, descriptive)"),
            pl.col("est_M").alias("TZP−SEMA (M)"),
            pl.col("se_M").alias("SE (M)"),
            pl.col("fdr_M_desc").alias("FDR (M, descriptive)"),
            pl.col("est_int").alias("Interaction (F−M)"),
            pl.col("se_int").alias("Interaction SE"),
            pl.col("p_int").alias("Interaction P-value"),
            pl.col("fdr_int").alias("Interaction FDR"),
            pl.col("int_direction").alias("Direction"),
        )
        .with_columns(pl.col(_ROUND).round(4))
        .sort([id_col, "Week"])
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    """Return list of (sheet_name, DataFrame) for the sex-interaction tables."""
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink", _build_platform_table("olink", olink_annot)),
        ("SomaScan", _build_platform_table("soma", soma_annot)),
    ]
