"""Olink Explore HT and SomaScan 11K assay-level QC for the SURMOUNT-5 substudy.

Per-assay/aptamer QC metrics — intra-assay CV and detection rates (overall and
per arm) — plus a flag indicating whether the assay was used for sample QC
(Used_for_Sample_QC = Status == "PASS"). Gene symbols and UniProt accessions
are sourced from the canonical ``analysis/outputs/uniprot_map.parquet`` lookup
so all supplementary tables use the same marker metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import QC_OLINK_DIR, QC_SOMA_DIR  # noqa: E402
OLINK_QC_TSV = QC_OLINK_DIR / "surmount5_olink.assay_qc_summary.tsv"
SOMA_QC_TSV = QC_SOMA_DIR / "surmount5_soma.assay_qc_summary.tsv"

from _annotations import build_marker_annotations  # noqa: E402


def _build_assay_qc(qc_path: Path, annot: pl.DataFrame, id_col: str) -> pl.DataFrame:
    qc = pl.read_csv(qc_path, separator="\t").rename(
        {
            "Detection_Rate_SEMA-2.4mg": "Detection_Rate_SEMA",
            "Detection_Rate_TZP-15mg": "Detection_Rate_TZP",
        }
    )
    panel = annot.select(
        pl.col("marker").alias(id_col),
        "Gene_Symbol",
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
    return (
        qc.join(panel, on=id_col, how="left")
        .with_columns((pl.col("Status") == "PASS").alias("Used_for_Sample_QC"))
        .select(
            id_col,
            "Gene_Symbol",
            "UniProt",
            "Intra_CV",
            "Detection_Rate_All",
            "Detection_Rate_SEMA",
            "Detection_Rate_TZP",
            "Used_for_Sample_QC",
        )
        .with_columns(pl.col(round_cols).round(4))
        .sort(id_col)
    )


def build() -> list[tuple[str, pl.DataFrame]]:
    olink_annot, soma_annot = build_marker_annotations()
    return [
        ("Olink Assay QC", _build_assay_qc(OLINK_QC_TSV, olink_annot, "OlinkID")),
        ("SomaScan Assay QC", _build_assay_qc(SOMA_QC_TSV, soma_annot, "SeqId")),
    ]
