"""Pinned HPA v25.1 and Genes & Health Seer annotations for Tables S1/S2.

HPA bulk data: https://v25.proteinatlas.org/download/proteinatlas.tsv.zip (the release's own
host, so the pinned checksum still matches after HPA's next release);
Human Protein Atlas v25.1, data available from v25.proteinatlas.org,
CC BY 4.0 (https://www.proteinatlas.org/about/licence).
Pietzner et al., Nat Genet (2026), doi:10.1038/s41588-026-02697-6, Supplementary Tables ST5 (Olink Explore HT
vs Seer Proteograph XT) and ST6 (SomaScan 11K vs Seer) in Genes & Health (about 1,400 participants); the workbook is
the pinned `genes_health` file of `analysis.external.pqtl_catalogs`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import openpyxl
import polars as pl

from analysis.external.hgnc_aliases import HGNC_PATH
from analysis.external.pinned import fetch_pinned

ROOT = Path(__file__).resolve().parents[2]
HPA_URL = "https://v25.proteinatlas.org/download/proteinatlas.tsv.zip"
HPA_SHA256 = "ad401f8519ecdee5b67b0ad3bf28175d402d1ea0ad6c542b90fcab2bee39b470"
HPA_PATH = ROOT / "data/external/hpa/proteinatlas.tsv.zip"
# HPA source column -> table column. The tissue column keeps the names of HPA's enriched or
# enhanced tissues and drops their nTPM values (`tissue_names`).
HPA_FIELDS = {
    "Secretome location": "HPA_Secretome_Location",
    "RNA tissue specific nTPM": "HPA_Enriched_Tissues",
}
GH_SEER_COLUMN = "GH_Seer_MS_Rho"
GH_SEER_MIN_N = 100
# Platform -> (Genes & Health supplement sheet, header of its assay ID column).
GH_SEER_SHEETS = {"olink": ("ST5", "Olink ID"), "soma": ("ST6", "SomaLogic ID")}


def tissue_names(expr: pl.Expr) -> pl.Expr:
    """``"blood vessel: 1281.4;pancreas: 901.3"`` -> ``"blood vessel, pancreas"`` (null stays null)."""
    return expr.str.split(";").list.eval(pl.element().str.replace(r":\s*[^:]*$", "").str.strip_chars()).list.join(", ")


def read_hpa(path: Path = HPA_PATH) -> pl.DataFrame:
    """Read selected bulk HPA columns after verifying the zip digest."""
    with zipfile.ZipFile(fetch_pinned(HPA_URL, path, HPA_SHA256)) as archive:
        raw = archive.read("proteinatlas.tsv")
    return pl.read_csv(io.BytesIO(raw), separator="\t", infer_schema_length=0, columns=["Ensembl", *HPA_FIELDS])


def read_hgnc(path: Path = HGNC_PATH) -> pl.DataFrame:
    """Read the pinned HGNC crosswalk through its existing fetch contract."""
    from analysis.external.hgnc_aliases import HGNC_SHA256, HGNC_URL

    return pl.read_csv(
        fetch_pinned(HGNC_URL, path, HGNC_SHA256),
        separator="\t", infer_schema_length=0,
        columns=["hgnc_id", "ensembl_gene_id"],
    )


def gh_seer_assay_id(platform: str, raw: str) -> str:
    """Genes & Health affinity ID -> Explore HT OlinkID or ``xxxxx-y`` SeqId.

    ``"A1BG_45433"`` -> ``"OID45433"``; ``"seq.10010.10"`` -> ``"10010-10"``.

    Raises:
        ValueError: For an ID that does not have the platform's form.
    """
    if platform == "olink":
        digits = raw.rsplit("_", 1)[-1]
        if "_" not in raw or not digits.isdigit():
            raise ValueError(f"Olink ID {raw!r} is not <assay>_<digits>")
        return f"OID{digits}"
    parts = raw.split(".")
    if len(parts) != 3 or parts[0] != "seq" or not (parts[1].isdigit() and parts[2].isdigit()):
        raise ValueError(f"SomaLogic ID {raw!r} is not seq.<digits>.<digits>")
    return f"{parts[1]}-{parts[2]}"


def gh_seer_workbook(path: Path | None = None) -> Path:
    """The pinned Genes & Health supplement (the file the pQTL catalogs read), SHA-256 verified."""
    from analysis.external.pqtl_catalogs import RAW_DIR, RAW_FILES

    raw = RAW_FILES["genes_health"]
    return fetch_pinned(raw.url, path or RAW_DIR / raw.name, raw.sha256)


def read_gh_seer(path: Path, platform: str) -> pl.DataFrame:
    """Read ST5 (Olink) or ST6 (SomaScan) with openpyxl at the Excel file boundary.

    The header is the first row containing the platform's ID column name. Returns one row per
    Seer protein group: ``assay_id``, ``accession`` (the sheet's ``UniProt unique``), ``seer_id``,
    ``n`` and ``rho`` (Spearman; the sheet's "NA" becomes null).
    """
    sheet, id_header = GH_SEER_SHEETS[platform]
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = wb[sheet].iter_rows(values_only=True)
    header = next(row for row in rows if id_header in row)
    ix = {name: header.index(name) for name in (id_header, "UniProt unique", "Seer ID", "N", "Spearman  corr.")}
    records = [
        {
            "assay_id": gh_seer_assay_id(platform, str(row[ix[id_header]])),
            "accession": row[ix["UniProt unique"]],
            "seer_id": row[ix["Seer ID"]],
            "n": row[ix["N"]],
            "rho": row[ix["Spearman  corr."]],
        }
        for row in rows
        if row[ix[id_header]] is not None
    ]
    wb.close()
    return pl.DataFrame(records, infer_schema_length=None).with_columns(
        pl.col("accession").cast(pl.String),
        pl.col("seer_id").cast(pl.String),
        pl.col("n").cast(pl.Int64),
        pl.col("rho").cast(pl.String).replace("NA", None).cast(pl.Float64),
    )


def _accession_rows(assays: pl.DataFrame) -> pl.DataFrame:
    return (
        assays.select("assay_id", pl.col("UniProt").cast(pl.String).str.split("; ").alias("accession"))
        .explode("accession")
        .filter(pl.col("accession").is_not_null() & (pl.col("accession") != ""))
    )


def annotate_hpa(assays: pl.DataFrame, hgnc: pl.DataFrame, hpa: pl.DataFrame) -> pl.DataFrame:
    """Annotate each HGNC gene via Ensembl, preserving assay/accession/gene order."""
    if hgnc["hgnc_id"].drop_nulls().n_unique() != hgnc["hgnc_id"].drop_nulls().len():
        raise ValueError("HGNC crosswalk has duplicate IDs")
    if hpa["Ensembl"].drop_nulls().n_unique() != hpa["Ensembl"].drop_nulls().len():
        raise ValueError("HPA bulk has duplicate Ensembl IDs")
    gene_rows = (
        assays.select("assay_id", pl.col("HGNC_ID").cast(pl.String).str.split("; ").alias("groups"))
        .explode("groups")
        .with_columns(pl.int_range(pl.len()).over("assay_id").alias("accession_order"))
        .with_columns(pl.col("groups").str.split("|").alias("gene_ids"))
        .explode("gene_ids")
        .with_columns(pl.int_range(pl.len()).over("assay_id", "accession_order").alias("gene_order"))
        .join(hgnc.select("hgnc_id", "ensembl_gene_id"), left_on="gene_ids", right_on="hgnc_id", how="left")
        .join(hpa.select("Ensembl", *HPA_FIELDS), left_on="ensembl_gene_id", right_on="Ensembl", how="left")
        .sort("assay_id", "accession_order", "gene_order")
    )
    for source, column in HPA_FIELDS.items():
        value = tissue_names(pl.col(source)) if column == "HPA_Enriched_Tissues" else pl.col(source)
        gene_rows = gene_rows.with_columns(value.fill_null("").alias(column))
    grouped = gene_rows.group_by("assay_id", "accession_order", maintain_order=True).agg(
        pl.col(column).str.join("|").alias(column) for column in HPA_FIELDS.values()
    ).sort("assay_id", "accession_order")
    collapsed = grouped.group_by("assay_id", maintain_order=True).agg(
        pl.col(column).str.join("; ").str.strip_chars("; | ").replace("", None).alias(column)
        for column in HPA_FIELDS.values()
    )
    return assays.select("assay_id").join(collapsed, on="assay_id", how="left", validate="1:1")


def annotate_gh_seer(assays: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    """``GH_Seer_MS_Rho`` by assay ID only (the ID identifies the reagent).

    An assay matched to several Seer protein groups keeps the group with the largest N (a null N or
    Seer number sorts last and never wins). Ties at
    that N go first to a group whose ``accession`` is one of the assay's annotated UniProt
    accessions (a tie-break, not a filter), then to the lowest numeric Seer ID (``prot.9`` before
    ``prot.10``). None of the tie-breaks looks at the correlation. The correlation is blank when
    the kept N is below ``GH_SEER_MIN_N`` or the sheet has no correlation.
    """
    annotated = _accession_rows(assays).unique().with_columns(pl.lit(0).alias("uniprot_rank"))
    matches = (
        source.join(annotated, on=["assay_id", "accession"], how="left")
        .with_columns(
            pl.col("uniprot_rank").fill_null(1),
            pl.col("seer_id").str.extract(r"(\d+)$").cast(pl.Int64).alias("seer_number"),
        )
        .sort(
            "assay_id",
            "n",
            "uniprot_rank",
            "seer_number",
            "seer_id",
            descending=[False, True, False, False, False],
            nulls_last=True,
        )
        .unique("assay_id", keep="first", maintain_order=True)
        .select(
            "assay_id",
            pl.when(pl.col("n") >= GH_SEER_MIN_N).then(pl.col("rho")).alias(GH_SEER_COLUMN),
        )
    )
    return assays.select("assay_id").join(matches, on="assay_id", how="left", validate="1:1")
