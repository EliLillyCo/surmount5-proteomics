"""Ensembl gene <-> UniProt cross-references, from a pinned Ensembl release (Fig 3b symbol aliases).

    pixi run python -m analysis.external.ensembl_uniprot

The disease-signature reversal of Fig 3b matches an HPA gene symbol to an assay gene by symbol
first, then, for a symbol we do not measure, through its Ensembl gene ID (``ENSG``). The IDs come from the gene -> transcript -> protein -> UniProt
cross-references in Ensembl release 116 (GRCh38),
``Homo_sapiens.GRCh38.116.uniprot.tsv.gz``. It is fetched on first use, SHA-256 verified and cached under ``data/external/ensembl/``
(gitignored). Ensembl data have no restrictions on use (https://www.ensembl.org/info/about/legal/).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.external.pinned import check_sha256, fetch_pinned

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 116
URL = (
    f"https://ftp.ensembl.org/pub/release-{RELEASE}/tsv/homo_sapiens/"
    f"Homo_sapiens.GRCh38.{RELEASE}.uniprot.tsv.gz"
)
SHA256 = "f1e26db23b0771f7a5778067af20683de63d863b1c0834aa564d5d036ec85864"
CACHE_PATH = ROOT / "data" / "external" / "ensembl" / f"Homo_sapiens.GRCh38.{RELEASE}.uniprot.tsv.gz"
# Entry-level UniProt cross-references; isoform rows (P01308-1) carry the db name Uniprot_isoform.
UNIPROT_DBS = ("Uniprot/SWISSPROT", "Uniprot/SPTREMBL")


def read_uniprot_xref(path: Path = CACHE_PATH) -> pl.DataFrame:
    """Unique ``ENSG ID, uniprot`` pairs of the pinned release, fetched first when absent.

    Raises:
        RuntimeError: When the file does not match ``SHA256``.
    """
    if path.exists():
        check_sha256(path, SHA256)
    else:
        fetch_pinned(URL, path, SHA256)
    return (
        pl.read_csv(path, separator="\t", infer_schema_length=0)
        .filter(pl.col("db_name").is_in(UNIPROT_DBS))
        .select(pl.col("gene_stable_id").alias("ENSG ID"), pl.col("xref").alias("uniprot"))
        .unique(maintain_order=True)
    )


if __name__ == "__main__":
    print(read_uniprot_xref().shape)
