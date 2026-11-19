"""Assay-to-gene mapping for Olink and SomaScan: the assay annotation (vendor lists + UniProt update layer)."""

from __future__ import annotations

import logging
from pathlib import Path

from analysis.external.vendor_assay_tables import ANNOTATION_PATH, assay_genes

logger = logging.getLogger(__name__)


def load_assay_genes(platform: str, annotation: Path = ANNOTATION_PATH) -> dict[str, list[str]]:
    """Assay ID (OlinkID or SeqId) -> gene symbols (1-to-many) from the assay annotation, for gene-level steps.

    A multi-target assay maps to the genes of all its accessions, and one accession can carry several
    genes (identical proteins from paralogous loci); an assay without a gene maps to an empty list.
    Assays marked ``Excluded_From_Gene_Level`` are omitted from gene-level analyses.
    The annotation is refused when built from another update layer or alias table
    (``vendor_assay_tables.read_annotation``).

    Args:
        platform: ``olink`` or ``soma``.
        annotation: The long annotation table (stage ``vendor_assay_tables``).
    """
    mapping = assay_genes(platform, annotation, gene_level=True)
    multi = sum(1 for genes in mapping.values() if len(genes) > 1)
    unmapped = sum(1 for genes in mapping.values() if not genes)
    logger.info(f"{platform} mapping: {len(mapping)} assays ({multi} multi-gene, {unmapped} without a gene)")
    return mapping
