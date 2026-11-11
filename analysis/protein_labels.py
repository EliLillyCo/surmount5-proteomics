"""Shared figure protein labels.

A label names an assay (an OlinkID or a SomaScan SeqId) by its genes in the assay annotation:

* one gene: the gene (``MRFAP1L2``);
* two genes: ``A/B`` (``IL12A/IL12B``, ``C4A/C4B``);
* more than two: ``FIRST +N``, N the number of other genes (``AMY1A +4``);
* genes in alphabetical order ignoring case (ties by the exact symbol), so amylase reads
  ``AMY1A +4`` whatever the accession order and ``C1orf43`` sorts beside ``C1QA``;
* an analyte assay (update-layer reason ``analyte label``) shows the vendor's assay name instead
  (``INS-CPEPTIDE``, ``NT-proBNP``); the annotation passes the name as ``analyte``;
* HGNC duplicate-copy symbols (``C4B_2``, ``GTF2H2C_2``) stay in the gene lists and are left out of the
  label.

The annotation stores the label as ``Display_Label`` (``analysis.external.vendor_assay_tables``);
``marker_labels`` reads it for the figures. Tables keep the full gene list and the vendor names.
Two assays of one panel can still share a label (two SomaScan aptamers of one gene).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


# HGNC duplicate-copy symbols (C4B_2, GTF2H2C_2): kept in the gene lists, left out of the label.
COPY_SYMBOL = re.compile(r".+_\d+$")


def protein_label(genes: Iterable[str | None], analyte: str | None = None) -> str | None:
    """The figure label of one assay; None when it has no gene and is no analyte."""
    if analyte:
        return analyte
    named = {g for g in genes if g}
    unique = sorted(
        {g for g in named if not COPY_SYMBOL.fullmatch(g)} or named, key=lambda g: (g.casefold(), g)
    )
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]}/{unique[1]}"
    return f"{unique[0]} +{len(unique) - 1}"


def marker_labels(platform: str, annotation: Path | None = None) -> dict[str, str]:
    """Assay ID -> ``Display_Label`` for one platform (``olink`` or ``soma``); assays without one are absent."""
    import polars as pl

    from analysis.external.vendor_assay_tables import ANNOTATION_PATH, read_annotation

    long = read_annotation(annotation or ANNOTATION_PATH).filter(
        (pl.col("Platform") == platform) & pl.col("Display_Label").is_not_null()
    )
    return dict(long.select("AssayID", "Display_Label").unique(maintain_order=True).iter_rows())
