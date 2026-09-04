from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _import_table_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tables"))
    table_annotations = importlib.import_module("_annotations")
    suppl_pathway = importlib.import_module("suppl_pathway_enrichment")
    return importlib.reload(table_annotations), importlib.reload(suppl_pathway)


def _synthetic_uniprot_map() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uniprot": ["P29460", "P06756", "P05106", "Q99999"],
            "gene_symbol": ["IL12B", "ITGAV", "ITGB3", "REG4"],
            "olink_olinkids": [
                ["OID43842", "OID43843"],
                None,
                None,
                ["OID40001"],
            ],
            "olink_assays": [
                ["IL12A_IL12B", "IL12B"],
                None,
                None,
                ["REG4"],
            ],
            "soma_seqids": [
                None,
                ["20187-10"],
                ["20187-10"],
                ["15613-16"],
            ],
            "soma_targets": [
                None,
                ["Integrin aVb3"],
                ["Integrin aVb3"],
                ["REG4"],
            ],
            "present_olink": [True, False, False, True],
            "present_soma": [False, True, True, True],
        }
    )


def test_build_marker_annotations_pairs_vendor_labels_positionally(monkeypatch):
    table_annotations, _ = _import_table_modules(monkeypatch)
    monkeypatch.setattr(table_annotations, "_load_uniprot_map", _synthetic_uniprot_map)

    olink, soma = table_annotations.build_marker_annotations()
    olink_rows = {row["marker"]: row for row in olink.to_dicts()}
    soma_rows = {row["marker"]: row for row in soma.to_dicts()}

    assert olink_rows["OID43842"]["Assay"] == "IL12A_IL12B"
    assert olink_rows["OID43843"]["Assay"] == "IL12B"
    assert soma_rows["20187-10"]["Target"] == "Integrin aVb3"
    assert soma_rows["20187-10"]["Gene_Symbol"] == "ITGAV; ITGB3"


def test_build_marker_annotations_rejects_mismatched_parallel_lists(monkeypatch):
    table_annotations, _ = _import_table_modules(monkeypatch)
    bad = _synthetic_uniprot_map().with_columns(
        pl.when(pl.col("uniprot") == "P29460")
        .then(pl.lit(["IL12A_IL12B"]))
        .otherwise(pl.col("olink_assays"))
        .alias("olink_assays")
    )
    monkeypatch.setattr(table_annotations, "_load_uniprot_map", lambda: bad)

    with pytest.raises(ValueError, match="Non-parallel olink_olinkids/olink_assays"):
        table_annotations.build_marker_annotations()


def test_load_marker_gene_pairs_preserves_multi_gene_markers(monkeypatch):
    table_annotations, _ = _import_table_modules(monkeypatch)
    monkeypatch.setattr(table_annotations, "_load_uniprot_map", _synthetic_uniprot_map)

    pairs = set(table_annotations.load_marker_gene_pairs().iter_rows())

    assert ("OID43843", "IL12B") in pairs
    assert ("20187-10", "ITGAV") in pairs
    assert ("20187-10", "ITGB3") in pairs


def test_sig_genes_for_cell_uses_assay_fallback_for_unmapped_markers(monkeypatch):
    _, suppl_pathway = _import_table_modules(monkeypatch)

    proteins = pl.DataFrame(
        {
            "marker": ["M1", "M2", "M3"],
            "Assay": ["REG4", "Integrin aVb3", "IGNORED"],
            "fdr": [0.01, 0.02, 0.20],
        }
    )
    marker_gene = pl.DataFrame(
        {
            "marker": ["M2", "M2"],
            "gene_symbol": ["ITGAV", "ITGB3"],
        }
    )

    genes = suppl_pathway._sig_genes_for_cell(proteins, marker_gene, 0.05)

    assert genes == {"REG4", "ITGAV", "ITGB3"}
