"""Constants and defaults shared across the pathway analysis package."""

from pathlib import Path

from analysis.statistics import SEED as SEED

# The Enrichr libraries as JSON (``{set_name: [genes]}``), one ``<library>.json`` each, written by
# the gene_sets stage (analysis/external/gene_sets.py) and read by the pathway CLI (``--cache-dir``).
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "gene_sets"

# ---------------------------------------------------------------------------
# Gene-set libraries (Enrichr names): three pathway and three ontology libraries plus
# CellMarker. The same seven serve cameraPR and ORA.
# ---------------------------------------------------------------------------
PATHWAY_LIBRARIES = [
    "Reactome_Pathways_2024",
    "KEGG_2026",
    "WikiPathways_2024_Human",
]

ONTOLOGY_LIBRARIES = [
    "GO_Biological_Process_2025",
    "GO_Molecular_Function_2025",
    "GO_Cellular_Component_2025",
]

CELL_TYPE_LIBRARIES = [
    "CellMarker_2024",
]

DEFAULT_GENE_SETS = PATHWAY_LIBRARIES + ONTOLOGY_LIBRARIES + CELL_TYPE_LIBRARIES

# Library members are translated to our gene symbols through the HGNC alias table when a library loads
# (analysis/external/hgnc_aliases.py), except the sets of a library whose name ends with the
# suffix given here. CellMarker's mouse sets are mapped through HCOP's human orthologs
# (analysis/external/hcop.py).
UNTRANSLATED_SETS = {"CellMarker_2024": " Mouse"}

# ---------------------------------------------------------------------------
# cameraPR settings
# ---------------------------------------------------------------------------
# Fixed cameraPR inter-gene correlation, passed explicitly and recorded in each result row.
CAMERA_INTER_GENE_COR = 0.01

# Measured-gene bounds of a tested set (after overlap with the ranked vector).
CAMERA_MIN_SIZE = 5
CAMERA_MAX_SIZE = 500

# ---------------------------------------------------------------------------
# Pinned input digests: SHA-256 of the exact gene-set files. The
# gene-set JSONs are Enrichr downloads (stage gene_sets); each is verified before use
# (analysis/pathway/verify.py). The assay -> gene mapping is the assay annotation
# (analysis/external/vendor_assay_tables.py), whose vendor inputs are pinned there.
# ---------------------------------------------------------------------------
GENE_SET_SHA256 = {
    "CellMarker_2024": "431fefa362e69391a9f7253672a39904eaa5472f2e390a4feefcca740ffbca6e",
    "GO_Biological_Process_2025": "b462ef8ccc7e9ba2ce4d9715eb615fb42a55aaea76c85a70c794c0fd67450475",
    "GO_Cellular_Component_2025": "f83d142e4a549fb997a8ca26084bff70e1029a18e4383bd87412dcda7a649dc3",
    "GO_Molecular_Function_2025": "0f64072668ef60eb5f979335de660d6006cd574a83ff02bde8ff22a2bc78256e",
    "KEGG_2026": "4daa93b69de2188d360114c6c6c1ba399822c16c46852b8505379a4be69ee8a4",
    "Reactome_Pathways_2024": "296f19bef3f3bcd27c2687fbb73badf7383cb4d950defb34983f53eeac09b493",
    "WikiPathways_2024_Human": "f2adf57d5cd9bb21b71ea40dbc8247cec63475342e6312612244acf708958659",
}

