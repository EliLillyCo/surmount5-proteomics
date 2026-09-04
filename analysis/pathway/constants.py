"""Constants and defaults shared across the pathway analysis package."""

from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "surmount5-proteomics" / "pathway_gene_sets"

# ---------------------------------------------------------------------------
# Gene-set library groups
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

DISEASE_GENETICS_LIBRARIES = [
    "GWAS_Catalog_2025",
    "Jensen_DISEASES_Curated_2025",
    "Human_Phenotype_Ontology",
]

TISSUE_CELLTYPE_LIBRARIES = [
    "GTEx_Tissues_V8_2023",
    "CellMarker_2024",
]

DRUG_PERTURBATION_LIBRARIES = [
    "DGIdb_Drug_Targets_2024",
]

DEFAULT_GENE_SETS = (
    PATHWAY_LIBRARIES
    + ONTOLOGY_LIBRARIES
    + DISEASE_GENETICS_LIBRARIES
    + TISSUE_CELLTYPE_LIBRARIES
    + DRUG_PERTURBATION_LIBRARIES
)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED = 0
