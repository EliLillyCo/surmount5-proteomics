"""Pathway-label cleanup shared by Fig 3a, Supplementary Fig 8 and Tables S15/S16.

The figure config CSVs (``pathway_figure_terms.csv`` and
``pathway_ora_figure_terms.csv``) keep two label columns:

  * ``original_term`` — the verbatim term name as exported by ``cameraPR`` /
    ORA from each upstream database (Reactome, GO, KEGG, etc.).
  * ``display_label`` — the configured grouping and join key.

``pathway_label`` cleans the original term and applies a matching short form.

This module exposes ``pathway_label(original_term)`` which figures and
the supplement-table builder both call.
"""

from __future__ import annotations

import re

from analysis.pathway.constants import DEFAULT_GENE_SETS

# ---------------------------------------------------------------------------
# Cleanup primitives
# ---------------------------------------------------------------------------


def _clean_term(term: str) -> str:
    """Strip accession suffixes and fix ALL-CAPS text from raw DB term names.

    * ``(GO:NNNNNNN)`` trailing accession → dropped.
    * ``WPNNNN`` trailing WikiPathways code → dropped.
    * ALL-CAPS strings (typical of KEGG) → first-letter
      capitalized, rest lowercased.
    """
    term = re.sub(r"\s*\(GO:\d+\)\s*$", "", term)
    term = re.sub(r"\s+WP\d+\s*$", "", term)
    if term == term.upper() and any(ch.isalpha() for ch in term):
        term = term[0] + term[1:].lower()
    return term.strip()


# ---------------------------------------------------------------------------
# Hand-shortened forms — keyed by the cleaned form of original_term.
# ---------------------------------------------------------------------------
#
# Edits should preserve the scientific meaning of the source term. Typical
# moves: contract a well-known compound noun (VEGF, IGF-I, ECM), drop a
# verbose qualifier ("Positive Regulation of X" → "X" when reordering doesn't
# change meaning), reorder awkward CellMarker word-order (Cell Type Tissue
# Species → Tissue Cell Types).
SHORT_FORMS: dict[str, str] = {
    # GO Biological Process
    "Positive Regulation of Lipoprotein Particle Clearance": "Lipoprotein Particle Clearance",
    "Cellular Response to Growth Hormone Stimulus": "Response to Growth Hormone",
    "Cellular Response to Glucose Stimulus": "Response to Glucose",
    "Negative Regulation of Appetite": "Appetite Suppression",
    "Vascular Endothelial Growth Factor Receptor Signaling Pathway": "VEGF Receptor Signaling",
    "Induction of Positive Chemotaxis": "Positive Chemotaxis",
    "Positive Regulation of Synapse Assembly": "Synapse Assembly",
    "Semaphorin-Plexin Signaling Pathway": "Semaphorin–Plexin Signaling",
    "Lymphocyte Migration Into Lymphoid Organs": "Lymphocyte Homing",
    "Mast Cell Activation Involved in Immune Response": "Mast Cell Activation",
    "Positive Regulation of Positive Chemotaxis": "Positive Chemotaxis Regulation",
    "High-Density Lipoprotein Particle Remodeling": "HDL Particle Remodeling",
    "Triglyceride Catabolic Process": "Triglyceride Catabolism",
    "Cellular Response to Leptin Stimulus": "Response to Leptin",
    "Cell-Cell Adhesion via Plasma-Membrane Adhesion Molecules": "Plasma-Membrane Cell–Cell Adhesion",
    "Insulin-Like Growth Factor Receptor Signaling Pathway": "IGF Receptor Signaling",
    # GO Molecular Function
    "Insulin-Like Growth Factor I Binding": "IGF-1 Binding",
    "Insulin-Like Growth Factor Binding": "IGF Binding",
    "Transforming Growth Factor Beta Binding": "TGF-β Binding",
    "Fibroblast Growth Factor Binding": "FGF Binding",
    # GO Cellular Component
    "Collagen-Containing Extracellular Matrix": "Collagen-Containing ECM",
    # Reactome
    "Formation of Fibrin Clot (Clotting Cascade)": "Fibrin Clot Formation",
    "Synthesis, Secretion, and Deacylation of Ghrelin": "Ghrelin Synthesis & Secretion",
    "Regulation of Complement Cascade": "Complement Cascade Regulation",
    "Activation of C3 and C5": "C3 & C5 Activation",
    "Binding and Uptake of Ligands by Scavenger Receptors": "Scavenger-Receptor Ligand Uptake",
    "Regulation of IGF Transport and Uptake by Insulin-like Growth Factor Binding Proteins (IGFBPs)": "IGFBP-Mediated IGF Transport",
    "Signaling by TGFB Family Members": "TGF-β Family Signaling",
    "Receptor-type Tyrosine-Protein Phosphatases": "Receptor Tyrosine Phosphatases",
    # CellMarker — reorder Cell-Type Tissue Species → Tissue Cell-Types
    "Endocrine Cell Pancreas Human": "Pancreatic Endocrine Cells",
    "Enteroendocrine Cell Intestine Mouse": "Intestinal Enteroendocrine Cells",
    "Enteroendocrine Cell Intestinal Crypt Mouse": "Intestinal Enteroendocrine Cells",
    "Acinar Cell Pancreatic Islet Human": "Pancreatic Acinar Cells",
    "Enterocyte Intestinal Crypt Mouse": "Intestinal Enterocytes",
    "Skeletal Stem And Progenitor Cell (SSPC) Cartilage Mouse": "Skeletal Stem/Progenitor (SSPC)",
    "Adipocyte Adipose Tissue Mouse": "Adipose Tissue Adipocytes",
    "Mural Cell Undefined Mouse": "Mural Cells",
    "Lymphoid Cell Undefined Human": "Lymphoid Cells",
    "Pericyte Kidney Mouse": "Kidney Pericytes",
    "Oligodendrocyte Progenitor Cell Brain Human": "Brain Oligodendrocyte Progenitors",
    "Mesenchymal Stromal Cell Bone Marrow Human": "Bone-Marrow Stromal Cells",
    # WikiPathways
    "Complement And Coagulation Cascades": "Complement & Coagulation Cascades",
    "Leptin And Adiponectin": "Leptin & Adiponectin Signaling",
    "Leptin Insulin Signaling Overlap": "Leptin–Insulin Signaling Overlap",
    "Netrin UNC5B Signaling": "Netrin–UNC5B Signaling",
    "TGFB Smad Signaling": "TGF-β–SMAD Signaling",
    # KEGG (post-clean): polish casing of multi-word terms
    "Cytoskeleton in muscle cells": "Muscle Cell Cytoskeleton",
    "Hypertrophic cardiomyopathy": "Hypertrophic Cardiomyopathy",
    "Pancreatic secretion": "Pancreatic Secretion",
    "Cytokine-cytokine receptor interaction": "Cytokine-Cytokine Receptor Interaction",
    "Ecm-receptor interaction": "ECM-Receptor Interaction",
    "Igsf cam signaling": "IGSF CAM Signaling",
    "Growth hormone synthesis, secretion and action": "GH Synthesis, Secretion & Action",
}


# Enrichr library name -> display name for the gray source suffix on the Fig 3a term labels.
SOURCE_NAMES: dict[str, str] = {
    "Reactome_Pathways_2024": "Reactome",
    "KEGG_2026": "KEGG",
    "WikiPathways_2024_Human": "WikiPathways",
    "GO_Biological_Process_2025": "GO BP",
    "GO_Molecular_Function_2025": "GO MF",
    "GO_Cellular_Component_2025": "GO CC",
    "CellMarker_2024": "CellMarker",
}

_unnamed = [lib for lib in DEFAULT_GENE_SETS if lib not in SOURCE_NAMES]
if _unnamed:
    raise KeyError(f"SOURCE_NAMES has no display name for gene-set libraries: {_unnamed}")


def pathway_label(original_term: str) -> str:
    """Return the figure/table display label for a pathway row.

    Resolution order:
      1. ``SHORT_FORMS[cleaned]`` if present (hand-shortened form).
      2. Cleaned ``original_term`` (accession-suffix stripped, casing fixed).
    """
    cleaned = _clean_term(original_term)
    return SHORT_FORMS.get(cleaned, cleaned)


__all__ = ["pathway_label", "SHORT_FORMS", "SOURCE_NAMES"]
