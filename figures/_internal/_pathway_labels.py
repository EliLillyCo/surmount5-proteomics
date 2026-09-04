"""Pathway-label cleanup helpers shared by fig3, supp_fig11, and the supp tables.

The figure config CSVs (``pathway_figure_terms.csv`` and
``pathway_ora_figure_terms.csv``) keep two label columns:

  * ``original_term`` — the verbatim term name as exported by ``cameraPR`` /
    ORA from each upstream database (Reactome, GO, KEGG, etc.).
  * ``display_label`` — a hand-curated short label kept around for theme-level
    grouping and as a stable join key for downstream code.

For the rendered pathway labels we want **the original database term**, lightly
cleaned (drop accession suffixes, fix ALL-CAPS), with hand-shortened forms
applied where the verbatim term is too long for the figure gutter or where
contracting a well-known compound noun (VEGF, IGF-I, TGF-β, ECM, SHBG) keeps
meaning intact.

Two original terms have manual *curated overrides* — the curated ``display_label``
is used in place of the cleaned-original because the source DB's term is more
generic than what we actually mean by the row:

  * ``Receptor Antagonist Activity`` → ``Wnt and Myostatin Antagonists``
    (the GO:MF term is generic; the curated label captures the dominant gene
    set in that row, which is Wnt/Myostatin antagonists).
  * ``LTC4-CYSLTR Mediated IL4 Production`` → ``Leukotriene–IL-4 Axis``
    (Reactome's mechanistic phrasing is gene-symbol-heavy; the curated phrase
    is the conventional axis name).

This module exposes ``pathway_label(original_term, curated)`` which figures and
the supplement-table builder both call.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Cleanup primitives
# ---------------------------------------------------------------------------


def _clean_term(term: str) -> str:
    """Strip accession suffixes and fix ALL-CAPS text from raw DB term names.

    * ``(GO:NNNNNNN)`` trailing accession → dropped.
    * ``WPNNNN`` trailing WikiPathways code → dropped.
    * ALL-CAPS strings (typical of KEGG / Jensen DISEASES) → first-letter
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
    "Binding and Uptake of Ligands by Scavenger Receptors": "Scavenger-Receptor Ligand Uptake",
    "Regulation of IGF Transport and Uptake by Insulin-like Growth Factor Binding Proteins (IGFBPs)": "IGFBP-Mediated IGF Transport",
    "Signaling by TGFB Family Members": "TGF-β Family Signaling",
    # CellMarker — reorder Cell-Type Tissue Species → Tissue Cell-Types
    "Endocrine Cell Pancreas Human": "Pancreatic Endocrine Cells",
    "Enteroendocrine Cell Intestine Mouse": "Intestinal Enteroendocrine Cells",
    "Enteroendocrine Cell Intestinal Crypt Mouse": "Intestinal Enteroendocrine Cells",
    "Acinar Cell Pancreatic Islet Human": "Pancreatic Acinar Cells",
    "Enterocyte Intestinal Crypt Mouse": "Intestinal Enterocytes",
    # GWAS Catalog
    "Sex Hormone-Binding Globulin Levels": "SHBG Levels",
    "Hdl Levels": "HDL Levels",
    # WikiPathways
    "Complement And Coagulation Cascades": "Complement & Coagulation Cascades",
    "Leptin And Adiponectin": "Leptin & Adiponectin Signaling",
    # KEGG (post-clean): polish casing of multi-word terms
    "Cytoskeleton in muscle cells": "Muscle Cell Cytoskeleton",
    "Hypertrophic cardiomyopathy": "Hypertrophic Cardiomyopathy",
    "Pancreatic secretion": "Pancreatic Secretion",
    "Cytokine-cytokine receptor interaction": "Cytokine-Cytokine Receptor Interaction",
    "Ecm-receptor interaction": "ECM-Receptor Interaction",
    # Jensen DISEASES
    "Hemolytic-uremic syndrome": "Hemolytic–Uremic Syndrome",
    "Immune system disease": "Immune System Disease",
}


# Manual *curated overrides* — original DB terms whose meaning is too generic
# to use verbatim, where the curated display_label is the better label. Keyed
# by cleaned original_term.
CURATED_OVERRIDES: dict[str, str] = {
    # GO:MF — generic "antagonist activity" hides the Wnt/Myostatin specificity
    "Receptor Antagonist Activity": "Wnt and Myostatin Antagonists",
    # Reactome — gene-symbol-heavy mechanistic phrasing; curated axis name reads cleaner
    "LTC4-CYSLTR Mediated IL4 Production": "Leukotriene–IL-4 Axis",
}


def pathway_label(original_term: str, curated: str | None = None) -> str:
    """Return the figure/table display label for a pathway row.

    Resolution order:
      1. ``CURATED_OVERRIDES[cleaned]`` if present (curation explicitly preferred).
      2. ``SHORT_FORMS[cleaned]`` if present (hand-shortened form).
      3. Cleaned ``original_term`` (accession-suffix stripped, casing fixed).

    The ``curated`` argument is accepted for parity with the figure mapping
    CSV's ``display_label`` column but is currently unused — a future caller
    could pass it to make the resolution data-driven. We keep CURATED_OVERRIDES
    explicit so the override list is auditable in code.
    """
    del curated  # accepted for future use; resolution is currently dict-driven
    cleaned = _clean_term(original_term)
    if cleaned in CURATED_OVERRIDES:
        return CURATED_OVERRIDES[cleaned]
    return SHORT_FORMS.get(cleaned, cleaned)


__all__ = ["pathway_label", "SHORT_FORMS", "CURATED_OVERRIDES"]
