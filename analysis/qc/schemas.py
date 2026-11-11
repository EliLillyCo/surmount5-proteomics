"""Shared output schemas (leaf module: imports nothing from the package)."""

from __future__ import annotations

import polars as pl

# The per-tube pQTL calls of pqtl.classify_pqtl.
PQTL_CALLS: tuple[str, ...] = ("match", "mismatch", "none")
# How an ungenotyped label's tubes are named (pqtl_ungenotyped_naming): "all_tubes"
# needs every fit tube to name one genotype, "two_agree" needs two or more to name one
# genotype and none to name another.
UNGENOTYPED_NAMINGS: tuple[str, ...] = ("all_tubes", "two_agree")
# Ancestry-aware pQTL null (pqtl_null_by_ancestry): "pcs" removes each tube's location
# by genotype principal components from the candidates' z. NULL_SCOPES (pqtl_null_scope)
# picks whose candidates it acts on.
NULL_BY_ANCESTRY: tuple[str, ...] = ("off", "pcs")
NULL_SCOPES: tuple[str, ...] = ("ungenotyped", "all")
# Where the pQTL step's genotype summaries come from. pqtl_kinship_source: "table" reads
# the genome-wide KING table (pqtl_kinship_table), "catalog" computes KING from the
# catalog-variant dosages. pqtl_pcs_source:
# "projected" reads the configured ancestry PC file (pqtl_ancestry_pcs), "catalog" computes
# PCs from the catalog-variant dosages. See genome_tables.py.
KINSHIP_SOURCES: tuple[str, ...] = ("table", "catalog")
PCS_SOURCES: tuple[str, ...] = ("projected", "catalog")
# Identity event types, in the order propose_scored assigns them.
IDENTITY_EVENT_TYPES: tuple[str, ...] = (
    "reciprocal",
    "one_way",
    "duplicate_draw",
    "unplaceable",
    "multi_match",
    "unattributable",
    "two_tube_mismatch",
    "majority_foreign",
    "unresolved",
    "vetoed",
    "ambiguous",
    "pqtl_only",
    "pqtl_mixture",  # Only under pqtl_mixture_warn.
)

# One row per genotype of a subject with proteomics, plus the genotypes with no
# proteomics that the evidence names (pqtl.dna_link_table). IID is the USUBJID.
DNA_LINK_SCHEMA: dict[str, pl.DataType] = {
    "USUBJID": pl.String,
    "IID": pl.String,
    "n_fit_tubes": pl.Int64,
    "n_match": pl.Int64,
    "n_mismatch": pl.Int64,
    "n_none": pl.Int64,
    "linked_z": pl.Float64,
    "best_other_USUBJID": pl.String,
    "best_other_z": pl.Float64,
    "typical_linked_z": pl.Float64,
    "link_mismatch": pl.Boolean,
    "duplicate_USUBJID": pl.String,
    "duplicate_kinship": pl.Float64,
    "relink_USUBJID": pl.String,
    "genetic_sex": pl.Int64,
    "chrx_f": pl.Float64,
    "chrx_het_rate": pl.Float64,
    "chry_call_rate": pl.Float64,
    "sex_karyotype": pl.String,
    "n_dna_sex_discordant": pl.Int64,
    # Karyotype note, annotation only (pqtl.dna_link_table).
    "proteome_sex_vs_genome": pl.String,
    "sex_note": pl.String,
    "n_pqtl_only": pl.Int64,
    "link_status": pl.String,
    "link_reason": pl.String,
    "trial": pl.String,
    "platform": pl.String,
}

# Column contract of the identity events output; readers depend on this order and these
# types.
IDENTITY_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "SampleID": pl.String,
    "USUBJID_LABELED": pl.String,
    "VISITNUM_LABELED": pl.Float64,
    "USUBJID_INFERRED": pl.String,
    "VISITNUM_INFERRED": pl.Float64,
    "event_type": pl.String,
    "action": pl.String,
    # The sample flag the event sets; null for a reassigned tube.
    "identity_flag": pl.String,
    # What named the foreign tube of a two-tube subject: pqtl, sex, age or none.
    "resolved_by": pl.String,
    "veto_reason": pl.String,
    "unplaced_reason": pl.String,
    "self_z": pl.Float64,
    "best_z": pl.Float64,
    "margin": pl.Float64,
    "trial": pl.String,
    "platform": pl.String,
    "SITEID": pl.String,
    "ADT": pl.Date,
    # The ADT_SOURCE the tube had before any move.
    "ADT_SOURCE": pl.String,
    # Days from the tube's ADT to the inferred subject's nearest record.
    "date_gap_days": pl.Int64,
    "p_male": pl.Float64,
    "pred_age": pl.Float64,
    "same_site": pl.Boolean,
    "genotype_kinship": pl.Float64,
    "genotype_unrelated": pl.Boolean,
    "pqtl_self_z": pl.Float64,
    "pqtl_best_z": pl.Float64,
    "pqtl_best_USUBJID": pl.String,
    "pqtl_inferred_z": pl.Float64,
    "pqtl_call": pl.String,
    "dna_sex_concordant": pl.Boolean,
    "well_position": pl.String,
}

# Lp scores of every tube in the sample QC summary: the final scores after the anchor
# rescoring (read by the decision rule), then the pass-1 best and
# second, which name a donor whose own tubes are not anchors. Null on unscored tubes.
IDENTITY_SCORE_COLUMNS: dict[str, pl.DataType] = {
    "identity_self_z": pl.Float64,
    "identity_best_USUBJID": pl.String,
    "identity_best_z": pl.Float64,
    "identity_second_z": pl.Float64,
    "identity_pass1_best_USUBJID": pl.String,
    "identity_pass1_best_z": pl.Float64,
    "identity_pass1_second_z": pl.Float64,
}

# Unfit-tube identity annotation (identity.unfit_identity): one row per technically
# failed tube, scored on pass-1 Lp against clean tubes only. It changes no event, flag
# or Status. Classes, in rule order of the self z: consistent_with_label (self z at or
# above self_max, on a failure kind with a reliable score), names_other_subject (self z
# below self_max, pass-1 best z at or above the fixed floor of 8 with the margin, best
# subject not a KING relative), no_comparator (no other clean tube of the subject),
# uninformative (everything else).
UNFIT_IDENTITY_CLASSES: tuple[str, ...] = (
    "consistent_with_label",
    "names_other_subject",
    "no_comparator",
    "uninformative",
)
UNFIT_IDENTITY_SCHEMA: dict[str, pl.DataType] = {
    "SampleID": pl.String,
    "USUBJID_LABELED": pl.String,
    "VISITNUM_LABELED": pl.Float64,
    # The tube's technical FAIL flags, sorted, ";"-joined.
    "technical_flags": pl.String,
    # Their short names, "+"-joined (median_iqr, pca, rowcheck, sample_qc).
    "failure_kind": pl.String,
    # Largest |z| of the failing outlier statistics minus the cut; null without one.
    "z_over_cut": pl.Float64,
    # The failure kind keeps a usable identity signal.
    "score_reliable": pl.Boolean,
    # One of UNFIT_IDENTITY_CLASSES.
    "unfit_identity": pl.String,
    "self_z": pl.Float64,
    # Pass-1 best other subject, against clean tubes.
    "best_USUBJID": pl.String,
    "best_z": pl.Float64,
    "second_z": pl.Float64,
    "margin": pl.Float64,
    # The best subject is a KING relative of the label; null with no best subject.
    "best_related": pl.Boolean,
    "same_site": pl.Boolean,
    # Null when pQTL is off.
    "pqtl_call": pl.String,
    "pqtl_self_z": pl.Float64,
    "pqtl_best_USUBJID": pl.String,
    # False on a tube that fails PCA or Median/IQR; null when pQTL is off.
    "pqtl_interpretable": pl.Boolean,
    # "no sex or age evidence".
    "sex_age_evidence": pl.String,
}
# Sample QC summary column holding the tube's unfit_identity class; null elsewhere.
UNFIT_IDENTITY_SUMMARY_COLUMN: str = "identity_unfit"


def pqtl_named_donor(columns) -> pl.Expr:
    """Builds the genotype a pQTL ``mismatch`` names, for every reader of the calls.

    The donor is the KING relative in ``pqtl_relative_USUBJID`` when the frame has
    that column and it is set, else ``pqtl_best_nonrel_USUBJID``. It is null on a tube
    that is no mismatch.

    Args:
        columns: The frame's column names. A frame without ``pqtl_relative_USUBJID``
            reads as the non-relative donor.

    Returns:
        A string expression.
    """
    donor = pl.col("pqtl_best_nonrel_USUBJID")
    if "pqtl_relative_USUBJID" in columns:
        donor = pl.coalesce(pl.col("pqtl_relative_USUBJID"), donor)
    return (
        pl.when(pl.col("pqtl_call") == "mismatch")
        .then(donor)
        .otherwise(pl.lit(None, dtype=pl.String))
    )


# One row per scored tube (pqtl_reference.pqtl_concordance_for_qc).
PQTL_SCORES_SCHEMA: dict[str, pl.DataType] = {
    "SampleID": pl.String,
    "USUBJID": pl.String,
    "pqtl_self_z": pl.Float64,
    "pqtl_best_z": pl.Float64,
    "pqtl_best_USUBJID": pl.String,
    # Null when the genome-wide KING table does not list the pair (unrelated).
    "pqtl_best_kinship": pl.Float64,
    "pqtl_best_nonrel_z": pl.Float64,
    "pqtl_best_nonrel_USUBJID": pl.String,
    # The KING relative a mismatch names under pqtl_relative_name_margin; null on
    # every other tube.
    "pqtl_relative_USUBJID": pl.String,
    "pqtl_call": pl.String,
    # Two-genotype mixture score against the labeled genotype
    # (pqtl_reference.mixture_scores); null for an ungenotyped label and for a link
    # that was dropped or held.
    "pqtl_mixture_z": pl.Float64,
    "dna_sex_concordant": pl.Boolean,
    "trial": pl.String,
    "platform": pl.String,
}
