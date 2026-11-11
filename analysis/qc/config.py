"""TOML trial configuration with defaults."""

from __future__ import annotations

import logging
import math
import re
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from analysis.qc.age_concordance import AGE_ENET_ALPHA, AGE_ENET_L1_RATIO
from analysis.qc.identity import (
    AGE_DISCORDANT_MADS,
    AGE_FIT_MADS,
    AGE_VETO_RESIDUALS,
    CLIP,
    ICC_CAP,
    ICC_MIN,
    MIN_GROUP,
    N_FOLDS,
    N_MEDIAN_ROUNDS,
    N_PCS,
    TIEBREAKS,
)
from analysis.qc.modeling import CV_FOLDS, KNN_K
from analysis.qc.outliers import OUTLIER_N_PCS, OUTLIER_N_STD
from analysis.qc.platform import OLINK_PC_NPX, Platform, get_platform, qc_platform
from analysis.qc.pqtl import (
    KING_DUPLICATE,
    KING_SECOND_DEGREE,
    KING_UNRELATED,
    SEX_X_HET_ONE,
    SEX_X_HET_TWO,
    SEX_Y_ABSENT,
    SEX_Y_PRESENT,
    SexCallThresholds,
)
from analysis.qc.pqtl_reference import (
    CATALOG_MERGES,
    CONTRIBUTION_CAP,
    MAF_MIN,
    NLP_RECOMPUTE_FLOOR,
    PALINDROMIC,
    PALINDROMIC_MODES,
)
from analysis.qc.schemas import (
    KINSHIP_SOURCES,
    NULL_BY_ANCESTRY,
    NULL_SCOPES,
    PCS_SOURCES,
    UNGENOTYPED_NAMINGS,
)
from analysis.qc.sex_concordance import SEX_C, SEX_PROB_THRESHOLD

logger = logging.getLogger(__name__)

SEX_MODELS = ("l1", "l2")
AGE_MODELS = ("elasticnet", "ridgecv", "ridgecv_grouped")
PQTL_DOSAGE_SCALINGS = ("standardized", "centered")
PQTL_BUILDS = ("hg19", "hg38")
# How a TOML spells identity.veto_unplaceable_best_min as unset (TOML has no null).
UNSET_VETO_UNPLACEABLE_BEST_MIN = "best_min"
SUBJECT_JOINS = ("SUBJID", "USUBJID")  # [manifest] subject_join
# [manifest] keys naming files (resolved against the TOML's directory); its other keys
# are settings.
MANIFEST_FILE_FIELDS = ("raw_key", "adsl", "adlb", "advs", "adig", "lb", "lims")

# Config keys holding file or directory paths; relative values resolve against the
# TOML's directory.
PATH_KEYS = (
    "pqtl_reference_catalog",
    "pqtl_genotype_dir",
    "pqtl_kinship_table",
    "pqtl_ancestry_pcs",
    "advs",
)
# Inputs pQTL concordance cannot run without.
PQTL_REQUIRED_KEYS = ("pqtl_reference_catalog", "pqtl_genotype_dir")
# pQTL fields whose default is the platform's (Platform.pqtl_defaults): field ->
# PqtlDefaults attribute.
PQTL_PLATFORM_KEYS = {
    "pqtl_best_min": "best_min",
    "pqtl_weight_exponent": "weight_exponent",
    "pqtl_match_floor": "match_floor",
    "pqtl_mismatch_floor": "mismatch_floor",
}

# [[corrections]]: fields each type requires, and the optional ones it accepts.
CORRECTION_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "subject_set": (("usubjid_sha256", "set"), ()),
    "rename": (("sampleid_sha256", "strip_suffix"), ()),
    "subject_from_accession": (
        ("sampleid_sha256", "from_usubjid_sha256", "to_usubjid_sha256"),
        (),
    ),
    "exclude": (("sampleid_sha256", "label"), ("usubjid_sha256",)),
    # reassign and visit_set apply in the manifest build and carry an
    # identifier-free label into the manifest's MANIFEST_CORRECTION column.
    "reassign": (
        ("sampleid_sha256", "to_usubjid_sha256", "to_visitnum", "label"),
        ("from_usubjid_sha256",),
    ),
    "visit_set": (("sampleid_sha256", "visitnum", "label"), ("usubjid_sha256",)),
    # bridge_swap: exchanges the kind and tube fields of a bridge well
    # (sampleid_sha256) and a study well (study_sampleid_sha256) in the manifest build;
    # both carry the label in MANIFEST_CORRECTION.
    "bridge_swap": (
        ("sampleid_sha256", "study_sampleid_sha256", "usubjid_sha256", "label"),
        (),
    ),
    # expected_sex: per-subject override of the sex the sex checks compare against.
    # Applied by QC, never by the build.
    "expected_sex": (("usubjid_sha256", "expected_sex"), ()),
    # dna_relink: the genome filed under from_usubjid_sha256 belongs to usubjid_sha256
    # (analysis/qc/dna_relink.py). Applied by the pQTL identity check and the
    # pharmacogenomics stage, never by the build.
    "dna_relink": (("from_usubjid_sha256", "usubjid_sha256", "label"), ()),
}
EXPECTED_SEX_VALUES = ("M", "F")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ManifestColumns:
    """Raw vendor key column names for the manifest build ([manifest.columns])."""

    sample_id: str
    subject: str
    accession: str
    sample_type: str
    visit_label: str | None = None
    date: str | None = None
    bridge_flag: str | None = None
    # The key's sample-volume note column, read by WARN_Low_Volume.
    volume_note: str | None = None


@dataclass(frozen=True)
class ManifestConfig:
    """Inputs of the manifest build ([manifest]); paths are absolute after loading."""

    raw_key: str
    adsl: str
    adlb: str
    advs: str
    columns: ManifestColumns
    adig: str | None = None
    # SDTM LB: the first tube-record source (LBREFID, USUBJID, VISITNUM, LBDTC).
    lb: str | None = None
    lims: str | None = None  # an optional LIMS file joined on Container Barcode
    # Settings, not paths. The ADSL column the key's subject joins: "SUBJID" (leading
    # zeros ignored) or "USUBJID" (exact, for an ADSL with no SUBJID).
    subject_join: str = "SUBJID"
    # A key with no bridge column also marks as bridge a row whose whole subject cell
    # matches this regular expression; None marks bridges by an empty subject only.
    bridge_subject_pattern: str | None = None

    def __post_init__(self):
        """Checks the join setting and the bridge pattern."""
        if self.subject_join not in SUBJECT_JOINS:
            raise ValueError(
                f"[manifest] subject_join must be one of {', '.join(SUBJECT_JOINS)}; got {self.subject_join!r}"
            )
        if self.bridge_subject_pattern is not None:
            try:
                re.compile(self.bridge_subject_pattern)
            except re.error as e:
                raise ValueError(
                    f"[manifest] bridge_subject_pattern is not a regular expression: {e}"
                ) from e
            if self.columns.bridge_flag is not None:
                raise ValueError(
                    "[manifest] bridge_subject_pattern is for a key with no bridge column; this one sets "
                    "[manifest.columns] bridge_flag"
                )

    def files(self) -> dict[str, str]:
        """Returns the input files this block names, in field order.

        Settings and columns are left out.
        """
        return {k: v for k, v in vars(self).items() if k in MANIFEST_FILE_FIELDS and v is not None}


@dataclass(frozen=True)
class Correction:
    """One [[corrections]] entry; IDs are SHA-256 hex digests, never plain IDs.

    ``set_values`` holds the TOML ``set`` table of a ``subject_set`` entry. ``label`` is
    the short, identifier-free text that names an ``exclude`` in the exclusion table
    (and the correction a ``reassign``, ``visit_set`` or ``bridge_swap`` made, in the
    manifest);
    ``reason`` keeps the evidence and stays in the TOML. ``to_visitnum`` and
    ``visitnum`` are visit codes (floats).
    """

    type: str
    reason: str
    sampleid_sha256: str | None = None
    usubjid_sha256: str | None = None
    from_usubjid_sha256: str | None = None
    study_sampleid_sha256: str | None = None
    to_usubjid_sha256: str | None = None
    set_values: dict[str, Any] | None = None
    strip_suffix: str | None = None
    label: str | None = None
    to_visitnum: float | None = None
    visitnum: float | None = None
    expected_sex: str | None = None

    def __post_init__(self):
        """Checks the entry's type, its fields and their formats."""
        if self.type not in CORRECTION_FIELDS:
            raise ValueError(
                f"Unknown correction type {self.type!r}; expected one of "
                f"{', '.join(CORRECTION_FIELDS)}"
            )
        if not self.reason or not self.reason.strip():
            raise ValueError(f"{self.type} correction has no reason")
        # TOML key -> value; the TOML "set" table is the set_values field.
        given = {
            ("set" if f.name == "set_values" else f.name): getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("type", "reason") and getattr(self, f.name) is not None
        }
        required, optional = CORRECTION_FIELDS[self.type]
        missing = [k for k in required if k not in given]
        if missing:
            raise ValueError(f"{self.type} correction lacks {', '.join(missing)}")
        extra = sorted(set(given) - set(required) - set(optional))
        if extra:
            raise ValueError(f"{self.type} correction does not take {', '.join(extra)}")
        if "label" in given and not given["label"].strip():
            raise ValueError(f"{self.type} correction has a blank label")
        for name, value in given.items():
            if name.endswith("_sha256") and not _SHA256.fullmatch(value):
                raise ValueError(
                    f"{self.type} correction {name} is not a lowercase SHA-256 hex digest"
                )
        if (
            "study_sampleid_sha256" in given
            and given["study_sampleid_sha256"] == self.sampleid_sha256
        ):
            raise ValueError(f"{self.type} correction names one well as both bridge and study well")
        for name in ("to_visitnum", "visitnum"):
            if name in given:
                value = given[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{self.type} correction {name} must be a number; got {value!r}"
                    )
                if not math.isfinite(value):
                    raise ValueError(f"{self.type} correction {name} must be finite; got {value}")
                object.__setattr__(self, name, float(value))
        if self.type == "dna_relink" and self.from_usubjid_sha256 == self.usubjid_sha256:
            raise ValueError("dna_relink correction names one participant as both genome holder and owner")
        if "set" in given and not given["set"]:
            raise ValueError("subject_set correction has an empty set table")
        if "expected_sex" in given and given["expected_sex"] not in EXPECTED_SEX_VALUES:
            raise ValueError(
                f"expected_sex correction expected_sex must be one of {', '.join(EXPECTED_SEX_VALUES)}; "
                f"got {given['expected_sex']!r}"
            )


@dataclass(frozen=True)
class TrialConfig:
    """A trial's QC configuration, flattened from its TOML.

    Each TOML table ([identity], [pqtl], [qc], ...) becomes fields prefixed with the
    table name; ``load_config`` does the flattening. Fields left as None that have a
    platform default are resolved on read through the properties (``best_min``,
    ``intra_cv``, ...), except the per-platform pQTL keys, which ``__post_init__``
    writes back.
    """

    study_name: str = ""
    platform: str = "olink"
    # The QC data column (vendor flags, CV, LOD detection, duplicate assays, outliers);
    # None is the platform's. The sex and age models, the Lp identity scores and the pQTL
    # scores always read the platform's identity layer.
    data_col: str | None = None
    # [olink] value_column: the Olink value column QC reads, "PCNormalizedNPX" or
    # "NPX". It sets the default QC data column and the identity layer (sex, age,
    # identity and incretin inputs); detection reads the LOD paired with it. Read the
    # resolved platform from qc_platform. The output carries both columns.
    olink_value_column: str = OLINK_PC_NPX
    # SomaScan: accept an ANML-normalized ADAT (anmlSMP in !ProcessSteps), or one whose header
    # states no steps.
    allow_anml: bool = False
    # [visits]: visit codes are floats, so fractional codes are kept. visits_baseline
    # None means the TOML has no [visits]; the clinical step then raises.
    visits_baseline: float | None = None
    visits_analysis: tuple[float, ...] = ()
    # Screening codes for the pre-dose baseline exception.
    visits_screening: tuple[float, ...] = (1.0,)
    # [manifest]: when set, QC builds the manifest instead of reading
    # --mapping-manifest.
    manifest: ManifestConfig | None = None
    # The trial's ADVS for a config without [manifest] (a path, relative to the TOML); a
    # config with [manifest] names it in [manifest] advs, and setting both raises. Read
    # it from advs_path.
    advs: str | None = None
    corrections: tuple[Correction, ...] = ()
    identity_enabled: bool = True
    # best_min and margin_min come from Platform.identity_defaults and the per-platform
    # pQTL keys from Platform.pqtl_defaults. identity.Rule keeps its own defaults.
    identity_self_max: float = 3.5
    # None means the platform default; read the resolved thresholds from best_min and
    # margin_min.
    identity_best_min: float | None = None
    identity_margin_min: float | None = None
    identity_sex_veto: bool = True
    # TOML has no null: identity_age_veto = false disables the age veto. Read the
    # resolved veto from age_veto_years, which is None when the veto is off. From
    # Python, identity_age_veto_years=None also turns it off.
    identity_age_veto: bool = True
    identity_age_veto_years: float | None = 20.0
    identity_tiebreak: str = "pqtl+sex_age"  # or "pqtl"
    # A tube moves only to a subject with a visit record within this many whole days of
    # the tube's draw date; 0 means the same day.
    identity_reassign_date_tolerance_days: int = 1
    # A positive pQTL match vetoes an Lp proposal and a FAIL.
    identity_pqtl_veto: bool = True
    # Lp scoring; defaults come from the identity module constants. n_pcs: PCs projected
    # out of the residuals. icc_min: a protein is kept when its training-fold ICC is
    # above it. icc_cap: caps the ICC in the weight icc / (1 - min(icc, icc_cap)). clip:
    # residuals clipped to +-clip (0 is off). min_group: an arm x visit group with fewer
    # fit rows falls back to its arm. median_rounds: alternating arm x visit / plate
    # median passes. n_folds: participant folds for the cross-fitted ICC. The
    # residualization also feeds the pQTL scoring.
    identity_n_pcs: int = N_PCS
    identity_icc_min: float = ICC_MIN
    identity_icc_cap: float = ICC_CAP
    identity_clip: float = CLIP
    identity_min_group: int = MIN_GROUP
    identity_median_rounds: int = N_MEDIAN_ROUNDS
    identity_n_folds: int = N_FOLDS
    # Sex and age evidence. sex_prob: the P(male) cut of the sex veto, the two-tube sex
    # tiebreak and the DNA-sex link (confident above it or below 1 - sex_prob).
    # age_discordant_mads, age_fit_mads: the age tiebreak calls a tube age-discordant
    # beyond the first (MADs of the age residual) and fitting the partner within the
    # second. age_veto_residual: "raw" (Predicted_Age) or "corrected" (bias-corrected
    # residual) is compared with the best subject's expected age.
    identity_sex_prob: float = SEX_PROB_THRESHOLD
    identity_age_discordant_mads: float = AGE_DISCORDANT_MADS
    identity_age_fit_mads: float = AGE_FIT_MADS
    identity_age_veto_residual: str = "raw"
    # A tube whose move a sex or age veto blocks, with self z below self_max and no
    # positive pQTL self-match, FAILs as foreign (unplaceable, vetoed_destination) when
    # its blocked match is at least identity_veto_unplaceable_best_min; off, it stays a
    # WARN. The floor is kept as given; None (from Python) or the string "best_min" (the
    # TOML spelling) resolves to best_min on read.
    identity_veto_unplaceable: bool = True
    identity_veto_unplaceable_best_min: float | str | None = 12.0
    # On: the sex veto and the two-tube tiebreak compare a tube with the protein sex of
    # the subject's own anchor tubes (pQTL-matched, else Lp anchors) when no
    # expected_sex correction names the subject and those tubes are all confidently one
    # sex; else with ADSL SEX. Off: the expected_sex exception, else ADSL SEX.
    identity_sex_from_anchors: bool = False
    # A subject of three or more fit tubes whose Lp-flagged tubes and Lp-clean tubes
    # form two groups is split by each group's confident protein sex against the
    # expected sex (the expected_sex exception, else ADSL SEX; never the anchors), then
    # by age: a foreign Lp-clean group FAILs as majority_foreign and the flagged tubes
    # are spared. Needs identity_tiebreak = "pqtl+sex_age".
    identity_split_tiebreak: bool = True
    sex_model: str = "l2"
    # P(male) cut for WARN_Sex_Concordance; the identity evidence cut is
    # identity_sex_prob.
    sex_warn_prob: float = SEX_PROB_THRESHOLD
    age_model: str = "elasticnet"
    # Assay-QC thresholds. None means the platform default (Platform.assay_defaults);
    # read the resolved values from intra_cv, detection_rate, saturation_rfu,
    # saturation_frac and sb_ratio (the last three SomaScan only).
    qc_intra_cv: float | None = None
    qc_detection_rate: float | None = None
    qc_saturation_rfu: float | None = None
    qc_saturation_frac: float | None = None
    qc_sb_ratio: float | None = None
    # Sample outliers: a sample is flagged when any of the first outlier_n_pcs
    # standardized PCs (or the median or IQR z) is beyond outlier_n_std; the outlier
    # matrix's KNN imputation uses outlier_knn_k neighbors. The driver's
    # --outlier-n-components and --outlier-n-std override the first two when given.
    qc_outlier_n_pcs: int = OUTLIER_N_PCS
    qc_outlier_n_std: float = OUTLIER_N_STD
    qc_outlier_knn_k: int = KNN_K
    # Each assay's outlier-matrix weight is its detection rate; False weights every
    # assay 1.0.
    qc_outlier_detection_weighting: bool = True
    # Olink WARN_Low_Volume: a sample whose manifest VOLUME_NOTE contains any of these
    # texts (case-insensitive, literal) is WARN. None: any non-blank note. Read only
    # when the manifest carries VOLUME_NOTE. SomaScan's rule is fixed ("Low Volume").
    qc_low_volume_match: tuple[str, ...] | None = None
    # Sex and age model inputs are every assay but the duplicate copies, with no
    # assay-QC filter; an assay missing in more than this fraction of the samples is
    # dropped.
    model_max_missing: float = 0.1
    # model_cv_folds: grouped CV folds of the sex and age models (and the inner CV of
    # the age bias correction). model_knn_k: neighbors of their KNN imputation. sex_c:
    # the sex model's C (--sex-concordance-c overrides it). age_enet_alpha,
    # age_enet_l1_ratio: the age ElasticNet (the ProtAge clock's values).
    model_cv_folds: int = CV_FOLDS
    model_knn_k: int = KNN_K
    sex_c: float = SEX_C
    age_enet_alpha: float = AGE_ENET_ALPHA
    age_enet_l1_ratio: float = AGE_ENET_L1_RATIO
    # Clip the standardized features to +-limit inside each training fold (None: off).
    sex_feature_clip: float | None = None
    age_feature_clip: float | None = None
    pqtl_enabled: bool = False
    # One parquet path or catalog name (pqtl_reference.CATALOG_NAMES), or a list of them
    # (same platform, hg38). A list is kept as a tuple.
    pqtl_reference_catalog: str | tuple[str, ...] | None = None
    pqtl_genotype_dir: str | None = None  # directory of per-chromosome PGENs
    # The PGEN naming family to read (``pqtl.pgen_prefixes`` syntax, e.g.
    # "TRIAL.chr{chrom}"); None takes the family covering the most chromosomes.
    pqtl_pgen_pattern: str | None = None
    # The genotype summaries of the pQTL step (genome_tables.py). kinship_table: the
    # KING parquet. ancestry_pcs: the PC TSV (dx_usubjid, PC1..PCn).
    # kinship_source "table" needs kinship_table, pcs_source "projected" needs
    # ancestry_pcs while the ancestry null is on; "catalog" computes either from the
    # catalog-variant dosages.
    pqtl_kinship_table: str | None = None
    pqtl_ancestry_pcs: str | None = None
    pqtl_kinship_source: str = "table"
    pqtl_pcs_source: str = "projected"
    # Chromosomes the genotype directory must hold when pQTL is enabled; chrX gives the
    # genetic-sex call.
    pqtl_expected_chromosomes: tuple[str, ...] = (*(str(c) for c in range(1, 23)), "X")
    pqtl_margin_min: float = 0.0
    # weight_exponent: per-side exponent on -log10(p). best_min: the smallest
    # best-other z that names a subject. None is the platform's
    # (Platform.pqtl_defaults), written into the field when the config is built, so
    # every reader reads a number. dataclasses.replace keeps the resolved value, so a
    # config for another platform is loaded anew, not replaced. dosage_scaling:
    # "centered" or "standardized".
    pqtl_weight_exponent: float | None = None
    pqtl_dosage_scaling: str = "centered"
    pqtl_best_min: float | None = None
    # Chance-match mitigations, both off by default. maf_min drops a protein whose lead
    # variant has trial MAF below it; contribution_cap clips each protein's term in a
    # tube-genotype correlation to +-cap (0 is no bound).
    pqtl_maf_min: float = MAF_MIN
    pqtl_contribution_cap: float = CONTRIBUTION_CAP
    # A catalog -log10(p) at or above this is recomputed from beta/se.
    pqtl_nlp_recompute_floor: float = NLP_RECOMPUTE_FLOOR
    # A/T and C/G SNPs: "drop" drops them all; "rescue" orients them by allele
    # frequency where the catalog EAF and the trial frequency are both clear of 0.5
    # (PALINDROMIC_EAF_MARGIN).
    pqtl_palindromic: str = PALINDROMIC
    # pQTL calls per tube: a self z at or above match_floor is a positive match, below
    # mismatch_floor a mismatch, between them no evidence; equal floors give a one-floor
    # rule. None is the platform's, resolved as pqtl_best_min is.
    pqtl_match_floor: float | None = None
    pqtl_mismatch_floor: float | None = None
    # KING kinship at or above relative_kinship makes two genotypes relatives (default:
    # second degree); at or above duplicate_kinship they are one genome. Defaults come
    # from pqtl.py.
    pqtl_relative_kinship: float = KING_SECOND_DEGREE
    pqtl_duplicate_kinship: float = KING_DUPLICATE
    # Genetic sex from two frequency-free axes (pqtl.SexCallThresholds): the chrX
    # heterozygosity rate is one X at or below sex_x_het_one and two X at or above
    # sex_x_het_two; the chrY call rate is absent at or below sex_y_absent and present
    # at or above sex_y_present; between the two cuts of an axis is gray, and a gray
    # sample has no sex call. Defaults come from pqtl.py.
    pqtl_sex_x_het_one: float = SEX_X_HET_ONE
    pqtl_sex_x_het_two: float = SEX_X_HET_TWO
    pqtl_sex_y_absent: float = SEX_Y_ABSENT
    pqtl_sex_y_present: float = SEX_Y_PRESENT
    # When set, a positive match also needs self z minus the best relative's z
    # (relatives at relative_kinship) to be at least this. None: no relative margin.
    pqtl_relative_margin: float | None = None
    # An ungenotyped label's tube counts as a mismatch only when every fit tube of the
    # subject names the same genotype at or above best_min.
    pqtl_ungenotyped_all_tubes: bool = True
    # How an ungenotyped label's tubes are named when ungenotyped_all_tubes is on.
    # "all_tubes": every fit tube names one genotype. "two_agree": two or more fit tubes
    # name one genotype and no fit tube names another; identity then FAILs the Lp-clean
    # tubes as majority_foreign only on an Lp split, sparing the flagged tubes.
    # "two_agree" needs pqtl_null_by_ancestry = "pcs".
    pqtl_ungenotyped_naming: str = "two_agree"
    # Ancestry-aware pQTL null. "off": the pooled null. "pcs": removes each scoped
    # tube's ancestry location (the leave-one-out fit of a robust regression of the
    # tube's z row on the trial's genotype PCs) from every candidate's z; a tube's own z
    # and best_min are untouched. pqtl_null_pcs is the number of PCs (read only when the
    # null is on; pqtl_reference.PCS_MIN_GAP guards the singular-value gap).
    # pqtl_null_scope: "ungenotyped" (tubes of ungenotyped labels) or "all"
    # (report-only).
    pqtl_null_by_ancestry: str = "pcs"
    pqtl_null_pcs: int = 2
    pqtl_null_scope: str = "ungenotyped"
    # When set, a genotyped tube whose best KING relative (below duplicate_kinship) fits
    # at least best_min, fits at least as well as any non-relative, and beats the tube's
    # own z by this margin is a mismatch naming that relative. None: relatives are never
    # named.
    pqtl_relative_name_margin: float | None = None
    # A fit tube with no other identity event whose pqtl_mixture_z is at least this is a
    # pqtl_mixture WARN (WARN_Identity_pQTL); the value comes from
    # pqtl.mixture_warn_threshold. None: no WARN; the column is computed whenever pQTL
    # runs.
    pqtl_mixture_warn: float | None = None
    # Keep the non-EUR discovery rows of an ancestry-tagged catalog.
    pqtl_include_non_eur: bool = True
    # How several reference catalogs combine: "strongest" keeps the strongest row per
    # protein across catalogs; "list_order" lets the first listed catalog with a usable
    # row for a protein supply all of its rows. With one catalog both are the same.
    pqtl_catalog_merge: str = "strongest"
    # Build and platform of a catalog file the registry does not know
    # (pqtl_reference.CATALOG_REGISTRY). For a registered file they must agree with the
    # registry or the load raises.
    pqtl_catalog_build: str | None = None
    pqtl_catalog_platform: str | None = None
    # On-treatment window. A subject whose completion_status ADSL column is
    # DISCONTINUED has a post-baseline draw on treatment through last dose +
    # discontinued_days_after_last_dose; COMPLETED keeps every draw; any other value
    # raises. window_stop_events (ADSL date columns) end either window early at the
    # earliest non-null date.
    clinical_completion_status: str = "EOTSTT"
    clinical_discontinued_days_after_last_dose: int = 7
    clinical_window_stop_events: list[str] = field(default_factory=list)
    # First dose for the pre-dose baseline exception: the first of these columns present
    # in ADSL.
    clinical_first_dose: list[str] = field(default_factory=lambda: ["TRTSDT", "TR01SDT"])
    # Last dose: the first of these columns present in ADSL; exported as TRTEDT in the
    # sample summary.
    clinical_last_dose: list[str] = field(default_factory=lambda: ["TRTEDT", "TR01EDT"])

    def __post_init__(self):
        """Validates every field and writes back the coerced and per-platform values."""
        get_platform(self.platform)  # an unknown platform fails here, not at first read
        self._check_olink_value_column()
        # Defaults are resolved on read (best_min, margin_min, age_veto_years), never
        # written back, so dataclasses.replace re-resolves them for the new values.
        self._coerce("identity_self_max", float)
        for name in ("identity_best_min", "identity_margin_min"):
            if getattr(self, name) is not None:
                self._coerce(name, float, lo=0.0)
        # The per-platform pQTL keys are the exception: written back here, because their
        # readers read the fields.
        pqtl_defaults = get_platform(self.platform).pqtl_defaults
        for name, attr in PQTL_PLATFORM_KEYS.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, getattr(pqtl_defaults, attr))
        self._check_visits()
        if self.advs is not None and self.manifest is not None:
            raise ValueError(
                "advs is set at the top level and in [manifest]; a config with [manifest] names ADVS in [manifest] advs only"
            )
        object.__setattr__(self, "corrections", tuple(self.corrections))
        bad = [c for c in self.corrections if not isinstance(c, Correction)]
        if bad:
            raise ValueError(f"corrections must be Correction entries; got {bad[0]!r}")
        self._check_expected_sex()
        if self.sex_model not in SEX_MODELS:
            raise ValueError(
                f"sex_model must be one of {', '.join(SEX_MODELS)}; got {self.sex_model!r}"
            )
        if self.age_model not in AGE_MODELS:
            raise ValueError(
                f"age_model must be one of {', '.join(AGE_MODELS)}; got {self.age_model!r}"
            )
        if isinstance(self.model_max_missing, bool) or not 0 <= self.model_max_missing < 1:
            raise ValueError(f"model_max_missing must be in [0, 1); got {self.model_max_missing!r}")
        self._coerce("qc_outlier_n_pcs", int, lo=1)
        self._coerce("qc_outlier_n_std", float, lo=0.0, lo_open=True)
        self._coerce("qc_outlier_knn_k", int, lo=1)
        if not isinstance(self.qc_outlier_detection_weighting, bool):
            raise ValueError(
                "qc_outlier_detection_weighting must be true or false; "
                f"got {self.qc_outlier_detection_weighting!r}"
            )
        self._check_low_volume_match()
        self._coerce("model_cv_folds", int, lo=2)
        self._coerce("model_knn_k", int, lo=1)
        self._coerce("sex_c", float, lo=0.0, lo_open=True)
        self._coerce("age_enet_alpha", float, lo=0.0, lo_open=True)
        self._coerce("age_enet_l1_ratio", float, lo=0.0, hi=1.0)
        for name in ("sex_feature_clip", "age_feature_clip"):
            limit = getattr(self, name)
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0:
                    raise ValueError(f"{name} must be a positive number or unset; got {limit!r}")
                object.__setattr__(self, name, float(limit))
        if (
            not isinstance(self.clinical_completion_status, str)
            or not self.clinical_completion_status
        ):
            raise ValueError("clinical_completion_status must name an ADSL column")
        self._coerce("clinical_discontinued_days_after_last_dose", int, lo=0)
        for name in ("clinical_first_dose", "clinical_last_dose"):
            if not getattr(self, name):
                raise ValueError(f"{name} must name at least one ADSL column")
        for name in (
            "clinical_window_stop_events",
            "clinical_first_dose",
            "clinical_last_dose",
        ):
            cols = getattr(self, name)
            dupes = sorted({c for c in cols if cols.count(c) > 1})
            if dupes:
                raise ValueError(f"{name} has duplicate columns: {', '.join(dupes)}")
        self._check_identity()
        self._check_pqtl()
        chroms = tuple(str(c) for c in self.pqtl_expected_chromosomes)
        if not chroms:
            raise ValueError("pqtl_expected_chromosomes must name at least one chromosome")
        object.__setattr__(self, "pqtl_expected_chromosomes", chroms)
        if self.pqtl_enabled:
            missing = [k for k in PQTL_REQUIRED_KEYS if not getattr(self, k)]
            if self.pqtl_kinship_source == "table" and not self.pqtl_kinship_table:
                missing.append("pqtl_kinship_table (or kinship_source = 'catalog')")
            if (
                self.pqtl_null_by_ancestry == "pcs"
                and self.pqtl_pcs_source == "projected"
                and not self.pqtl_ancestry_pcs
            ):
                missing.append("pqtl_ancestry_pcs (or pcs_source = 'catalog')")
            if missing:
                raise ValueError(f"pqtl_enabled requires {', '.join(missing)}")

    def _check_olink_value_column(self):
        """Checks ``[olink] value_column`` and warns when it is not the default column.

        Raises:
            ValueError: If the value is not ``"PCNormalizedNPX"`` or ``"NPX"``, or a
                SomaScan config sets it.
        """
        qc_platform(self.platform, self.olink_value_column)
        if self.olink_value_column != OLINK_PC_NPX:
            logger.warning(
                "[olink] value_column = %r: the identity thresholds (best_min, "
                "margin_min, self_max, the pQTL floors) and the sex and age model "
                "settings keep their %s defaults; the sample-matrix check stays on %s",
                self.olink_value_column,
                OLINK_PC_NPX,
                OLINK_PC_NPX,
            )

    @property
    def qc_platform(self) -> Platform:
        """The platform with the value column QC reads (``[olink] value_column``)."""
        return qc_platform(self.platform, self.olink_value_column)

    def _check_low_volume_match(self):
        """Checks that ``qc_low_volume_match`` is None or distinct, non-blank texts.

        A list is stored as a tuple; texts that differ only in case raise.
        """
        match = self.qc_low_volume_match
        if match is None:
            return
        if (
            isinstance(match, str)
            or not isinstance(match, (list, tuple))
            or not match
            or not all(isinstance(t, str) and t.strip() for t in match)
        ):
            raise ValueError(
                f"qc_low_volume_match must be a non-empty list of non-blank texts; got {match!r}"
            )
        lowered = [t.lower() for t in match]
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"qc_low_volume_match has texts that differ only in case: {match!r}")
        object.__setattr__(self, "qc_low_volume_match", tuple(match))

    def _coerce(self, name: str, kind: type, *, lo=None, hi=None, lo_open=False, hi_open=False):
        """Checks that a field is a finite number in range and stores it as ``kind``.

        Bools never count as numbers. A float field takes an int and stores it as a
        float.

        Args:
            name: The field to check.
            kind: ``int`` or ``float``; ``int`` refuses a float value.
            lo: Lower bound, or None for none.
            hi: Upper bound, or None for none.
            lo_open: Whether the lower bound is excluded.
            hi_open: Whether the upper bound is excluded.

        Raises:
            ValueError: If the value is not a number of that kind, not finite, or out of
                range.
        """
        value = getattr(self, name)
        ok = isinstance(value, int) if kind is int else isinstance(value, (int, float))
        if isinstance(value, bool) or not ok:
            raise ValueError(
                f"{name} must be {'an integer' if kind is int else 'a number'}; got {value!r}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value}")
        low = lo is not None and (value <= lo if lo_open else value < lo)
        high = hi is not None and (value >= hi if hi_open else value > hi)
        if low or high:
            left = "" if lo is None else f"{'(' if lo_open else '['}{lo}"
            right = "" if hi is None else f"{hi}{')' if hi_open else ']'}"
            span = (
                f"in {left}, {right}"
                if lo is not None and hi is not None
                else (
                    f"{'>' if lo_open else '>='} {lo}"
                    if lo is not None
                    else f"{'<' if hi_open else '<='} {hi}"
                )
            )
            raise ValueError(f"{name} must be {span}; got {value!r}")
        object.__setattr__(self, name, kind(value))

    def _check_bool(self, *names: str) -> None:
        """Checks that each field in ``names`` is a bool.

        A TOML string such as ``"false"`` is truthy in Python, so it must never stand in
        for false.

        Raises:
            ValueError: If a field holds anything but True or False.
        """
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false; got {value!r}")

    def _check_identity(self):
        """Checks the identity keys and the Lp scoring keys."""
        self._check_bool("identity_enabled", "identity_sex_veto", "identity_age_veto")
        if self.identity_age_veto_years is not None:
            self._coerce("identity_age_veto_years", float, lo=0.0)
        if self.identity_tiebreak not in TIEBREAKS:
            raise ValueError(
                f"identity_tiebreak must be one of {', '.join(TIEBREAKS)}; got {self.identity_tiebreak!r}"
            )
        self._coerce("identity_n_pcs", int, lo=0)
        self._coerce("identity_icc_min", float, lo=0.0, hi=1.0, hi_open=True)
        self._coerce("identity_icc_cap", float, lo=0.0, hi=1.0, lo_open=True, hi_open=True)
        self._coerce("identity_clip", float, lo=0.0)
        self._coerce("identity_min_group", int, lo=1)
        self._coerce("identity_median_rounds", int, lo=1)
        self._coerce("identity_n_folds", int, lo=2)
        self._coerce("identity_sex_prob", float, lo=0.5, hi=1.0, hi_open=True)
        self._coerce("identity_age_discordant_mads", float, lo=0.0, lo_open=True)
        self._coerce("identity_age_fit_mads", float, lo=0.0, lo_open=True)
        if self.identity_age_veto_residual not in AGE_VETO_RESIDUALS:
            raise ValueError(
                f"identity_age_veto_residual must be one of {', '.join(AGE_VETO_RESIDUALS)}; "
                f"got {self.identity_age_veto_residual!r}"
            )
        self._coerce("sex_warn_prob", float, lo=0.5, hi=1.0, hi_open=True)
        days = self.identity_reassign_date_tolerance_days
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError(
                f"identity_reassign_date_tolerance_days must be an integer >= 0; got {days!r}"
            )
        if not isinstance(self.identity_pqtl_veto, bool):
            raise ValueError(
                f"identity_pqtl_veto must be true or false; got {self.identity_pqtl_veto!r}"
            )
        if not isinstance(self.identity_veto_unplaceable, bool):
            raise ValueError(
                f"identity_veto_unplaceable must be true or false; got {self.identity_veto_unplaceable!r}"
            )
        if self.identity_veto_unplaceable_best_min == UNSET_VETO_UNPLACEABLE_BEST_MIN:
            # The TOML spelling of unset (TOML has no null): the floor follows best_min.
            object.__setattr__(self, "identity_veto_unplaceable_best_min", None)
        if self.identity_veto_unplaceable_best_min is not None:
            self._coerce("identity_veto_unplaceable_best_min", float)
        if not isinstance(self.identity_split_tiebreak, bool):
            raise ValueError(
                f"identity_split_tiebreak must be true or false; got {self.identity_split_tiebreak!r}"
            )
        if self.identity_split_tiebreak and self.identity_tiebreak != "pqtl+sex_age":
            raise ValueError(
                "identity_split_tiebreak decides by sex and age, so it needs identity_tiebreak = 'pqtl+sex_age'; "
                f"got identity_tiebreak {self.identity_tiebreak!r}"
            )
        if not isinstance(self.identity_sex_from_anchors, bool):
            raise ValueError(
                f"identity_sex_from_anchors must be true or false; got {self.identity_sex_from_anchors!r}"
            )

    def _check_expected_sex(self):
        """Checks that no subject has conflicting ``expected_sex`` corrections.

        A subject takes at most one ``expected_sex`` exception, and none when a
        ``subject_set`` changes its ``SEX``.
        """
        expected = [c.usubjid_sha256 for c in self.corrections if c.type == "expected_sex"]
        repeated = {d for d in expected if expected.count(d) > 1}
        if repeated:
            raise ValueError(f"{len(repeated)} subjects have more than one expected_sex correction")
        sex_set = {
            c.usubjid_sha256
            for c in self.corrections
            if c.type == "subject_set" and "SEX" in c.set_values
        }
        both = sex_set & set(expected)
        if both:
            raise ValueError(
                f"{len(both)} subjects have both an expected_sex correction and a subject_set that changes SEX; "
                "keep one"
            )

    def _check_pqtl(self):
        """Checks the pQTL keys and stores numbers as floats and lists as tuples."""
        self._check_bool("pqtl_enabled")
        self._coerce("pqtl_margin_min", float, lo=0.0)
        cat = self.pqtl_reference_catalog
        if isinstance(cat, (list, tuple)):
            cat = tuple(cat)
            if not cat:
                raise ValueError("pqtl_reference_catalog is an empty list")
            if not all(isinstance(c, str) and c.strip() for c in cat):
                raise ValueError(
                    f"pqtl_reference_catalog entries must be non-empty strings; got {list(cat)}"
                )
            dupes = sorted({c for c in cat if cat.count(c) > 1})
            if dupes:
                raise ValueError(
                    f"pqtl_reference_catalog lists a catalog twice: {', '.join(dupes)}"
                )
            object.__setattr__(self, "pqtl_reference_catalog", cat)
        elif cat is not None and not (isinstance(cat, str) and cat.strip()):
            raise ValueError(
                f"pqtl_reference_catalog must be a string or a list of strings; got {cat!r}"
            )
        for name in (
            "pqtl_weight_exponent",
            "pqtl_best_min",
            "pqtl_match_floor",
            "pqtl_mismatch_floor",
            "pqtl_relative_kinship",
            "pqtl_duplicate_kinship",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number; got {value!r}")
            object.__setattr__(self, name, float(value))
        if not math.isfinite(self.pqtl_weight_exponent) or self.pqtl_weight_exponent < 0:
            raise ValueError(
                f"pqtl_weight_exponent must be finite and >= 0; got {self.pqtl_weight_exponent}"
            )
        self._coerce("pqtl_best_min", float, lo=0.0)
        self._coerce("pqtl_maf_min", float, lo=0.0, hi=0.5)
        self._coerce("pqtl_contribution_cap", float, lo=0.0)
        self._coerce("pqtl_nlp_recompute_floor", float, lo=0.0)
        for name in ("pqtl_match_floor", "pqtl_mismatch_floor"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite; got {getattr(self, name)}")
        if self.pqtl_mismatch_floor > self.pqtl_match_floor:
            raise ValueError(
                f"pqtl_mismatch_floor {self.pqtl_mismatch_floor} must not exceed "
                f"pqtl_match_floor {self.pqtl_match_floor}"
            )
        for name in ("pqtl_relative_kinship", "pqtl_duplicate_kinship"):
            value = getattr(self, name)
            if not 0 < value <= 0.5:
                raise ValueError(f"{name} must be in (0, 0.5]; got {value}")
        for name in (
            "pqtl_sex_x_het_one",
            "pqtl_sex_x_het_two",
            "pqtl_sex_y_absent",
            "pqtl_sex_y_present",
        ):
            self._coerce(name, float, lo=0.0, hi=1.0)
        if self.pqtl_sex_x_het_one >= self.pqtl_sex_x_het_two:
            raise ValueError(
                f"pqtl_sex_x_het_one {self.pqtl_sex_x_het_one} must be below "
                f"pqtl_sex_x_het_two {self.pqtl_sex_x_het_two}"
            )
        if self.pqtl_sex_y_absent >= self.pqtl_sex_y_present:
            raise ValueError(
                f"pqtl_sex_y_absent {self.pqtl_sex_y_absent} must be below "
                f"pqtl_sex_y_present {self.pqtl_sex_y_present}"
            )
        for name, allowed in (
            ("pqtl_kinship_source", KINSHIP_SOURCES),
            ("pqtl_pcs_source", PCS_SOURCES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"{name} must be one of {', '.join(allowed)}; got {getattr(self, name)!r}"
                )
        if self.pqtl_kinship_source == "table":
            # The table omits pairs below its floor, so a lower cut cannot be read.
            for name in ("pqtl_relative_kinship", "pqtl_duplicate_kinship"):
                if getattr(self, name) < KING_UNRELATED:
                    raise ValueError(
                        f"{name} {getattr(self, name)} is below the KING table's floor {KING_UNRELATED}: "
                        "the table does not list those pairs (kinship_source = 'catalog' computes every pair)"
                    )
        if self.pqtl_relative_kinship >= self.pqtl_duplicate_kinship:
            raise ValueError(
                f"pqtl_relative_kinship {self.pqtl_relative_kinship} must be below "
                f"pqtl_duplicate_kinship {self.pqtl_duplicate_kinship}"
            )
        if not isinstance(self.pqtl_ungenotyped_all_tubes, bool):
            raise ValueError(
                f"pqtl_ungenotyped_all_tubes must be true or false; got {self.pqtl_ungenotyped_all_tubes!r}"
            )
        if self.pqtl_relative_margin is not None:
            self._coerce("pqtl_relative_margin", float, lo=0.0)
        if self.pqtl_ungenotyped_naming not in UNGENOTYPED_NAMINGS:
            raise ValueError(
                f"pqtl_ungenotyped_naming must be one of {', '.join(UNGENOTYPED_NAMINGS)}; "
                f"got {self.pqtl_ungenotyped_naming!r}"
            )
        if self.pqtl_ungenotyped_naming != "all_tubes" and not self.pqtl_ungenotyped_all_tubes:
            raise ValueError(
                "pqtl_ungenotyped_naming applies only with pqtl_ungenotyped_all_tubes on; "
                "ungenotyped_all_tubes = false names every tube on its own"
            )
        if self.pqtl_null_by_ancestry not in NULL_BY_ANCESTRY:
            raise ValueError(
                f"pqtl_null_by_ancestry must be one of {', '.join(NULL_BY_ANCESTRY)}; got {self.pqtl_null_by_ancestry!r}"
            )
        self._coerce("pqtl_null_pcs", int, lo=1)
        if self.pqtl_null_scope not in NULL_SCOPES:
            raise ValueError(
                f"pqtl_null_scope must be one of {', '.join(NULL_SCOPES)}; got {self.pqtl_null_scope!r}"
            )
        if self.pqtl_ungenotyped_naming == "two_agree" and self.pqtl_null_by_ancestry == "off":
            raise ValueError(
                "pqtl_ungenotyped_naming = 'two_agree' needs pqtl_null_by_ancestry = 'pcs'"
            )
        if self.pqtl_relative_name_margin is not None:
            self._coerce("pqtl_relative_name_margin", float, lo=0.0)
        if self.pqtl_mixture_warn is not None:
            self._coerce("pqtl_mixture_warn", float)
            if self.pqtl_contribution_cap > 0:
                # The mixture score reads plain correlations, not the capped ones S
                # uses, so the WARN's threshold is fixed only for the uncapped score.
                raise ValueError(
                    "pqtl_mixture_warn is refused with pqtl_contribution_cap > 0 (the mixture score is uncapped)"
                )
        if self.pqtl_palindromic not in PALINDROMIC_MODES:
            raise ValueError(
                f"pqtl_palindromic must be one of {', '.join(PALINDROMIC_MODES)}; got {self.pqtl_palindromic!r}"
            )
        if self.pqtl_dosage_scaling not in PQTL_DOSAGE_SCALINGS:
            raise ValueError(
                f"pqtl_dosage_scaling must be one of {', '.join(PQTL_DOSAGE_SCALINGS)}; "
                f"got {self.pqtl_dosage_scaling!r}"
            )
        if not isinstance(self.pqtl_include_non_eur, bool):
            raise ValueError(
                f"pqtl_include_non_eur must be true or false; got {self.pqtl_include_non_eur!r}"
            )
        if self.pqtl_catalog_merge not in CATALOG_MERGES:
            raise ValueError(
                f"pqtl_catalog_merge must be one of {', '.join(CATALOG_MERGES)}; got {self.pqtl_catalog_merge!r}"
            )
        if self.pqtl_catalog_build is not None and self.pqtl_catalog_build not in PQTL_BUILDS:
            raise ValueError(
                f"pqtl_catalog_build must be one of {', '.join(PQTL_BUILDS)}; got {self.pqtl_catalog_build!r}"
            )
        if self.pqtl_catalog_platform is not None:
            get_platform(self.pqtl_catalog_platform)

    def sex_call_thresholds(self) -> SexCallThresholds:
        """Returns the genetic-sex call's four cuts, from the ``pqtl_sex_*`` keys."""
        return SexCallThresholds(
            x_het_one=self.pqtl_sex_x_het_one,
            x_het_two=self.pqtl_sex_x_het_two,
            y_absent=self.pqtl_sex_y_absent,
            y_present=self.pqtl_sex_y_present,
        )

    @property
    def advs_path(self) -> str | None:
        """The trial's ADVS path, or None when the config names none.

        ``[manifest] advs`` when the config has ``[manifest]``, else the top-level
        ``advs``.
        """
        return self.manifest.advs if self.manifest is not None else self.advs

    @property
    def best_min(self) -> float:
        """Lp best-subject z threshold: ``identity_best_min``, else the platform's."""
        if self.identity_best_min is not None:
            return self.identity_best_min
        return float(get_platform(self.platform).identity_defaults.best_min)

    @property
    def margin_min(self) -> float:
        """Lp best-minus-second margin: ``identity_margin_min``, else the platform's."""
        if self.identity_margin_min is not None:
            return self.identity_margin_min
        return float(get_platform(self.platform).identity_defaults.margin_min)

    @property
    def veto_unplaceable_best_min(self) -> float:
        """Blocked-match z a vetoed tube needs to FAIL; ``best_min`` when unset."""
        if self.identity_veto_unplaceable_best_min is not None:
            return self.identity_veto_unplaceable_best_min
        return self.best_min

    def _assay_default(self, key: str, name: str) -> float | None:
        value = getattr(self, key)
        if value is not None:
            return float(value)
        default = getattr(get_platform(self.platform).assay_defaults, name)
        return None if default is None else float(default)

    @property
    def intra_cv(self) -> float:
        """Intra-plate CV threshold (%): ``qc_intra_cv``, else the platform's."""
        return self._assay_default("qc_intra_cv", "intra_cv")

    @property
    def detection_rate(self) -> float:
        """Detection sample threshold: ``qc_detection_rate``, else the platform's."""
        return self._assay_default("qc_detection_rate", "detection_rate")

    @property
    def saturation_rfu(self) -> float | None:
        """Saturation RFU ceiling (SomaScan); None on Olink unless set."""
        return self._assay_default("qc_saturation_rfu", "saturation_rfu")

    @property
    def saturation_frac(self) -> float | None:
        """Sample fraction above the ceiling that flags an analyte (SomaScan)."""
        return self._assay_default("qc_saturation_frac", "saturation_frac")

    @property
    def sb_ratio(self) -> float | None:
        """Minimum signal-to-background ratio (SomaScan)."""
        return self._assay_default("qc_sb_ratio", "sb_ratio")

    @property
    def age_veto_years(self) -> float | None:
        """Age-veto tolerance in years, or None when the age veto is off."""
        return self.identity_age_veto_years if self.identity_age_veto else None

    def _check_visits(self):
        """Coerces the [visits] codes to floats and checks that they agree."""
        for name in ("visits_analysis", "visits_screening"):
            codes = tuple(float(v) for v in getattr(self, name))
            if len(set(codes)) != len(codes):
                raise ValueError(f"{name} has duplicate visit codes: {list(codes)}")
            object.__setattr__(self, name, codes)
        if self.visits_baseline is None:
            if self.visits_analysis:
                raise ValueError("visits_analysis is set without visits_baseline")
            return
        object.__setattr__(self, "visits_baseline", float(self.visits_baseline))
        if self.visits_baseline not in self.visits_analysis:
            raise ValueError(
                f"visits_baseline {self.visits_baseline} is not in visits_analysis "
                f"{list(self.visits_analysis)}"
            )
        overlap = sorted(set(self.visits_screening) & set(self.visits_analysis))
        if overlap:
            raise ValueError(f"visits_screening and visits_analysis share codes: {overlap}")


def _resolve_path(key: str, value: str, base: Path) -> str:
    """Resolves a relative path against ``base``.

    A ``pqtl_reference_catalog`` value that is not a ``.parquet`` path is a catalog name
    and stays as given.
    """
    if key == "pqtl_reference_catalog" and not value.endswith(".parquet"):
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else base / p)


def _strict_init(cls, values: dict, where: str):
    """Builds dataclass ``cls`` from ``values``, naming unknown or missing keys.

    Raises:
        ValueError: If a key is unknown or a required one is missing; the message names
            the table ``where``.
    """
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {', '.join(unknown)}")
    missing = [
        f.name
        for f in fields(cls)
        if f.name not in values and f.default is MISSING and f.default_factory is MISSING
    ]
    if missing:
        raise ValueError(f"{where} lacks {', '.join(missing)}")
    return cls(**values)


def _parse_manifest(raw: dict, base: Path) -> ManifestConfig:
    """Parses [manifest] and [manifest.columns]; file paths resolve against ``base``."""
    raw = dict(raw)
    if "columns" not in raw:
        raise ValueError("[manifest] lacks [manifest.columns]")
    columns = _strict_init(ManifestColumns, raw.pop("columns"), "[manifest.columns]")
    paths = {
        k: _resolve_path(k, v, base) if k in MANIFEST_FILE_FIELDS else v for k, v in raw.items()
    }
    return _strict_init(ManifestConfig, {**paths, "columns": columns}, "[manifest]")


def _parse_corrections(entries) -> tuple[Correction, ...]:
    """Parses [[corrections]] entries; the TOML ``set`` table becomes ``set_values``."""
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError("corrections must be [[corrections]] tables, not a path or value")
    out = []
    for entry in entries:
        entry = dict(entry)
        if "set" in entry:
            entry["set_values"] = entry.pop("set")
        if "type" not in entry:
            raise ValueError(f"correction entry has no type: keys {', '.join(sorted(entry))}")
        out.append(_strict_init(Correction, entry, f"{entry['type']} correction"))
    return tuple(out)


def load_config(
    path: str | None,
    *,
    platform: str | None = None,
    study_name: str | None = None,
) -> TrialConfig:
    """Loads a TOML trial config; a None path gives all defaults.

    Relative paths in the TOML resolve against the TOML file's directory.

    Args:
        path: The TOML file, or None.
        platform: The driver's ``--platform`` value; fills ``platform`` when the TOML leaves
            it unset and must agree with it otherwise.
        study_name: The driver's ``--study-name`` value; fills ``study_name`` when the TOML
            leaves it unset.

    Returns:
        The validated ``TrialConfig``.

    Raises:
        ValueError: If a key is unknown, set both in its table and at the top
            level, or the platform disagrees with ``platform``.
    """
    if path is None:
        return config_from_mapping({}, None, platform=platform, study_name=study_name)
    toml_path = Path(path).absolute()
    return config_from_mapping(
        tomllib.loads(toml_path.read_text()),
        toml_path.parent,
        platform=platform,
        study_name=study_name,
    )


def config_from_mapping(
    raw: dict,
    base: Path | None,
    *,
    platform: str | None = None,
    study_name: str | None = None,
) -> TrialConfig:
    """Builds a trial config from a parsed TOML mapping, as ``load_config`` does.

    Args:
        raw: The TOML's tables and keys; not modified.
        base: Directory relative paths resolve against (the TOML's); None when ``raw``
            is empty.
        platform: Fills ``platform`` when ``raw`` leaves it unset and must agree with it
            otherwise.
        study_name: Fills ``study_name`` when ``raw`` leaves it unset.

    Returns:
        The validated ``TrialConfig``.

    Raises:
        ValueError: If a key is unknown, set both in its table and at the top
            level, or the platform disagrees with ``platform``.
    """
    flat: dict = {}
    if raw:
        raw = dict(raw)
        # Flatten nested [identity], [pqtl], [clinical], [visits], [qc] and [olink]
        # tables; a key may not be set both in its table and at the top level
        # (qc_intra_cv and [qc] intra_cv).
        for table in ("identity", "pqtl", "clinical", "visits", "qc", "olink"):
            nested = {f"{table}_{k}": v for k, v in raw.pop(table, {}).items()}
            twice = sorted(set(nested) & set(raw))
            if twice:
                raise ValueError(f"{', '.join(twice)} set both in [{table}] and at the top level")
            flat.update(nested)
        if "manifest" in raw:
            flat["manifest"] = _parse_manifest(raw.pop("manifest"), base)
        if "corrections" in raw:
            flat["corrections"] = _parse_corrections(raw.pop("corrections"))
        flat.update(raw)
        valid = {f.name for f in fields(TrialConfig)}
        unknown = set(flat) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        for key in PATH_KEYS:
            if flat.get(key):
                value = flat[key]
                flat[key] = (
                    [_resolve_path(key, v, base) for v in value]
                    if isinstance(value, list)
                    else _resolve_path(key, value, base)
                )

    if platform is not None:
        if "platform" in flat and get_platform(flat["platform"]) is not get_platform(platform):
            raise ValueError(
                f"Config platform {flat['platform']!r} does not match --platform {platform!r}"
            )
        flat.setdefault("platform", platform)
    if study_name and not flat.get("study_name"):
        flat["study_name"] = study_name
    return TrialConfig(**flat)
