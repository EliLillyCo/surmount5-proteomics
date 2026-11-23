"""Identity check: Lp scoring and the subject reassignment rule.

Classifies tubes into the event types of ``schemas.IDENTITY_EVENT_TYPES``, with
thresholds from ``TrialConfig``. Only ``reciprocal`` and ``one_way`` events are
reassigned; every other event is flagged, and ``EVENT_FLAGS`` gives the sample flag each
sets. A subject is a target only through its anchor samples (fit, with
``self_z >= self_max`` or a positive pQTL match), and a tube moves only onto the
inferred subject's own ADaM visit record on its draw date.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from analysis.qc.clinical import (
    CORRECTION_EXCLUDE_FLAG,
    FAIL_IDENTITY_FOREIGN,
    FAIL_IDENTITY_UNRESOLVED,
    IDENTITY_FAIL_FLAGS,
    IDENTITY_WARN_FLAGS,
    WARN_IDENTITY_AMBIGUOUS,
    WARN_IDENTITY_PQTL,
    WARN_IDENTITY_VETOED,
)
from analysis.qc.corrections import EXPECTED_SEX_COL
from analysis.qc.flags import (
    FAIL_MEDIAN_IQR_OUTLIER,
    FAIL_PCA_OUTLIER,
    FAIL_ROWCHECK,
    FAIL_SAMPLE_QC,
)
from analysis.qc.modeling import duplicate_assay_ids, subject_folds
from analysis.qc.schemas import (
    DNA_LINK_SCHEMA,
    IDENTITY_EVENT_TYPES,
    IDENTITY_EVENTS_SCHEMA,
    PQTL_SCORES_SCHEMA,
    UNFIT_IDENTITY_SCHEMA,
    UNGENOTYPED_NAMINGS,
    pqtl_named_donor,
)
from analysis.qc.sex_concordance import SEX_PROB_THRESHOLD
from analysis.qc.utils import ensure_date_column, wide_matrix

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

N_PCS: int = 20  # principal components projected out of residuals
CLIP: float = 6.0  # bound on robust-scaled residuals
MIN_GROUP: int = 20  # smallest arm x visit group estimated on its own
N_MEDIAN_ROUNDS: int = 3  # alternating arm x visit / plate median rounds
N_FOLDS: int = 5  # participant folds for cross-fitted protein selection
ICC_MIN: float = 0.2  # training-fold intraclass correlation threshold
ICC_CAP: float = 0.95  # ICC above this adds no weight: w = icc / (1 - min(icc, cap))
AGE_DISCORDANT_MADS: float = 2.0  # |age residual| beyond this many MADs is discordant
AGE_FIT_MADS: float = 1.0  # residual against the partner within this many MADs fits
TIEBREAKS: tuple[str, ...] = ("pqtl", "pqtl+sex_age")
# The split tiebreak decides by sex and age, so the Rule and the config refuse it with
# any other tiebreak.
SPLIT_TIEBREAK_NEEDS: str = "pqtl+sex_age"
# Which age residual the age veto compares with the best subject's expected age: the raw
# predicted age, or the bias-corrected residual.
AGE_VETO_RESIDUALS: tuple[str, ...] = ("raw", "corrected")

# Sample flags that make a sample unusable as an Lp reference ("unfit").
TECHNICAL_FAIL_FLAGS: frozenset[str] = frozenset(
    {
        FAIL_SAMPLE_QC,
        FAIL_PCA_OUTLIER,
        FAIL_MEDIAN_IQR_OUTLIER,
        FAIL_ROWCHECK,
    }
)
# Unfit samples: the technical failures plus the tube an ``exclude`` correction
# removes from every analysis. Proposals and anchors are fit samples only, so an unfit
# sample is never a reference, never proposed and never moved.
UNFIT_FLAGS: frozenset[str] = TECHNICAL_FAIL_FLAGS | {CORRECTION_EXCLUDE_FLAG}

# ── The unfit-tube identity annotation ───────────────────────────────────────
# Each technical FAIL flag's short name in the annotation's failure_kind, in this order.
UNFIT_FAILURE_KINDS: dict[str, str] = {
    FAIL_MEDIAN_IQR_OUTLIER: "median_iqr",
    FAIL_PCA_OUTLIER: "pca",
    FAIL_ROWCHECK: "rowcheck",
    FAIL_SAMPLE_QC: "sample_qc",
}
assert set(UNFIT_FAILURE_KINDS) == TECHNICAL_FAIL_FLAGS
# The flags whose outlier statistics (PCk_std; Z_Median and Z_IQR) give a tube's
# distance over its cut.
_OUTLIER_STATISTICS: dict[str, str] = {
    FAIL_PCA_OUTLIER: r"^PC\d+_std$",
    FAIL_MEDIAN_IQR_OUTLIER: r"^Z_(Median|IQR)$",
}
# Per platform, the flag combinations whose failed tubes' identity scores are treated
# as unreliable (score_reliable False).
UNRELIABLE_FAILURES: dict[str, tuple[frozenset[str], ...]] = {
    "olink": (
        frozenset({FAIL_MEDIAN_IQR_OUTLIER, FAIL_PCA_OUTLIER}),
        frozenset({FAIL_SAMPLE_QC}),
    ),
    "soma": (frozenset({FAIL_SAMPLE_QC}),),
}
# A tube more than this many z over its outlier cut is unfit.
UNFIT_MAX_Z_OVER_CUT: float = 5.0
# Strong-match floor of the annotation.
UNFIT_BEST_MIN: float = 8.0
# Technically failed tubes have no sex or age prediction (the models skip them) and
# are given none.
NO_SEX_AGE_EVIDENCE: str = "no sex or age evidence"


# ── Lp scoring settings ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LpParams:
    """Lp scoring settings, the ``[identity]`` keys ``n_pcs`` to ``n_folds``.

    The module constants are only these fields' defaults. ``clip`` of 0 turns off the
    clipping of robust-scaled residuals.
    """

    n_pcs: int = N_PCS
    icc_min: float = ICC_MIN
    icc_cap: float = ICC_CAP
    clip: float = CLIP
    min_group: int = MIN_GROUP
    median_rounds: int = N_MEDIAN_ROUNDS
    n_folds: int = N_FOLDS

    @classmethod
    def from_config(cls, config) -> LpParams:
        """Builds the settings from a ``TrialConfig``."""
        return cls(
            n_pcs=config.identity_n_pcs,
            icc_min=config.identity_icc_min,
            icc_cap=config.identity_icc_cap,
            clip=config.identity_clip,
            min_group=config.identity_min_group,
            median_rounds=config.identity_median_rounds,
            n_folds=config.identity_n_folds,
        )


DEFAULT_LP = LpParams()


# ── Rule ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    """Thresholds and switches of the identity reassignment rule."""

    self_max: float = 3.5
    best_min: float = 8.0
    margin_min: float = 1.0
    sex_veto: bool = True
    age_veto_years: float | None = 10.0
    tiebreak: str = "pqtl"
    # A tube moves only onto a destination record this many days from its ADT.
    date_tolerance_days: int = 1
    # A positive pQTL match vetoes a move and an identity FAIL.
    pqtl_veto: bool = True
    # The sex evidence cut (veto, tiebreak, DNA link), the age tiebreak's MAD multiples
    # and the age veto's residual. Defaults are the module constants.
    sex_prob: float = SEX_PROB_THRESHOLD
    age_discordant_mads: float = AGE_DISCORDANT_MADS
    age_fit_mads: float = AGE_FIT_MADS
    age_veto_residual: str = "raw"
    # A tube blocked by a sex or age veto that matches no subject of its own and has no
    # positive pQTL self-match is unplaceable (vetoed_destination, FAIL) when its
    # blocked match is at least veto_unplaceable_best_min (None: best_min). Off, it
    # stays WARN.
    veto_unplaceable: bool = False
    veto_unplaceable_best_min: float | None = None
    # The sex veto and the two-tube tiebreak read a subject without an expected_sex
    # exception as its own anchor tubes' protein sex, when they are all confidently one
    # sex (expected_sex_adsl). Off, they read ADSL SEX.
    sex_from_anchors: bool = False
    # Under "two_agree", an ungenotyped label's Lp split (flagged tubes naming nobody,
    # two or more Lp-clean tubes all pQTL mismatches naming one donor) FAILs the
    # Lp-clean tubes as majority_foreign and spares the flagged ones.
    ungenotyped_naming: str = "all_tubes"
    # The two-tube tiebreak for subjects of three or more fit tubes whose flagged and
    # Lp-clean tubes form two groups: each group's confident protein sex against the
    # expected sex, then age (_split_events). It needs tiebreak "pqtl+sex_age"; with
    # "pqtl" the Rule refuses it rather than leave it silently inert.
    split_tiebreak: bool = False
    # A fit tube with no other event whose pqtl_mixture_z reaches this is a pqtl_mixture
    # WARN. None: off.
    mixture_warn: float | None = None

    def __post_init__(self) -> None:
        """Checks the pQTL keys and the split tiebreak's needs."""
        if self.mixture_warn is not None and (
            isinstance(self.mixture_warn, bool)
            or not isinstance(self.mixture_warn, (int, float))
            or not np.isfinite(self.mixture_warn)
        ):
            raise ValueError(
                f"mixture_warn must be a finite number or None, got {self.mixture_warn!r}"
            )
        if not isinstance(self.split_tiebreak, bool):
            raise ValueError(f"split_tiebreak must be true or false, got {self.split_tiebreak!r}")
        if self.split_tiebreak and self.tiebreak != SPLIT_TIEBREAK_NEEDS:
            raise ValueError(
                f"identity split_tiebreak decides by sex and age, so it needs identity tiebreak = {SPLIT_TIEBREAK_NEEDS!r}; "
                f"got tiebreak {self.tiebreak!r}"
            )
        if self.ungenotyped_naming not in UNGENOTYPED_NAMINGS:
            raise ValueError(
                f"ungenotyped_naming must be one of {UNGENOTYPED_NAMINGS}, got {self.ungenotyped_naming!r}"
            )
        if self.tiebreak not in TIEBREAKS:
            raise ValueError(f"identity tiebreak must be one of {TIEBREAKS}, got {self.tiebreak!r}")
        if self.age_veto_residual not in AGE_VETO_RESIDUALS:
            raise ValueError(
                f"age_veto_residual must be one of {AGE_VETO_RESIDUALS}, got {self.age_veto_residual!r}"
            )


# ── Scoring helpers ──────────────────────────────────────────────────────────


def group_median(X: np.ndarray, codes: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Returns per-group column medians of the ``fit`` rows, broadcast to n x p."""
    E = np.zeros_like(X)
    for g in np.unique(codes):
        rows = codes == g
        E[rows] = np.nanmedian(X[rows & fit], axis=0)
    return E


def arm_visit_effect(
    X: np.ndarray,
    arm: np.ndarray,
    visit: np.ndarray,
    fit: np.ndarray,
    params: LpParams = DEFAULT_LP,
) -> np.ndarray:
    """Returns arm x visit column medians of the ``fit`` rows as an n x p matrix.

    Groups with fewer than ``params.min_group`` fit rows fall back to the arm median
    over all visits, or the global median when the arm itself is small.
    """
    E = np.empty_like(X)
    for a in np.unique(arm):
        in_arm = arm == a
        pool = in_arm & fit if (in_arm & fit).sum() >= params.min_group else fit
        fallback = np.nanmedian(X[pool], axis=0)
        for v in np.unique(visit[in_arm]):
            rows = in_arm & (visit == v)
            E[rows] = (
                np.nanmedian(X[rows & fit], axis=0)
                if (rows & fit).sum() >= params.min_group
                else fallback
            )
    return E


def residualize(
    X: np.ndarray,
    arm: np.ndarray,
    visit: np.ndarray,
    plates: np.ndarray,
    fit: np.ndarray,
    params: LpParams = DEFAULT_LP,
) -> tuple[np.ndarray, np.ndarray]:
    """Removes arm x visit and plate medians, then scales by MAD, clips and zero-fills.

    Returns:
        ``(Z, keep)``: the scaled residuals of the columns with a finite, positive MAD,
        and that column mask.
    """
    plate_eff = np.zeros_like(X)
    for _ in range(params.median_rounds):
        grp_eff = arm_visit_effect(X - plate_eff, arm, visit, fit, params)
        plate_eff = group_median(X - grp_eff, plates, fit)
    R = X - grp_eff - plate_eff
    med = np.nanmedian(R[fit], axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(R[fit] - med), axis=0)
    keep = np.isfinite(mad) & (mad > 0)
    Z = (R[:, keep] - med[keep]) / mad[keep]
    if params.clip > 0:
        Z = np.clip(Z, -params.clip, params.clip)
    return np.nan_to_num(Z, nan=0.0), keep


def remove_top_pcs(Z: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    """Projects the top ``k`` principal components of the fit rows out of every row."""
    Zf = Z[fit] - Z[fit].mean(axis=0)
    _, _, Vt = np.linalg.svd(Zf, full_matrices=False)
    V = Vt[:k].T
    return Z - (Z @ V) @ V.T


def participant_folds(usubjids: list[str], n_folds: int, seed: int = 0) -> dict[str, int]:
    """Maps each participant to a fold by a stable hash of its USUBJID.

    See ``modeling.subject_folds``.
    """
    return subject_folds(usubjids, n_folds, seed, salt="identity")


# ── Similarity functions ─────────────────────────────────────────────────────


def rowstd(X: np.ndarray) -> np.ndarray:
    """Centers and scales each row."""
    X = X - X.mean(axis=1, keepdims=True)
    return X / X.std(axis=1, keepdims=True)


def icc(Z: np.ndarray, usubjid: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Returns per-protein intraclass correlation from within-participant pairs."""
    X = Z[rows] - Z[rows].mean(axis=0)
    u = usubjid[rows]
    num = np.zeros(Z.shape[1])
    den = np.zeros(Z.shape[1])
    for p in np.unique(u):
        idx = np.where(u == p)[0]
        if len(idx) < 2:
            continue
        s = X[idx].sum(axis=0)
        q = (X[idx] ** 2).sum(axis=0)
        num += (s**2 - q) / 2
        den += (len(idx) - 1) * q / 2
    return num / den


def within_platform_similarity(
    Z: np.ndarray,
    usubjid: np.ndarray,
    fold: np.ndarray,
    train: np.ndarray,
    params: LpParams = DEFAULT_LP,
) -> np.ndarray:
    """Returns the cross-fitted, ICC-weighted correlation of every sample pair."""
    C = np.zeros((Z.shape[0], Z.shape[0]))
    for f in np.unique(fold):
        rows = np.where(train & (fold != f))[0]
        ic = icc(Z, usubjid, rows)
        sel = ic > params.icc_min
        logger.info("fold %d: %d within-platform proteins", f, int(sel.sum()))
        w = ic[sel] / (1 - np.minimum(ic[sel], params.icc_cap))
        X = (Z[:, sel] - Z[rows][:, sel].mean(axis=0)) * np.sqrt(w)
        X = rowstd(X)
        q = fold == f
        C[q] = X[q] @ X.T / sel.sum()
    return C


def robust_z(values: np.ndarray, null_mask: np.ndarray) -> np.ndarray:
    """Returns ``(values - median) / (1.4826 * MAD)`` over ``values[null_mask]``."""
    null = values[null_mask]
    med = np.median(null)
    return (values - med) / (1.4826 * np.median(np.abs(null - med)))


def row_robust_z(C: np.ndarray, u_query: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Returns the row-wise robust z of ``C`` against other participants' columns."""
    out = np.empty_like(C)
    for i in range(C.shape[0]):
        out[i] = robust_z(C[i], u_ref != u_query[i])
    return out


# ── Matrix building ──────────────────────────────────────────────────────────


def build_identity_matrix(
    qc_instance,
) -> tuple[np.ndarray, pl.DataFrame, list[str]]:
    """Builds the n x p sample-by-protein matrix for identity scoring.

    Study samples only, on the platform's identity layer.
    Duplicate-assay copies (``FAIL_Duplicate_Assay``) are left out, as in outlier
    detection; every other assay enters. A sample's values across plates are averaged, and it
    gets its lowest plate. Samples with no manifest row or a null
    USUBJID are dropped (the count is logged).

    Returns:
        ``(X, samples_df, protein_ids)``, rows sorted by sample ID: row i of ``X`` is
        row i of ``samples_df`` (the ``adsl_merged`` labels plus the plate), and
        ``protein_ids`` labels the columns of ``X`` in lexicographic order.

    Raises:
        ValueError: If a sample ID has more than one manifest row.
    """
    sid = qc_instance.sample_id_col
    pid = qc_instance.protein_id_col
    dcol = qc_instance.platform.identity_layer
    plate_col = qc_instance.plate_id_col

    bridge_ids = qc_instance.bridge_sample_ids

    df = qc_instance.samples.select([sid, pid, dcol])
    if bridge_ids:
        df = df.filter(~pl.col(sid).cast(pl.Utf8).is_in(list(bridge_ids)))

    # A duplicate copy would count its protein twice in every sample correlation. The
    # flags come from filter_assay_duplicates, which runs before identity.
    duplicates = duplicate_assay_ids(qc_instance)
    if duplicates:
        df = df.filter(~pl.col(pid).cast(pl.Utf8).is_in(duplicates))
    logger.info("identity matrix: %d duplicate-assay copies excluded", len(duplicates))

    # Average across plates.
    df = df.group_by([sid, pid]).agg(pl.col(dcol).mean())
    ids, protein_cols, X_all = wide_matrix(df.collect(), index=sid, on=pid, values=dcol, sort=True)
    wide_ids = ids.to_frame()

    labels = qc_instance.adsl_merged.collect()
    if plate_col in labels.columns:
        labels = labels.drop(plate_col)
    samples_df = wide_ids.join(labels, on=sid, how="inner", maintain_order="left")
    duplicated = samples_df.filter(pl.col(sid).is_duplicated())[sid].unique().sort()
    if duplicated.len() > 0:
        raise ValueError(
            "identity check: more than one manifest row for SampleID "
            + ", ".join(str(s) for s in duplicated.to_list())
        )
    samples_df = samples_df.filter(pl.col("USUBJID").is_not_null())
    n_dropped = wide_ids.height - samples_df.height
    if n_dropped:
        logger.info(
            "identity check: %d samples excluded (no manifest row or null USUBJID)",
            n_dropped,
        )

    # Lowest plate over the measurement rows, so the label does not depend on the
    # input row order.
    plates = qc_instance.samples.group_by(sid).agg(pl.col(plate_col).min()).collect()
    samples_df = samples_df.join(plates, on=sid, how="left", maintain_order="left").sort(sid)
    # The matrix rows in the order of the labeled samples.
    row_of = (
        samples_df.select(sid)
        .join(wide_ids.with_row_index("_row"), on=sid, how="left", maintain_order="left")["_row"]
        .to_numpy()
    )
    X = np.asfortranarray(X_all[row_of])  # the layout the pivot's to_numpy gave
    assert samples_df.height == X.shape[0]
    return X, samples_df, protein_cols


@dataclass(frozen=True, eq=False)
class IdentityMatrix:
    """A QC instance's identity matrix, built once for the pQTL and identity stages.

    ``X``, ``samples`` and ``protein_ids`` are ``build_identity_matrix``'s;
    ``residuals`` is the cache both stages residualize through; ``source`` records what
    the matrix was built from (``_matrix_source``).
    """

    X: np.ndarray
    samples: pl.DataFrame
    protein_ids: list[str]
    residuals: ResidualCache
    source: tuple


def _matrix_source(qc_instance) -> tuple:
    """Returns what ``build_identity_matrix`` reads that a later step could change."""
    return (
        qc_instance.samples,
        qc_instance.adsl_merged,
        qc_instance.sample_metadata,
        tuple(duplicate_assay_ids(qc_instance)),
        frozenset(qc_instance.bridge_sample_ids),
    )


def identity_matrix(qc_instance) -> IdentityMatrix:
    """Builds the identity matrix once, for both stages that need it.

    The driver parks the result on ``qc.shared_identity_matrix`` before the pQTL stage and
    clears it after the identity stage; ``matrix_for`` hands it to ``pqtl_inputs`` and
    ``identity_inputs``, which would otherwise each build it.
    """
    X, samples_df, protein_ids = build_identity_matrix(qc_instance)
    return IdentityMatrix(X, samples_df, protein_ids, ResidualCache(X), _matrix_source(qc_instance))


def matrix_for(
    qc_instance,
) -> tuple[np.ndarray, pl.DataFrame, list[str], ResidualCache | None]:
    """Returns ``(X, samples, protein_ids, residuals)`` for a stage.

    Uses the shared matrix on ``qc.shared_identity_matrix`` when the driver parked one,
    else a fresh ``build_identity_matrix`` with no residual cache.

    Raises:
        ValueError: If the shared matrix was built from other data than ``qc_instance``
            holds now.
    """
    shared = getattr(qc_instance, "shared_identity_matrix", None)
    if shared is None:
        X, samples_df, protein_ids = build_identity_matrix(qc_instance)
        return X, samples_df, protein_ids, None
    now = _matrix_source(qc_instance)
    if not (
        all(a is b for a, b in zip(shared.source[:3], now[:3], strict=True))
        and shared.source[3:] == now[3:]
    ):
        raise ValueError("the shared identity matrix was built from data that has since changed")
    return shared.X, shared.samples, shared.protein_ids, shared.residuals


# ── Scoring wrappers ─────────────────────────────────────────────────────────


class ResidualCache:
    """A cache of one identity matrix's residuals, per arm, plate column and params.

    The pQTL stage (``pqtl_reference.pqtl_matrix``) and the Lp stage (``lp_matrix``) of
    one run residualize the same matrix with the same settings. The stored ``(Z, keep)``
    are read-only, so a stage that wrote into the shared arrays would fail loudly.

    Args:
        X: The matrix the cache is for (``identity_matrix``).
    """

    def __init__(self, X: np.ndarray):
        """Starts an empty cache for ``X``."""
        self._X = X
        self._entries: dict[tuple, tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]] = {}

    def residualize(
        self,
        X: np.ndarray,
        samples: pl.DataFrame,
        adsl: pl.DataFrame,
        arm_col: str,
        plate_col: str,
        params: LpParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``residualize_samples``'s ``(Z, keep)``, computed once per settings.

        Of ``adsl`` only ``USUBJID`` and the arm column are compared with the first
        call's: residualization reads nothing else, so another column (such as the
        expected-sex exception the identity stage's ADSL may carry) is no difference.

        Raises:
            ValueError: If ``X`` is not the cache's matrix, or ``samples`` or ``adsl``
                differ from the first call's at the same settings.
        """
        if X is not self._X:
            raise ValueError(
                "residual cache: asked for the residuals of a matrix other than the one it was made for"
            )
        key = (arm_col, plate_col, params)
        entry = self._entries.get(key)
        if entry is None:
            Z, keep = _residualize_samples(X, samples, adsl, arm_col, plate_col, params)
            Z.setflags(write=False)
            keep.setflags(write=False)
            self._entries[key] = (samples, adsl.select("USUBJID", arm_col), Z, keep)
            return Z, keep
        cached_samples, cached_adsl, Z, keep = entry
        if not (
            samples.equals(cached_samples) and adsl.select("USUBJID", arm_col).equals(cached_adsl)
        ):
            raise ValueError(
                "residual cache: the samples or ADSL differ from those the cached residuals were computed on"
            )
        return Z, keep


def _residualize_samples(
    X: np.ndarray,
    samples: pl.DataFrame,
    adsl: pl.DataFrame,
    arm_col: str,
    plate_col: str,
    params: LpParams,
) -> tuple[np.ndarray, np.ndarray]:
    labeled = samples.join(
        adsl.select("USUBJID", pl.col(arm_col).alias("_arm")),
        on="USUBJID",
        how="left",
        maintain_order="left",
        validate="m:1",
    )
    assert labeled.height == X.shape[0]
    return residualize(
        X,
        labeled["_arm"].fill_null("NA").to_numpy(),
        labeled["VISITNUM"].cast(pl.String).fill_null("NA").to_numpy(),
        labeled[plate_col].to_numpy(),
        labeled["fit"].to_numpy(),
        params,
    )


def residualize_samples(
    X: np.ndarray,
    samples: pl.DataFrame,
    adsl: pl.DataFrame,
    arm_col: str = "TRT01A",
    plate_col: str = "PlateID",
    params: LpParams = DEFAULT_LP,
    cache: ResidualCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs ``residualize`` with arm, visit, plate and fit taken from ``samples``.

    ``samples`` rows follow ``X`` and carry ``USUBJID``, ``VISITNUM``, ``plate_col`` and
    a boolean ``fit`` column; the arm comes from ``adsl``. With a ``cache`` the
    residuals are computed once and shared.
    """
    if cache is not None:
        return cache.residualize(X, samples, adsl, arm_col, plate_col, params)
    return _residualize_samples(X, samples, adsl, arm_col, plate_col, params)


def lp_matrix(
    X: np.ndarray,
    samples: pl.DataFrame,
    adsl: pl.DataFrame,
    seed: int = 0,
    arm_col: str = "TRT01A",
    plate_col: str = "PlateID",
    params: LpParams = DEFAULT_LP,
    residuals: ResidualCache | None = None,
) -> np.ndarray:
    """Returns the robust-z within-platform similarity of every sample pair.

    Each row is a robust z against the other participants' samples; the diagonal is NaN.

    Args:
        X: The n x p identity matrix.
        samples: Rows of ``X`` with ``USUBJID``, ``VISITNUM``, the plate and ``fit``.
        adsl: ADSL with ``USUBJID`` and the arm column.
        seed: Seed of the participant folds.
        arm_col: Treatment arm column in ``adsl``.
        plate_col: Plate column in ``samples``.
        params: Scoring settings; the defaults are the module constants.
        residuals: Residuals shared with the pQTL stage (``identity_matrix``).
    """
    fit = samples["fit"].to_numpy()
    Z, _ = residualize_samples(
        X,
        samples,
        adsl,
        arm_col=arm_col,
        plate_col=plate_col,
        params=params,
        cache=residuals,
    )
    Z = remove_top_pcs(Z, fit, params.n_pcs)
    u = samples["USUBJID"].to_numpy()
    folds = participant_folds(list(u), params.n_folds, seed=seed)
    f = np.array([folds[x] for x in u])
    Wz = row_robust_z(within_platform_similarity(Z, u, f, fit, params), u, u)
    np.fill_diagonal(Wz, np.nan)
    return Wz


def subject_scores(
    Wz: np.ndarray,
    usubjid: np.ndarray,
    fit: np.ndarray,
    anchor: np.ndarray | None = None,
) -> pl.DataFrame:
    """Scores each sample against its labeled subject and the two best other subjects.

    A subject's score for a sample is its best z over that subject's fit samples. With
    ``anchor`` (a boolean mask over samples), the two best other subjects are scored on
    their anchor samples only, so a subject with no anchor is never a best or second
    subject; ``self_z`` still uses every fit sample of the labeled subject.
    ``best_subject`` and ``second_subject`` are null when no other subject has a usable
    sample.
    """
    n = len(usubjid)
    subjects, codes = np.unique(usubjid, return_inverse=True)
    W = np.where(np.isnan(Wz), -np.inf, Wz)
    np.fill_diagonal(W, -np.inf)
    W[:, ~fit] = -np.inf
    order = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(codes[order]) != 0])
    per_subject = np.maximum.reduceat(W[:, order], starts, axis=1)
    rows = np.arange(n)
    own = per_subject[rows, codes].copy()
    if anchor is not None:
        W[:, ~anchor] = -np.inf
        per_subject = np.maximum.reduceat(W[:, order], starts, axis=1)
    per_subject[rows, codes] = -np.inf
    top = np.argsort(-per_subject, axis=1, kind="stable")[:, :2]
    best = per_subject[rows, top[:, 0]]
    second = per_subject[rows, top[:, 1]]

    def _z(v):
        return pl.Series(np.where(np.isfinite(v), v, np.nan)).fill_nan(None)

    def _subject(v, k):
        names = subjects[top[:, k]].astype(str).tolist()
        return pl.Series(
            [n if ok else None for n, ok in zip(names, np.isfinite(v), strict=True)],
            dtype=pl.String,
        )

    return pl.DataFrame(
        {
            "sample_idx": np.arange(n, dtype=np.int64),
            "self_z": _z(own),
            "best_subject": _subject(best, 0),
            "best_z": _z(best),
            "second_subject": _subject(second, 1),
            "second_z": _z(second),
        }
    )


# ── Proposal logic ───────────────────────────────────────────────────────────

_EVENT_ROW_SCHEMA: dict[str, pl.DataType] = {
    "SampleID": pl.String,
    "USUBJID_INFERRED": pl.String,
    "VISITNUM_INFERRED": pl.Float64,
    "event_type": pl.String,
    "action": pl.String,
    "veto_reason": pl.String,
    "unplaced_reason": pl.String,
}
# What propose_scored returns: the rule's part of the public events frame.
_PROPOSE_SCHEMA: dict[str, pl.DataType] = _EVENT_ROW_SCHEMA | {
    "resolved_by": pl.String,
    "date_gap_days": pl.Int64,
}

REASSIGNED_EVENT_TYPES: tuple[str, ...] = ("reciprocal", "one_way")

# Bias-corrected age residual added to the age OOF predictions by the age
# module; when absent, the raw residual against the labeled age is used.
AGE_RESIDUAL_CORRECTED_COL: str = "Age_Residual_Corrected"


def _cast_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Selects ``schema``'s columns from ``df`` in order, cast to its types.

    Visit codes are Float64.
    """
    return df.select([pl.col(n).cast(t) for n, t in schema.items()])


def _first_draw(scored: pl.DataFrame) -> pl.DataFrame:
    """Returns each labeled subject's earliest draw date (``_first_ADT``)."""
    return scored.group_by("USUBJID").agg(pl.col("ADT").min().alias("_first_ADT"))


def _expected_age(age: str, first_adt: str) -> pl.Expr:
    """Returns the expected age at ``ADT``: ``age`` at ``first_adt`` plus time since."""
    return pl.col(age) + (pl.col("ADT") - pl.col(first_adt)).dt.total_days() / 365.0


def _strong(scored: pl.DataFrame, rule: Rule) -> pl.DataFrame:
    """Returns the fit, flagged samples that meet the best-z and margin thresholds.

    A null VISITNUM does not exclude a sample: its visit comes from the destination's
    visit records (``analysis.qc.placement.place``).
    """
    return scored.filter(
        pl.col("fit")
        & pl.col("self_z").is_not_null()
        & pl.col("best_subject").is_not_null()
        & (pl.col("self_z") < rule.self_max)
        & (pl.col("best_z") >= rule.best_min)
        & ((pl.col("best_z") - pl.col("second_z").fill_null(float("-inf"))) >= rule.margin_min)
    )


def _candidates(
    pool: pl.DataFrame,
    scored: pl.DataFrame,
    adsl: pl.DataFrame,
    rule: Rule,
    p_male_col: str | None,
    age_col: str | None,
) -> pl.DataFrame:
    """Adds ``veto_reason`` to ``pool`` (strong proposals): the first veto that fires.

    In order: ``pqtl`` when ``rule.pqtl_veto`` is on and the tube's ``pqtl_call`` is
    ``match`` (its genotype matches its label); ``sex`` when the predicted sex
    contradicts the best subject's ``SEX``; ``age`` when the predicted age contradicts
    the best subject's expected age at the draw. Null when none fires. ``pqtl_call``
    ``none`` and null evidence (a sample outside the sex or age model, or a subject with
    a null SEX/AGE) never veto; the proposal then rests on Lp alone. ADSL must have the
    columns (``propose_scored`` checks).
    """
    pool = pool.with_columns(ensure_date_column(pool, "ADT"))
    ev_cols = [c for c in (p_male_col, age_col) if c is not None and c in pool.columns]
    pool = pool.with_columns([pl.col(c).fill_nan(None) for c in ev_cols])
    helper_cols: list[str] = []
    reason = pl.when(pl.lit(rule.pqtl_veto) & (pl.col("_call") == "match")).then(pl.lit("pqtl"))

    if rule.sex_veto and p_male_col is not None:
        pool = pool.join(
            adsl.select(
                pl.col("USUBJID").alias("best_subject"),
                pl.col("SEX").alias("_best_SEX"),
            ),
            on="best_subject",
            how="left",
            maintain_order="left",
        )
        helper_cols.append("_best_SEX")
        sex_contradicts = (
            (pl.col("_best_SEX") == "M") & (pl.col(p_male_col) < 1 - rule.sex_prob)
        ) | ((pl.col("_best_SEX") == "F") & (pl.col(p_male_col) > rule.sex_prob))
        reason = reason.when(sex_contradicts.fill_null(False)).then(pl.lit("sex"))

    if rule.age_veto_years is not None and age_col is not None:
        scored = scored.with_columns(ensure_date_column(scored, "ADT"))
        pool = pool.join(
            adsl.select(
                pl.col("USUBJID").alias("best_subject"),
                pl.col("AGE").alias("_best_AGE"),
            ),
            on="best_subject",
            how="left",
            maintain_order="left",
        ).join(
            _first_draw(scored).rename({"USUBJID": "best_subject"}),
            on="best_subject",
            how="left",
            maintain_order="left",
        )
        helper_cols += ["_best_AGE", "_first_ADT"]
        if rule.age_veto_residual == "corrected":
            if AGE_RESIDUAL_CORRECTED_COL not in pool.columns:
                raise ValueError(
                    f"age_veto_residual is 'corrected' but the age predictions carry no "
                    f"{AGE_RESIDUAL_CORRECTED_COL}"
                )
            # The corrected residual is against the label's expected age; move it to the
            # best subject's, as the tiebreak does.
            pool = pool.join(
                adsl.select(pl.col("USUBJID"), pl.col("AGE").alias("_lab_AGE")),
                on="USUBJID",
                how="left",
                maintain_order="left",
            ).join(
                _first_draw(scored).rename({"_first_ADT": "_lab_first_ADT"}),
                on="USUBJID",
                how="left",
                maintain_order="left",
            )
            helper_cols += ["_lab_AGE", "_lab_first_ADT"]
            residual = (
                pl.col(AGE_RESIDUAL_CORRECTED_COL).fill_nan(None)
                + _expected_age("_lab_AGE", "_lab_first_ADT")
                - _expected_age("_best_AGE", "_first_ADT")
            )
        else:
            residual = pl.col(age_col) - _expected_age("_best_AGE", "_first_ADT")
        age_contradicts = residual.abs() > rule.age_veto_years
        reason = reason.when(age_contradicts.fill_null(False)).then(pl.lit("age"))

    return pool.with_columns(
        reason.otherwise(pl.lit(None, dtype=pl.String)).alias("veto_reason")
    ).drop(helper_cols)


def _two_sample_subjects(
    scored: pl.DataFrame, flagged: pl.DataFrame, strong: pl.DataFrame
) -> pl.DataFrame:
    """Finds the two-sample subjects whose flagged tubes all point at one partner.

    A labeled subject qualifies when it has exactly two fit samples, one or more of
    them flagged, and every flagged one has the same best other subject, the partner,
    whether or not the partner points back. Lp cannot tell which of such a subject's two
    tubes moved: one cross-matches the partner because it holds the partner's blood, the
    other because the partner holds a tube of this subject's. The subject or the partner
    must have a strong proposal toward the other.

    Returns:
        ``SampleID``, ``USUBJID``, ``VISITNUM``, ``partner`` for both fit tubes of each
        qualifying subject.
    """
    flagged_ids = flagged["SampleID"].to_list()
    strong_pairs = strong.select(
        pl.col("USUBJID").alias("_from"), pl.col("best_subject").alias("_to")
    ).unique()
    fit = scored.filter(pl.col("fit")).with_columns(
        pl.col("SampleID").is_in(flagged_ids).alias("_flag")
    )
    pointed = pl.col("best_subject").filter(pl.col("_flag"))
    subj = (
        fit.group_by("USUBJID")
        .agg(
            pl.len().alias("_n"),
            pl.col("_flag").any().alias("_any_flagged"),
            pointed.n_unique().alias("_n_partners"),
            pointed.null_count().alias("_n_null"),
            pointed.first().alias("partner"),
        )
        .filter(
            (pl.col("_n") == 2)
            & pl.col("_any_flagged")
            & (pl.col("_n_partners") == 1)
            & (pl.col("_n_null") == 0)
        )
    )
    outgoing = subj.join(
        strong_pairs,
        left_on=["USUBJID", "partner"],
        right_on=["_from", "_to"],
        how="semi",
    )
    incoming = subj.join(
        strong_pairs,
        left_on=["partner", "USUBJID"],
        right_on=["_from", "_to"],
        how="semi",
    )
    keep = pl.concat([outgoing, incoming]).select("USUBJID", "partner").unique()
    return (
        fit.join(keep, on="USUBJID", how="inner")
        .select("SampleID", "USUBJID", "VISITNUM", "partner")
        .sort("SampleID")
    )


def _age_residuals(
    scored: pl.DataFrame, adsl: pl.DataFrame, age_col: str
) -> tuple[pl.DataFrame, float]:
    """Returns each sample's age residual and the residual MAD over the fit samples.

    The residual is ``Age_Residual_Corrected`` when the age module provides it, else the
    raw predicted-minus-expected age.

    Returns:
        ``(frame, mad)``: ``SampleID``, ``_exp_lab`` (expected age under the label) and
        ``_resid``; and the residual MAD (scaled by 1.4826) over all fit samples, 0 when
        there are none.
    """
    first = _first_draw(scored)
    age = adsl.select("USUBJID", pl.col("AGE").alias("_AGE"))
    corrected = AGE_RESIDUAL_CORRECTED_COL in scored.columns
    res = (
        scored.join(age, on="USUBJID", how="left", maintain_order="left")
        .join(first, on="USUBJID", how="left", maintain_order="left")
        .with_columns(_expected_age("_AGE", "_first_ADT").alias("_exp_lab"))
        .with_columns(
            (
                pl.col(AGE_RESIDUAL_CORRECTED_COL)
                if corrected
                else pl.col(age_col) - pl.col("_exp_lab")
            )
            .fill_nan(None)
            .alias("_resid")
        )
    )
    r = res.filter(pl.col("fit") & pl.col("_resid").is_not_null())["_resid"]
    mad = 1.4826 * float((r - r.median()).abs().median()) if r.len() > 0 else 0.0
    logger.info(
        "identity tiebreak: age residual MAD %.2f years (%s residual)",
        mad,
        "bias-corrected" if corrected else "raw",
    )
    return res.select("SampleID", "_exp_lab", "_resid"), mad


def _anchor_sex(scored: pl.DataFrame, rule: Rule, p_male_col: str) -> pl.DataFrame:
    """Returns the protein sex of each subject's own anchor tubes (``_anchor_SEX``).

    A subject is listed only when its anchor tubes with a prediction are all confidently
    one sex (beyond ``rule.sex_prob``). Its anchor tubes are its fit tubes whose
    ``_call`` is ``match`` (its own genome) when it has any, else its fit tubes with
    ``self_z >= rule.self_max``. Tubes resolved by the two-tube tiebreak are not among
    them, since the tiebreak reads this.
    """
    fit = pl.col("fit")
    match = fit & (pl.col("_call") == "match")
    lp = fit & (pl.col("self_z") >= rule.self_max).fill_null(False)
    anchor = pl.when(match.any().over("USUBJID")).then(match).otherwise(lp)
    p = pl.col(p_male_col).fill_nan(None)
    return (
        scored.filter(anchor & pl.col("USUBJID").is_not_null())
        .filter(p.is_not_null())
        .group_by("USUBJID")
        .agg(
            pl.when((p > rule.sex_prob).all())
            .then(pl.lit("M"))
            .when((p < 1 - rule.sex_prob).all())
            .then(pl.lit("F"))
            .alias("_anchor_SEX")
        )
        .filter(pl.col("_anchor_SEX").is_not_null())
        .sort("USUBJID")
    )


def expected_sex_adsl(
    adsl: pl.DataFrame, scored: pl.DataFrame, rule: Rule, p_male_col: str | None
) -> pl.DataFrame:
    """Returns ``adsl`` with ``SEX`` set to the expected protein sex.

    The sex veto, the two-tube tiebreak and ``_sex_off`` compare a tube with this sex,
    taken in order from:

    1. the subject's ``expected_sex`` exception (column ``EXPECTED_SEX_COL``, present
       only when the config has one; ``identity_inputs`` adds it);
    2. with ``rule.sex_from_anchors``, its own anchor tubes' protein sex
       (``_anchor_sex``);
    3. else ADSL ``SEX`` as recorded.

    The exception column is dropped. With no exception column and ``sex_from_anchors``
    off (or no sex predictions), ``adsl`` is returned unchanged. ``scored`` needs
    ``_call`` (``_with_pqtl_call``).
    """
    has_exception = EXPECTED_SEX_COL in adsl.columns
    anchors = rule.sex_from_anchors and p_male_col is not None
    if not has_exception and not anchors:
        return adsl
    if "SEX" not in adsl.columns:
        return adsl.drop(EXPECTED_SEX_COL, strict=False)
    expected = [pl.col(EXPECTED_SEX_COL)] if has_exception else []
    if anchors:
        adsl = adsl.join(
            _anchor_sex(scored, rule, p_male_col),
            on="USUBJID",
            how="left",
            maintain_order="left",
        )
        expected.append(pl.col("_anchor_SEX"))
    out = adsl.with_columns(
        pl.coalesce(*expected, pl.col("SEX").cast(pl.String)).cast(adsl.schema["SEX"]).alias("SEX")
    )
    changed = out["SEX"].cast(pl.String).ne_missing(adsl["SEX"].cast(pl.String))
    logger.info(
        "identity sex evidence: %d subjects read as an expected protein sex other than ADSL SEX "
        "(expected_sex exceptions %s; anchor tubes %s)",
        out.filter(changed)["USUBJID"].n_unique(),
        "present" if has_exception else "none",
        "on" if anchors else "off",
    )
    return out.drop(EXPECTED_SEX_COL, "_anchor_SEX", strict=False)


def _sex_off(p_male_col: str, label_sex: str, sex_prob: float) -> pl.Expr:
    """Returns True where the predicted sex is confidently opposite ``label_sex``.

    Confident means beyond ``sex_prob``; an unknown sex on either side is False.
    """
    p = pl.col(p_male_col).fill_nan(None)
    return (
        ((p > sex_prob) & (pl.col(label_sex) == "F"))
        | ((p < 1 - sex_prob) & (pl.col(label_sex) == "M"))
    ).fill_null(False)


def _pair_evidence(
    group: pl.DataFrame,
    scored: pl.DataFrame,
    adsl: pl.DataFrame,
    rule: Rule,
    p_male_col: str | None,
    age_col: str | None,
) -> pl.DataFrame:
    """Adds per-sample move evidence to the samples in ``group``.

    Boolean columns:

    - ``_pqtl_hit``: ``pqtl_call`` is ``mismatch`` and the named donor is the partner
      (``_call`` and ``_pdonor`` come with ``scored``).
    - ``_sex_hit``: predicted sex confidently contradicts the labeled subject's
      ``SEX`` and matches the partner's.
    - ``_age_out``: the absolute age residual exceeds ``rule.age_discordant_mads`` MADs;
      ``_age_fit``: the residual against the partner's expected age is within
      ``rule.age_fit_mads`` MADs (``_age_residuals``). With no usable residuals (none,
      or MAD 0) age resolves nothing.

    Sex and age columns are filled only when ``rule.tiebreak`` is ``"pqtl+sex_age"``;
    otherwise they are False.
    """
    ev = group.join(
        scored.drop("USUBJID", "VISITNUM"),
        on="SampleID",
        how="left",
        maintain_order="left",
    ).with_columns(
        ((pl.col("_call") == "mismatch") & (pl.col("_pdonor") == pl.col("partner")))
        .fill_null(False)
        .alias("_pqtl_hit")
    )
    false = pl.lit(False)

    sex_age = rule.tiebreak == "pqtl+sex_age"
    sex_hit = false
    if sex_age and p_male_col is not None and "SEX" in adsl.columns:
        sex = adsl.select("USUBJID", "SEX")
        ev = ev.join(
            sex.rename({"SEX": "_lab_SEX"}),
            on="USUBJID",
            how="left",
            maintain_order="left",
        ).join(
            sex.rename({"USUBJID": "partner", "SEX": "_par_SEX"}),
            on="partner",
            how="left",
            maintain_order="left",
        )
        sex_hit = _sex_off(p_male_col, "_lab_SEX", rule.sex_prob) & (
            pl.col("_par_SEX") != pl.col("_lab_SEX")
        ).fill_null(False)
    ev = ev.with_columns(sex_hit.alias("_sex_hit"))

    age_out, age_fit = false, false
    if sex_age and age_col is not None and "AGE" in adsl.columns:
        res, mad = _age_residuals(scored, adsl, age_col)
        first = _first_draw(scored)
        age = adsl.select("USUBJID", pl.col("AGE").alias("_AGE"))
        ev = (
            ev.join(res, on="SampleID", how="left", maintain_order="left")
            .join(
                age.rename({"USUBJID": "partner", "_AGE": "_par_AGE"}),
                on="partner",
                how="left",
                maintain_order="left",
            )
            .join(
                first.rename({"USUBJID": "partner", "_first_ADT": "_par_first_ADT"}),
                on="partner",
                how="left",
                maintain_order="left",
            )
            .with_columns(
                (
                    pl.col("_resid")
                    + pl.col("_exp_lab")
                    - _expected_age("_par_AGE", "_par_first_ADT")
                ).alias("_resid_par")
            )
        )
        if mad > 0:
            age_out = (pl.col("_resid").abs() > rule.age_discordant_mads * mad).fill_null(False)
            age_fit = (pl.col("_resid_par").abs() <= rule.age_fit_mads * mad).fill_null(False)
    return ev.with_columns(age_out.alias("_age_out"), age_fit.alias("_age_fit"))


def _tube_evidence(
    tubes: pl.DataFrame,
    scored: pl.DataFrame,
    adsl: pl.DataFrame,
    rule: Rule,
    p_male_col: str | None,
    age_col: str | None,
) -> pl.DataFrame:
    """Adds the two-tube tiebreak evidence to ``tubes``.

    Sex and age are read only when ``rule.tiebreak`` is ``"pqtl+sex_age"``; under
    ``"pqtl"`` every evidence column is False, so only pQTL decides the two-tube table.

    Args:
        tubes: Rows of ``scored`` for the fit tubes of two-tube subjects.
        scored: Every scored tube, for the age residuals and their MAD.
        adsl: ADSL with ``SEX`` and ``AGE`` (either may be absent).
        rule: The identity rule; ``tiebreak``, ``sex_prob`` and ``age_discordant_mads``
            are read.
        p_male_col: Column of predicted P(male), or None without a sex model.
        age_col: Column of the age residual source, or None without an age model.

    Returns:
        ``tubes`` with ``_sex_off`` (predicted sex confidently opposite the label's ADSL
        ``SEX`` at ``rule.sex_prob``) and ``_age_out`` / ``_age_in`` (the age residual
        beyond / within ``rule.age_discordant_mads`` MADs; both False when the residual
        is unknown or the MAD is 0).
    """
    false = pl.lit(False)
    sex_age = rule.tiebreak == "pqtl+sex_age"
    sex_off = false
    if sex_age and p_male_col is not None and "SEX" in adsl.columns:
        tubes = tubes.join(
            adsl.select("USUBJID", pl.col("SEX").alias("_lab_SEX")),
            on="USUBJID",
            how="left",
            maintain_order="left",
        )
        sex_off = _sex_off(p_male_col, "_lab_SEX", rule.sex_prob)
    age_out, age_in = false, false
    if sex_age and age_col is not None and "AGE" in adsl.columns:
        res, mad = _age_residuals(scored, adsl, age_col)
        tubes = tubes.join(
            res.select("SampleID", "_resid"),
            on="SampleID",
            how="left",
            maintain_order="left",
        )
        if mad > 0:
            beyond = pl.col("_resid").abs() > rule.age_discordant_mads * mad
            age_out = beyond.fill_null(False)
            age_in = (~beyond).fill_null(False)
    return tubes.with_columns(
        sex_off.alias("_sex_off"), age_out.alias("_age_out"), age_in.alias("_age_in")
    )


def _related(a: str | None, b: str | None, relatives: frozenset[tuple[str, str]]) -> bool:
    return (a, b) in relatives or (b, a) in relatives


def _pqtl_contradicts(
    samples: list[dict],
    pick: dict,
    donor: str | None,
    relatives: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    """Checks whether pQTL contradicts ``pick`` as the moved sample.

    True when pQTL says ``pick`` matches its label, that another of ``samples`` is the
    stray, or (with an expected ``donor``) that ``pick`` holds a credible third
    subject's blood: a named donor that is neither ``donor`` nor a KING relative of it.
    """
    third = (
        donor is not None
        and pick["_call"] == "mismatch"
        and pick["_pdonor"] is not None
        and pick["_pdonor"] != donor
        and not _related(pick["_pdonor"], donor, relatives)
    )
    return (
        pick["_call"] == "match"
        or third
        or any(s["_call"] == "mismatch" for s in samples if s is not pick)
    )


def _moved_sample(
    samples: list[dict],
    partner_samples: list[dict],
    relatives: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[dict | None, str]:
    """Returns the one sample a subject's own evidence says moved, and what said so.

    In order: pQTL (exactly one sample is a ``mismatch`` naming the partner, and either
    every other sample of the subject is a positive ``match`` or the partner's tube at
    that visit is a ``mismatch`` naming this subject); then sex; then age (exactly one
    sample age-discordant, and it fits the partner's age). A sex or age pick that pQTL
    contradicts (``_pqtl_contradicts``) is rejected. Returns ``(None, "none")`` when
    nothing names a sample.
    """
    subject, partner = samples[0]["USUBJID"], samples[0]["partner"]
    hits = [s for s in samples if s["_pqtl_hit"]]
    if len(hits) == 1:
        pick = hits[0]
        others_match = all(s["_call"] == "match" for s in samples if s is not pick)
        named_back = pick["VISITNUM"] is not None and any(
            t["VISITNUM"] == pick["VISITNUM"]
            and t["_call"] == "mismatch"
            and t["_pdonor"] == subject
            for t in partner_samples
        )
        if others_match or named_back:
            return pick, "pqtl"
    picks = []
    sex = [s for s in samples if s["_sex_hit"]]
    if len(sex) == 1:
        picks.append((sex[0], "sex"))
    out = [s for s in samples if s["_age_out"]]
    if len(out) == 1 and out[0]["_age_fit"]:
        picks.append((out[0], "age"))
    for pick, basis in picks:
        if not _pqtl_contradicts(samples, pick, partner, relatives):
            return pick, basis
    return None, "none"


def _resolve_two_sample_subjects(
    evidence: pl.DataFrame,
    partner_tubes: pl.DataFrame,
    relatives: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[str], set[str], dict[str, str]]:
    """Returns ``(moved, staying, basis)`` for the two-sample subjects in ``evidence``.

    Each subject is resolved on its own evidence (``_moved_sample``); nothing is
    inferred from a partner's missing evidence.

    - Mutual pair (the partner is a two-sample subject pointing back): both sides must
      resolve on their own evidence, to the same non-null visit.
    - Otherwise: the subject's moved tube implies a reciprocal exchange with the partner's
      tube at the same visit (the subject's other tube also cross-matches the partner,
      so a one-way move can't explain it). That partner tube must exist, be unique, and
      pQTL must not contradict it.

    A subject that does not resolve has neither a moved nor a staying sample; the
    two-tube table decides its tubes (``_two_tube_events``). ``basis`` maps each moved
    sample to the evidence that named it (``pqtl``, ``sex``, ``age``).
    ``partner_tubes`` has ``USUBJID``, ``SampleID``, ``VISITNUM``, ``_call`` and
    ``_pdonor`` for every sample of every partner.
    """
    by_subject: dict[str, list[dict]] = {}
    for row in evidence.sort("SampleID").to_dicts():
        by_subject.setdefault(row["USUBJID"], []).append(row)
    tubes: dict[str, list[dict]] = {}
    for row in partner_tubes.sort("SampleID").to_dicts():
        tubes.setdefault(row["USUBJID"], []).append(row)
    own = {
        subject: _moved_sample(samples, tubes.get(samples[0]["partner"], []), relatives)
        for subject, samples in by_subject.items()
    }

    moved: set[str] = set()
    staying: set[str] = set()
    basis: dict[str, str] = {}
    for subject in sorted(by_subject):
        samples = by_subject[subject]
        ids = {s["SampleID"] for s in samples}
        partner = samples[0]["partner"]
        mine, why = own[subject]
        theirs = by_subject.get(partner)
        if mine is None or mine["VISITNUM"] is None:
            continue
        if theirs is not None and theirs[0]["partner"] == subject:
            consistent = (
                own[partner][0] is not None and own[partner][0]["VISITNUM"] == mine["VISITNUM"]
            )
        else:
            partner_samples = tubes.get(partner, [])
            at_visit = [t for t in partner_samples if t["VISITNUM"] == mine["VISITNUM"]]
            consistent = len(at_visit) == 1 and not _pqtl_contradicts(
                partner_samples, at_visit[0], subject, relatives
            )
        if not consistent:
            continue
        moved.add(mine["SampleID"])
        basis[mine["SampleID"]] = why
        staying |= ids - {mine["SampleID"]}
    return moved, staying, basis


# Events that name another subject as the tube's source.
NAMED_DONOR_TYPES: tuple[str, ...] = (
    "reciprocal",
    "one_way",
    "duplicate_draw",
    "unplaceable",
    "multi_match",
)

# The sample flag each event type sets; None for a reassigned tube, which the clinical
# rules then judge under its new subject.
EVENT_FLAGS: dict[str, str | None] = {
    "reciprocal": None,
    "one_way": None,
    "duplicate_draw": FAIL_IDENTITY_FOREIGN,
    "unplaceable": FAIL_IDENTITY_FOREIGN,
    "multi_match": FAIL_IDENTITY_FOREIGN,
    "unattributable": FAIL_IDENTITY_FOREIGN,
    "two_tube_mismatch": FAIL_IDENTITY_FOREIGN,
    "majority_foreign": FAIL_IDENTITY_FOREIGN,
    "unresolved": FAIL_IDENTITY_UNRESOLVED,
    "vetoed": WARN_IDENTITY_VETOED,
    "ambiguous": WARN_IDENTITY_AMBIGUOUS,
    "pqtl_only": WARN_IDENTITY_PQTL,
    "pqtl_mixture": WARN_IDENTITY_PQTL,
}
assert tuple(EVENT_FLAGS) == IDENTITY_EVENT_TYPES


def identity_flags(events: pl.DataFrame) -> dict[str, list[str]]:
    """Maps every identity FAIL and WARN flag to the SampleIDs whose event sets it.

    The flag comes from the event type (``EVENT_FLAGS``). IDs are sorted; a flag no
    event sets has an empty list.

    Raises:
        ValueError: If an event type is not in ``EVENT_FLAGS``.
    """
    unknown = sorted(set(events["event_type"].to_list()) - set(EVENT_FLAGS))
    if unknown:
        raise ValueError(f"identity events of unknown type: {', '.join(unknown)}")
    flagged = events.with_columns(
        pl.col("event_type").replace_strict(EVENT_FLAGS, return_dtype=pl.String).alias("_flag")
    )
    return {
        flag: sorted(flagged.filter(pl.col("_flag") == flag)["SampleID"].to_list())
        for flag in (*IDENTITY_FAIL_FLAGS, *IDENTITY_WARN_FLAGS)
    }


def _two_tube_events(
    tubes: pl.DataFrame,
    event_of: dict[str, tuple[str, str | None]],
    kept: set[str],
    anchored: set[str],
    rule: Rule,
) -> tuple[list[dict], dict[str, str]]:
    """Returns events for the open tubes of two-fit-tube subjects with a flagged tube.

    ``tubes`` holds both fit tubes of each such subject (``SampleID``, ``USUBJID``,
    ``_call``, ``_pdonor``, ``_target``, ``_sex_off``, ``_age_out``, ``_age_in``);
    ``event_of`` maps each tube that already has an event to ``(event_type,
    veto_reason)``; ``kept`` are the staying tubes of reassigned pairs (no event). A
    tube is open when it is in neither. ``anchored`` holds the subjects with an anchor.

    One tube open, the other with an event:

    - the other is ``vetoed`` (pqtl), so a positive match: this tube is ``vetoed``
      (pqtl) when it also matches (both match, Lp disagrees), else ``two_tube_mismatch``
      (resolved by pqtl);
    - the other carries a named-donor event or another veto: this tube gets no event
      when it is a positive match, ``unresolved`` (no anchor) or ``unattributable`` when
      pQTL calls it a mismatch, else ``ambiguous`` (the other tube explains its flag).

    Both open, the tiebreak in order: both positive matches, both ``vetoed`` (pqtl); one
    positive match, the other ``two_tube_mismatch`` (pqtl); exactly one tube's predicted
    sex opposite its ADSL ``SEX`` (sex); one tube age-discordant and the other within
    (age); sex and age only under ``tiebreak = "pqtl+sex_age"`` (``_tube_evidence``
    leaves their columns False otherwise). A sex or age pick that pQTL contradicts is
    rejected. Nothing decides: both ``unresolved``.

    Returns:
        Event rows (``SampleID``, ``USUBJID_INFERRED``, ``event_type``,
        ``veto_reason``) and ``resolved_by`` for the ``two_tube_mismatch`` rows.
    """
    by_subject: dict[str, list[dict]] = {}
    for row in tubes.sort("SampleID").to_dicts():
        by_subject.setdefault(row["USUBJID"], []).append(row)
    rows: list[dict] = []
    basis: dict[str, str] = {}

    def add(t: dict, event_type: str, inferred: str | None, veto: str | None = None) -> None:
        rows.append(
            {
                "SampleID": t["SampleID"],
                "USUBJID_INFERRED": inferred,
                "event_type": event_type,
                "veto_reason": veto,
            }
        )

    for subject in sorted(by_subject):
        pair = by_subject[subject]
        assert len(pair) == 2, f"{subject}: two-tube table on {len(pair)} tubes"
        open_ = [t for t in pair if t["SampleID"] not in event_of and t["SampleID"] not in kept]
        if not open_:
            continue
        if len(open_) == 1:
            t = open_[0]
            o = pair[1] if t is pair[0] else pair[0]
            assert o["SampleID"] in event_of, f"{subject}: open tube beside a kept tube"
            o_type, o_veto = event_of[o["SampleID"]]
            if o_type == "vetoed" and o_veto == "pqtl":
                if t["_call"] == "match":
                    if rule.pqtl_veto:
                        add(t, "vetoed", t["_target"], "pqtl")
                else:
                    add(t, "two_tube_mismatch", t["_pdonor"])
                    basis[t["SampleID"]] = "pqtl"
            elif t["_call"] == "match":
                pass
            elif t["_call"] == "mismatch":
                add(
                    t,
                    "unattributable" if subject in anchored else "unresolved",
                    t["_target"],
                )
            else:
                add(t, "ambiguous", None)
            continue
        matches = [t for t in pair if t["_call"] == "match"]
        if len(matches) == 2 and rule.pqtl_veto:
            for t in pair:
                add(t, "vetoed", t["_target"], "pqtl")
            continue
        pick, why = None, "none"
        if len(matches) == 1:
            pick, why = (pair[1] if matches[0] is pair[0] else pair[0]), "pqtl"
        else:
            sex = [t for t in pair if t["_sex_off"]]
            out = [t for t in pair if t["_age_out"]]
            within = [t for t in pair if t["_age_in"]]
            for hits, name in ((sex, "sex"), (out if len(within) == 1 else [], "age")):
                if len(hits) == 1 and not _pqtl_contradicts(pair, hits[0], None):
                    pick, why = hits[0], name
                    break
        if pick is None:
            for t in pair:
                add(t, "unresolved", t["_target"])
        else:
            add(pick, "two_tube_mismatch", pick["_pdonor"])
            basis[pick["SampleID"]] = why
    return rows, basis


def _group_sex(group: list[dict], p_male_col: str | None, sex_prob: float) -> str | None:
    """Returns the sex every predicted tube of ``group`` confidently shares, or None.

    ``M`` or ``F`` when every tube with a sex prediction is beyond ``sex_prob`` that way
    and one or more has a prediction.
    """
    if p_male_col is None:
        return None
    p = [t[p_male_col] for t in group if t[p_male_col] is not None and not np.isnan(t[p_male_col])]
    if not p:
        return None
    if all(x > sex_prob for x in p):
        return "M"
    if all(x < 1 - sex_prob for x in p):
        return "F"
    return None


def _group_age(group: list[dict], cut: float | None) -> str | None:
    """Returns ``out`` or ``in`` when all of ``group`` is beyond or within ``cut``.

    None otherwise, when a residual is missing, or without a cut.
    """
    if cut is None:
        return None
    r = [t["_resid"] for t in group]
    if any(x is None or np.isnan(x) for x in r):
        return None
    if all(abs(x) > cut for x in r):
        return "out"
    if all(abs(x) <= cut for x in r):
        return "in"
    return None


def _split_events(
    scored: pl.DataFrame,
    leftover: pl.DataFrame,
    kept: set[str],
    rule: Rule,
    adsl_recorded: pl.DataFrame,
    p_male_col: str | None,
    age_col: str | None,
) -> tuple[pl.DataFrame, dict[str, str], set[str]]:
    """Returns events for the Lp splits of subjects with three or more fit tubes.

    A subject is split when its Lp-flagged fit tubes (the flagged group, one or more)
    are all in ``leftover`` (no event yet) and it has two or more Lp-clean fit tubes
    (the clean group). The clean group is foreign, its tubes ``majority_foreign`` naming
    ``_pdonor`` and the flagged tubes spared (no event), when:

    - ``rule.ungenotyped_naming == "two_agree"`` (``resolved_by`` pqtl): the label is
      ungenotyped (``pqtl_self_z`` null on every fit tube), every clean tube is a pQTL
      ``mismatch`` naming one and the same donor, and no flagged tube is a mismatch or
      names one;
    - else ``rule.split_tiebreak``: exactly one group's confident protein sex
      (``_group_sex`` at ``rule.sex_prob``) is the expected sex and the other's the
      opposite, the expected sex being the ``expected_sex`` exception, else ADSL ``SEX``
      (never the anchors, since the clean group is the subject's Lp anchors); then age
      (one group's residuals all beyond ``rule.age_discordant_mads`` MADs, the other's
      all within). A pick is rejected when the foreign group holds a positive ``match``
      or the other group a ``mismatch``. A pick of the flagged group changes nothing
      (its tubes go on to step 6 of ``propose_scored``).

    Returns:
        The event rows, ``resolved_by`` per FAILed tube and the spared flagged tubes.
    """
    open_ids = set(leftover["SampleID"].to_list())
    flagged = (pl.col("self_z") < rule.self_max).fill_null(False)
    tubes = scored.filter(pl.col("fit") & pl.col("USUBJID").is_not_null()).with_columns(
        flagged.alias("_flagged")
    )
    subjects = (
        tubes.group_by("USUBJID")
        .agg(
            pl.len().alias("_n"),
            pl.col("_flagged").sum().alias("_n_flagged"),
            (pl.col("_flagged") & pl.col("SampleID").is_in(list(open_ids))).sum().alias("_n_open"),
            pl.col("SampleID").is_in(list(kept)).any().alias("_kept"),
        )
        .filter(
            (pl.col("_n_flagged") >= 1)
            & (pl.col("_n_open") == pl.col("_n_flagged"))
            & (pl.col("_n") - pl.col("_n_flagged") >= 2)
            & ~pl.col("_kept")
        )["USUBJID"]
        .to_list()
    )
    m1 = rule.ungenotyped_naming == "two_agree" and "pqtl_call" in tubes.columns
    m4 = rule.split_tiebreak
    tubes = tubes.filter(pl.col("USUBJID").is_in(subjects))
    expected: dict[str, str] = {}
    cut = None
    if m4 and subjects:
        if p_male_col is not None and "SEX" in adsl_recorded.columns:
            # The exception, else ADSL SEX: never the anchors, which are the clean group
            # being judged.
            adsl_expected = expected_sex_adsl(
                adsl_recorded, scored, replace(rule, sex_from_anchors=False), p_male_col
            )
            expected = dict(
                adsl_expected.select("USUBJID", pl.col("SEX").cast(pl.String))
                .unique(subset=["USUBJID"])
                .iter_rows()
            )
        if age_col is not None and "AGE" in adsl_recorded.columns:
            res, mad = _age_residuals(scored, adsl_recorded, age_col)
            tubes = tubes.join(
                res.select("SampleID", "_resid"),
                on="SampleID",
                how="left",
                maintain_order="left",
            )
            cut = rule.age_discordant_mads * mad if mad > 0 else None
    if "_resid" not in tubes.columns:
        tubes = tubes.with_columns(pl.lit(None, dtype=pl.Float64).alias("_resid"))

    by_subject: dict[str, list[dict]] = {}
    for row in tubes.sort("SampleID").to_dicts():
        by_subject.setdefault(row["USUBJID"], []).append(row)
    rows: list[dict] = []
    basis: dict[str, str] = {}
    spared: set[str] = set()
    for subject in sorted(by_subject):
        group = by_subject[subject]
        clean = [t for t in group if not t["_flagged"]]
        flg = [t for t in group if t["_flagged"]]
        why = None
        if m1:
            donors = {t["_pdonor"] for t in clean}
            if (
                all(t["pqtl_self_z"] is None for t in group)
                and all(t["_call"] == "mismatch" for t in clean)
                and len(donors) == 1
                and None not in donors
                and all(t["_call"] != "mismatch" and t["_pdonor"] is None for t in flg)
            ):
                why = "pqtl"
        if why is None and m4:
            picks: list[tuple[str, str]] = []
            exp = expected.get(subject)
            s_clean = _group_sex(clean, p_male_col, rule.sex_prob)
            s_flg = _group_sex(flg, p_male_col, rule.sex_prob)
            if exp in ("M", "F") and s_clean is not None and s_flg is not None:
                if s_clean != exp and s_flg == exp:
                    picks.append(("clean", "sex"))
                elif s_flg != exp and s_clean == exp:
                    picks.append(("flagged", "sex"))
            a_clean, a_flg = _group_age(clean, cut), _group_age(flg, cut)
            if a_clean == "out" and a_flg == "in":
                picks.append(("clean", "age"))
            elif a_flg == "out" and a_clean == "in":
                picks.append(("flagged", "age"))
            for side, name in picks:
                foreign, other = (clean, flg) if side == "clean" else (flg, clean)
                if any(t["_call"] == "match" for t in foreign) or any(
                    t["_call"] == "mismatch" for t in other
                ):
                    continue
                why = name if side == "clean" else None
                break
        if why is None:
            continue
        for t in clean:
            rows.append({"SampleID": t["SampleID"], "USUBJID_INFERRED": t["_pdonor"]})
            basis[t["SampleID"]] = why
        spared |= {t["SampleID"] for t in flg}
    logger.info(
        "identity Lp splits: %d candidate subjects (three or more fit tubes, every flagged tube open, two or more "
        "Lp-clean); %d Lp-clean tubes FAIL as majority_foreign (%s), %d flagged tubes spared",
        len(by_subject),
        len(rows),
        ", ".join(f"{n} {b}" for b, n in sorted(Counter(basis.values()).items())) or "none",
        len(spared),
    )
    frame = pl.DataFrame(
        rows, schema={"SampleID": pl.String, "USUBJID_INFERRED": pl.String}
    ).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("VISITNUM_INFERRED"),
        pl.lit("majority_foreign").alias("event_type"),
        pl.lit("flagged").alias("action"),
        pl.lit(None, dtype=pl.String).alias("veto_reason"),
        pl.lit(None, dtype=pl.String).alias("unplaced_reason"),
    )
    return _cast_schema(frame, _EVENT_ROW_SCHEMA), basis, spared


def _event_rows(
    df: pl.DataFrame,
    event_type: str,
    action: str,
    subject: str | None,
    visit: str | None = None,
    *,
    unplaced: str | pl.Expr | None = None,
    veto: str | pl.Expr | None = None,
) -> pl.DataFrame:
    """Returns the rows of ``df`` as events (the first columns of ``_PROPOSE_SCHEMA``).

    ``subject`` and ``visit`` name the inferred-subject and inferred-visit columns of
    ``df`` (None: null for every row); a row with no inferred subject has no inferred
    visit. ``unplaced`` and ``veto`` are one reason for every row (a string), an
    expression giving it per row, or None.
    """

    def reason(r: str | pl.Expr | None) -> pl.Expr:
        if r is None:
            return pl.lit(None, dtype=pl.String)
        return pl.lit(r, dtype=pl.String) if isinstance(r, str) else r

    subj = pl.col(subject) if subject is not None else pl.lit(None, dtype=pl.String)
    vis = (
        pl.when(subj.is_not_null()).then(pl.col(visit))
        if visit is not None
        else pl.lit(None, dtype=pl.Float64)
    )
    return _cast_schema(
        df.select(
            "SampleID",
            subj.alias("USUBJID_INFERRED"),
            vis.alias("VISITNUM_INFERRED"),
            pl.lit(event_type).alias("event_type"),
            pl.lit(action).alias("action"),
            reason(veto).alias("veto_reason"),
            reason(unplaced).alias("unplaced_reason"),
        ),
        _EVENT_ROW_SCHEMA,
    )


def _with_pqtl_call(scored: pl.DataFrame) -> pl.DataFrame:
    """Adds ``_call`` and ``_pdonor`` to ``scored``.

    ``_call`` is ``pqtl_call``, or ``none`` without pQTL. ``_pdonor`` is the donor a
    ``mismatch`` names (``schemas.pqtl_named_donor``: the KING relative in
    ``pqtl_relative_USUBJID`` when set, else the named non-relative donor); null on
    every other tube.
    """
    if "pqtl_call" not in scored.columns:
        return scored.with_columns(
            pl.lit("none", dtype=pl.String).alias("_call"),
            pl.lit(None, dtype=pl.String).alias("_pdonor"),
        )
    return scored.with_columns(
        pl.col("pqtl_call").fill_null("none").alias("_call"),
        pqtl_named_donor(scored.columns).alias("_pdonor"),
    )


def _relative_pairs(relatives: frozenset[tuple[str, str]]) -> pl.DataFrame:
    """Returns every ordered pair of KING relatives as ``best_subject``, ``_pdonor``."""
    pairs = sorted({(a, b) for a, b in relatives} | {(b, a) for a, b in relatives})
    return pl.DataFrame(
        {"best_subject": [a for a, _ in pairs], "_pdonor": [b for _, b in pairs]},
        schema={"best_subject": pl.String, "_pdonor": pl.String},
    )


def relative_pairs(evidence, min_kinship: float) -> frozenset[tuple[str, str]]:
    """Returns genotyped subject pairs with KING kinship of at least ``min_kinship``.

    Read from ``PqtlEvidence.kinship``; ``min_kinship`` is ``pqtl_relative_kinship``.
    Each pair is ordered (smaller USUBJID first), so the set survives a JSON round trip
    that sorts pairs and a pair is one element.
    """
    k = np.nan_to_num(np.asarray(evidence.kinship, dtype=float), nan=-np.inf)
    i, j = np.nonzero(np.triu(k >= min_kinship, k=1))
    u = evidence.geno_usubjids
    return frozenset(
        tuple(sorted((u[a], u[b]))) for a, b in zip(i.tolist(), j.tolist(), strict=True)
    )


def _with_partner_acc(pool: pl.DataFrame) -> pl.DataFrame:
    """Adds ``_partner_acc``: the accessions of each row's reciprocal partners.

    The partners are the pool tubes that would leave the row's destination slot in the
    same reciprocal move (labeled with the destination, pointing back at this row's
    label). Comma-joined; null when there is none.
    """
    if "AccessionNumber" not in pool.columns:
        return pool.with_columns(pl.lit(None, dtype=pl.String).alias("_partner_acc"))
    back = (
        pool.filter(pl.col("AccessionNumber").is_not_null())
        .group_by(
            pl.col("best_subject").alias("USUBJID"),
            pl.col("USUBJID").alias("best_subject"),
        )
        .agg(pl.col("AccessionNumber").sort().str.join(",").alias("_partner_acc"))
    )
    return pool.join(back, on=["USUBJID", "best_subject"], how="left", maintain_order="left")


def propose(
    scored: pl.DataFrame,
    adsl: pl.DataFrame,
    rule: Rule,
    p_male_col: str | None,
    age_col: str | None,
    Wz: np.ndarray | None = None,
    *,
    records: pl.DataFrame,
    relatives: frozenset[tuple[str, str]] = frozenset(),
) -> pl.DataFrame:
    """Classifies all fit samples with ``self_z < self_max`` into events.

    Returns only the events of ``propose_scored``, in ``_PROPOSE_SCHEMA``.
    """
    return propose_scored(
        scored,
        adsl,
        rule,
        p_male_col,
        age_col,
        Wz,
        records=records,
        relatives=relatives,
    )[0]


def propose_scored(
    scored: pl.DataFrame,
    adsl: pl.DataFrame,
    rule: Rule,
    p_male_col: str | None,
    age_col: str | None,
    Wz: np.ndarray | None = None,
    *,
    records: pl.DataFrame,
    relatives: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Classifies all fit samples with ``self_z < self_max`` into events.

    The rule, in order:

    1. Two-sample subjects whose flagged tubes point at one partner are resolved on
       their own evidence (``_resolve_two_sample_subjects``).
    2. Proposals are vetoed (``_candidates``, then the partner of a vetoed move),
       checked against the donor pQTL names (``donor_conflict``) and placed onto the
       destination's own visit record within ``rule.date_tolerance_days`` of the draw,
       with no fallback to the label's visit.
    3. Placed tubes are ``one_way``, ``reciprocal`` or ``duplicate_draw`` by slot
       occupancy; collisions, broken pairs and, with ``rule.veto_unplaceable``, strong
       sex or age vetoes are ``unplaceable``.
    4. Leftover flagged tubes are ``vetoed`` (positive pQTL match), ``unplaceable``
       (``target_no_anchor``) or ``multi_match``.
    5. Two-fit-tube subjects follow ``_two_tube_events``; Lp splits of three or
       more fit tubes follow ``_split_events`` when a split key is on.
    6. Other flagged tubes are ``ambiguous``, ``unresolved`` (no anchor) or
       ``unattributable``.
    7. Lp-clean pQTL mismatches are ``majority_foreign`` or ``pqtl_only``; with
       ``rule.mixture_warn``, a high ``pqtl_mixture_z`` is ``pqtl_mixture`` (WARN).

    ``reciprocal`` and ``one_way`` are ``reassigned``, every other type ``flagged``. The
    correctly labeled tube of a reassigned two-sample pair gets no event. With pQTL off
    the pQTL keys (``mixture_warn``, ``ungenotyped_naming``) are inert.

    Args:
        scored: One row per sample with its Lp scores against every fit sample
            (``subject_scores`` without anchors) and, when pQTL ran, its
            ``PQTL_SCORED_COLUMNS``.
        adsl: ADSL with ``SEX`` and ``AGE`` for the enabled vetoes and tiebreaks.
        rule: The decision rule.
        p_male_col: Column of predicted P(male), or None without a sex model.
        age_col: Column of predicted age, or None without an age model.
        Wz: The Lp z matrix in ``scored`` row order. With it, best and second subjects
            are rescored on anchors only: fit samples with ``self_z >= self_max`` or a
            positive pQTL ``match``, and the correctly labeled tube of a resolved
            two-sample subject; a subject with no anchor is then never a target.
            Without it the scores are final and every subject is anchored.
        records: The dated ADaM visit records (``USUBJID``, ``VISITNUM``, ``ADT``,
            ``REF``) that placement reads.
        relatives: Pairs of genotyped subjects that are KING relatives.

    Returns:
        ``(events, scored)``. The events are in ``_PROPOSE_SCHEMA``:
        ``USUBJID_INFERRED`` is the named donor, ``VISITNUM_INFERRED`` the placed
        visit, ``veto_reason`` and ``unplaced_reason`` the cause of a ``vetoed`` or
        ``unplaceable`` event, ``resolved_by`` the evidence that named the tube (else
        ``none``) and ``date_gap_days`` the days from the tube's ``ADT`` to the inferred
        subject's nearest record. ``scored`` carries the final (anchor) scores.

    Raises:
        ValueError: If an enabled veto has its evidence but not its ADSL column
            (``SEX`` or ``AGE``), ``records`` lacks a column, or with pQTL on the scores
            lack a column a set pQTL key reads (``pqtl_mixture_z``, ``pqtl_self_z``).
    """
    # Imported here: placement imports manifest, which imports config, which imports
    # pqtl, which imports this module.
    from analysis.qc.placement import place

    if rule.sex_veto and p_male_col is not None and "SEX" not in adsl.columns:
        raise ValueError("identity sex veto is on but ADSL has no SEX column")
    if rule.age_veto_years is not None and age_col is not None and "AGE" not in adsl.columns:
        raise ValueError("identity age veto is on but ADSL has no AGE column")
    missing = [c for c in ("USUBJID", "VISITNUM", "ADT", "REF") if c not in records.columns]
    if missing:
        raise ValueError(f"identity visit records lack {', '.join(missing)}")
    # A pQTL key is inert on a run without pQTL (no pqtl_call); with pQTL on, a column
    # the key reads must be there (scores computed without it lack it).
    pqtl_on = "pqtl_call" in scored.columns
    for needed, column in (
        (rule.mixture_warn is not None, "pqtl_mixture_z"),
        (rule.ungenotyped_naming == "two_agree", "pqtl_self_z"),
    ):
        if needed and pqtl_on and column not in scored.columns:
            raise ValueError(
                f"the pQTL scores carry no {column}, which this rule reads: rescore them"
            )
    scored = _with_pqtl_call(scored.with_columns(ensure_date_column(scored, "ADT")))
    adsl_recorded = adsl
    # SEX from here on is the expected protein sex (an expected_sex exception, the
    # anchors with rule.sex_from_anchors, else ADSL SEX); AGE and every other column are
    # as recorded.
    adsl = expected_sex_adsl(adsl, scored, rule, p_male_col)
    is_flagged = pl.col("fit") & pl.col("self_z").is_not_null() & (pl.col("self_z") < rule.self_max)

    # ── Two-sample subjects: Lp can't say which tube moved ───────────────
    group = _two_sample_subjects(scored, scored.filter(is_flagged), _strong(scored, rule))
    moved: set[str] = set()
    staying: set[str] = set()
    resolved_by: dict[str, str] = {}
    if group.height > 0:
        partners = group["partner"].unique().to_list()
        partner_tubes = scored.filter(pl.col("USUBJID").is_in(partners)).select(
            "USUBJID", "SampleID", "VISITNUM", "_call", "_pdonor"
        )
        moved, staying, resolved_by = _resolve_two_sample_subjects(
            _pair_evidence(group, scored, adsl, rule, p_male_col, age_col),
            partner_tubes,
            relatives,
        )
    partner_of = dict(group.select("SampleID", "partner").iter_rows())
    subject_of = dict(group.select("SampleID", "USUBJID").iter_rows())

    # ── Anchors: rescore best/second subjects on anchor samples only ─────
    # Pass-1 best subject, kept as the lead when it has no anchor.
    lead = scored.select(
        "SampleID",
        pl.when(pl.col("best_z") >= rule.best_min).then(pl.col("best_subject")).alias("_lead"),
    )
    stuck: set[str] = set()
    if Wz is not None:
        assert Wz.shape[0] == scored.height
        fit = scored["fit"].to_numpy()
        anchor = (
            scored.select(
                (
                    pl.col("fit")
                    & (
                        (pl.col("self_z") >= rule.self_max).fill_null(False)
                        | (pl.col("_call") == "match")
                    )
                )
                | pl.col("SampleID").is_in(list(staying))
            )
            .to_series()
            .to_numpy()
        )
        rescored = subject_scores(Wz, scored["USUBJID"].to_numpy(), fit, anchor)
        scored = scored.with_columns(
            rescored.select("best_subject", "best_z", "second_subject", "second_z")
        )
        anchored = set(scored.filter(pl.Series(anchor))["USUBJID"].to_list())
        # A two-sample move is never made into a subject with no anchor.
        stuck = {s for s in moved if partner_of[s] not in anchored}
        moved -= stuck
        lead = lead.with_columns(
            pl.when(~pl.col("_lead").is_in(list(anchored))).then(pl.col("_lead")).alias("_lead")
        )
    else:
        anchored = set(scored["USUBJID"].to_list())
        lead = lead.with_columns(pl.lit(None, dtype=pl.String).alias("_lead"))
    flagged = scored.filter(is_flagged)
    logger.info(
        "identity check: %d labeled subjects with flagged samples and no anchor "
        "(never a target; their own flagged tubes are judged as any other)",
        len(set(flagged["USUBJID"].to_list()) - anchored),
    )

    multi = (
        (pl.col("best_z") >= rule.best_min)
        & (pl.col("second_z") >= rule.best_min)
        & (pl.col("best_z") - pl.col("second_z") < rule.margin_min)
    ).fill_null(False)
    # A multi-match whose pQTL names one of the two candidates, when they are not
    # relatives, is a proposal toward the named one.
    named_multi = (
        scored.filter(
            is_flagged
            & multi
            & ~pl.col("SampleID").is_in(list(partner_of))
            & (pl.col("_call") == "mismatch")
            & (
                (pl.col("_pdonor") == pl.col("best_subject"))
                | (pl.col("_pdonor") == pl.col("second_subject"))
            ).fill_null(False)
        )
        .join(
            _relative_pairs(relatives).rename({"_pdonor": "second_subject"}),
            on=["best_subject", "second_subject"],
            how="anti",
        )
        .with_columns(pl.col("_pdonor").alias("best_subject"))
    )
    pool = pl.concat(
        [
            _strong(scored, rule).filter(~pl.col("SampleID").is_in(list(partner_of))),
            scored.filter(pl.col("SampleID").is_in(list(moved))).with_columns(
                pl.col("SampleID")
                .replace_strict(partner_of, default=None, return_dtype=pl.String)
                .alias("best_subject")
            ),
            named_multi,
        ]
    ).sort("SampleID")

    proposed = pool

    # ── Vetoes, before placement: label, never drop ──────────────────────
    labeled = _candidates(pool, scored, adsl, rule, p_male_col, age_col)
    vetoed = labeled.filter(pl.col("veto_reason").is_not_null())
    pool = labeled.filter(pl.col("veto_reason").is_null())
    # The reciprocal of a vetoed move is vetoed with it.
    back = vetoed.select(
        pl.col("best_subject").alias("USUBJID"), pl.col("USUBJID").alias("best_subject")
    ).unique()
    vetoed = pl.concat(
        [
            vetoed,
            pool.join(back, on=["USUBJID", "best_subject"], how="semi").with_columns(
                pl.lit("partner").alias("veto_reason")
            ),
        ]
    )
    pool = pool.join(back, on=["USUBJID", "best_subject"], how="anti")
    # A sex- or age-vetoed tube that matches nobody of its own and whose blocked match
    # is strong enough is foreign, not merely WARN. A pQTL or partner veto stays WARN.
    vetoed_foreign = vetoed.clear()
    if rule.veto_unplaceable:
        floor = (
            rule.best_min
            if rule.veto_unplaceable_best_min is None
            else rule.veto_unplaceable_best_min
        )
        is_foreign = (
            pl.col("veto_reason").is_in(["sex", "age"])
            & (pl.col("self_z") < rule.self_max)
            & (pl.col("_call") != "match")
            & (pl.col("best_z") >= floor)
        ).fill_null(False)
        vetoed_foreign = vetoed.filter(is_foreign)
        vetoed = vetoed.filter(~is_foreign)
    # pQTL names another donor than Lp, and the two are not relatives.
    other_donor = pool.filter(
        (pl.col("_call") == "mismatch")
        & pl.col("_pdonor").is_not_null()
        & (pl.col("_pdonor") != pl.col("best_subject"))
    ).join(_relative_pairs(relatives), on=["best_subject", "_pdonor"], how="anti")
    pool = pool.join(other_donor.select("SampleID"), on="SampleID", how="anti")

    # ── Placement: the destination's own record on the draw date ─────────
    pool = place(
        _with_partner_acc(pool),
        "best_subject",
        records,
        rule.date_tolerance_days,
        partner_acc="_partner_acc",
    )
    unplaced = pool.filter(pl.col("unplaced_reason").is_not_null())
    cand = pool.filter(pl.col("unplaced_reason").is_null())

    # ── Slot occupancy: the inferred subject at the placed visit ─────────
    slots = scored.filter(pl.col("VISITNUM").is_not_null()).select(
        pl.col("USUBJID").alias("best_subject"),
        pl.col("VISITNUM").cast(pl.Float64).alias("_inf_visit"),
        pl.col("SampleID").alias("_slot"),
    )
    slot_n = slots.group_by("best_subject", "_inf_visit").agg(pl.len().alias("_n"))
    # A slot tube points back when it is itself a proposal toward this tube's label,
    # whether or not its own move survives placement and the vetoes.
    cand_best = proposed.select(
        pl.col("SampleID").alias("_slot"),
        pl.col("best_subject").alias("_slot_best"),
    )
    cand_with_n = cand.join(
        slot_n, on=["best_subject", "_inf_visit"], how="left", maintain_order="left"
    ).with_columns(pl.col("_n").fill_null(0))

    # ── Slot holds several samples (_n > 1): duplicate_draw ──────────────
    crowded = cand_with_n.filter(pl.col("_n") > 1)

    # ── Clear slot (_n <= 1): classify ───────────────────────────────────
    clear = (
        cand_with_n.filter(pl.col("_n") <= 1)
        .join(slots, on=["best_subject", "_inf_visit"], how="left", maintain_order="left")
        .join(cand_best, on="_slot", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("_n") == 0)
            .then(pl.lit("one_way"))
            .when(pl.col("_slot_best") == pl.col("USUBJID"))
            .then(pl.lit("reciprocal"))
            .otherwise(pl.lit("duplicate_draw"))
            .alias("_type"),
        )
    )
    proposals = clear.filter(pl.col("_type").is_in(REASSIGNED_EVENT_TYPES))
    occupied = clear.filter(pl.col("_type") == "duplicate_draw")

    # ── Collision: two proposals for one slot, neither moves ─────────────
    in_slot = pl.len().over("best_subject", "_inf_visit")
    collided = proposals.filter(in_slot > 1)
    proposals = proposals.filter(in_slot == 1)

    # ── Pair integrity: a reciprocal moves only if its partner does ──────
    broken: list[pl.DataFrame] = []
    while True:
        lost = proposals.filter(
            (pl.col("_type") == "reciprocal")
            & ~pl.col("_slot").is_in(proposals["SampleID"].to_list())
        )
        if lost.height == 0:
            break
        broken.append(lost)
        proposals = proposals.filter(~pl.col("SampleID").is_in(lost["SampleID"].to_list()))

    # ── The staying tube of a reassigned pair gets no event ──────────────
    reassigned_subjects = {subject_of[s] for s in moved & set(proposals["SampleID"].to_list())}
    kept = {s for s in staying if subject_of[s] in reassigned_subjects}

    target = (
        pl.when(pl.col("SampleID").is_in(list(partner_of)))
        .then(pl.col("SampleID").replace_strict(partner_of, default=None, return_dtype=pl.String))
        .when(pl.col("best_z") >= rule.best_min)
        .then(pl.col("best_subject"))
        .otherwise(pl.col("_lead"))
    )
    unplaceable = pl.concat(
        [
            _event_rows(
                unplaced,
                "unplaceable",
                "flagged",
                "best_subject",
                unplaced=pl.col("unplaced_reason"),
            ),
            _event_rows(
                other_donor,
                "unplaceable",
                "flagged",
                "best_subject",
                unplaced="donor_conflict",
            ),
            _event_rows(collided, "unplaceable", "flagged", "best_subject", unplaced="collision"),
            _event_rows(
                vetoed_foreign,
                "unplaceable",
                "flagged",
                "best_subject",
                unplaced="vetoed_destination",
                veto=pl.col("veto_reason"),
            ),
            *(
                _event_rows(b, "unplaceable", "flagged", "best_subject", unplaced="pair_broken")
                for b in broken
            ),
            _event_rows(
                scored.filter(pl.col("SampleID").is_in(list(stuck)))
                .join(lead, on="SampleID", how="left", maintain_order="left")
                .with_columns(target.alias("_target")),
                "unplaceable",
                "flagged",
                "_target",
                unplaced="target_no_anchor",
            ),
        ]
    )
    parts = [
        *(
            _event_rows(
                proposals.filter(pl.col("_type") == t),
                t,
                "reassigned",
                "best_subject",
                "_inf_visit",
            )
            for t in REASSIGNED_EVENT_TYPES
        ),
        _event_rows(
            pl.concat(
                [
                    crowded.select("SampleID", "best_subject", "_inf_visit"),
                    occupied.select("SampleID", "best_subject", "_inf_visit"),
                ]
            ),
            "duplicate_draw",
            "flagged",
            "best_subject",
            "_inf_visit",
        ),
        _event_rows(vetoed, "vetoed", "flagged", "best_subject", veto=pl.col("veto_reason")),
        unplaceable,
    ]

    # ── Flagged tubes left over ──────────────────────────────────────────
    n_fit = scored.filter(pl.col("fit")).group_by("USUBJID").agg(pl.len().alias("_n_fit"))
    two_tube = set(
        n_fit.filter(pl.col("_n_fit") == 2)
        .join(flagged.select("USUBJID").unique(), on="USUBJID", how="semi")["USUBJID"]
        .to_list()
    )
    done = set(pl.concat(parts)["SampleID"].to_list()) | kept | set(partner_of)
    rest = (
        flagged.filter(~pl.col("SampleID").is_in(list(done)))
        .join(lead, on="SampleID", how="left", maintain_order="left")
        .with_columns(target.alias("_target"))
    )
    in_pair = pl.col("USUBJID").is_in(list(two_tube))
    # A positive pQTL match vetoes a FAIL, not only a move; in a two-tube subject the
    # two-tube table decides.
    own_genotype = pl.lit(rule.pqtl_veto) & (pl.col("_call") == "match")
    leads = ~own_genotype & pl.col("_lead").is_not_null()
    multi_left = ~own_genotype & ~leads & multi
    parts += [
        _event_rows(
            rest.filter(own_genotype & ~in_pair),
            "vetoed",
            "flagged",
            "_target",
            veto="pqtl",
        ),
        _event_rows(
            rest.filter(leads),
            "unplaceable",
            "flagged",
            "_lead",
            unplaced="target_no_anchor",
        ),
        _event_rows(rest.filter(multi_left), "multi_match", "flagged", "best_subject"),
    ]
    leftover = rest.filter(~own_genotype & ~leads & ~multi_left & ~in_pair)

    # ── Two-tube subjects ────────────────────────────────────────────────
    so_far = pl.concat(parts)
    pair_tubes = _tube_evidence(
        scored.filter(pl.col("fit") & pl.col("USUBJID").is_in(list(two_tube)))
        .join(lead, on="SampleID", how="left", maintain_order="left")
        .with_columns(target.alias("_target")),
        scored,
        adsl,
        rule,
        p_male_col,
        age_col,
    )
    s2_rows, s2_basis = _two_tube_events(
        pair_tubes,
        {
            s: (t, v)
            for s, t, v in so_far.select("SampleID", "event_type", "veto_reason").iter_rows()
        },
        kept,
        anchored,
        rule,
    )
    resolved_by |= s2_basis
    if s2_rows:
        parts.append(
            _cast_schema(
                pl.DataFrame(
                    s2_rows,
                    schema={
                        "SampleID": pl.String,
                        "USUBJID_INFERRED": pl.String,
                        "event_type": pl.String,
                        "veto_reason": pl.String,
                    },
                ).with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("VISITNUM_INFERRED"),
                    pl.lit("flagged").alias("action"),
                    pl.lit(None, dtype=pl.String).alias("unplaced_reason"),
                ),
                _EVENT_ROW_SCHEMA,
            )
        )

    # ── Lp splits of three or more fit tubes ─────────────────────────────
    spared: set[str] = set()
    if (rule.ungenotyped_naming == "two_agree" and pqtl_on) or rule.split_tiebreak:
        split_rows, split_basis, spared = _split_events(
            scored, leftover, kept, rule, adsl_recorded, p_male_col, age_col
        )
        resolved_by |= split_basis
        parts.append(split_rows)
        leftover = leftover.filter(~pl.col("SampleID").is_in(list(spared)))

    # ── Three or more fit tubes: the flagged tubes left over ─────────────
    so_far = pl.concat(parts)
    named = so_far.filter(pl.col("event_type").is_in(NAMED_DONOR_TYPES))["SampleID"].to_list()
    fit_tubes = scored.filter(pl.col("fit")).with_columns(
        pl.col("SampleID").is_in(named).alias("_named")
    )
    per_subject = fit_tubes.group_by("USUBJID").agg(
        pl.len().alias("_n_fit"), pl.col("_named").sum().alias("_n_named")
    )
    leftover = leftover.join(per_subject, on="USUBJID", how="left", maintain_order="left")
    # Flagged only because every other tube is named as someone else's.
    explained = (
        (pl.col("_n_fit") > 1)
        & (pl.col("_n_named") == pl.col("_n_fit") - 1)
        & (pl.col("_call") != "mismatch")
    )
    unanchored = ~pl.col("USUBJID").is_in(list(anchored))
    parts += [
        _event_rows(leftover.filter(explained), "ambiguous", "flagged", None),
        _event_rows(leftover.filter(~explained & unanchored), "unresolved", "flagged", "_lead"),
        _event_rows(leftover.filter(~explained & ~unanchored), "unattributable", "flagged", None),
    ]
    # Majority foreign: a subject whose Lp-flagged tube is a positive match FAILs its
    # Lp-clean pQTL mismatches.
    has_event = set(pl.concat(parts)["SampleID"].to_list())
    true_owner = set(
        flagged.filter(pl.col("_call") == "match")
        .join(per_subject.filter(pl.col("_n_fit") >= 3), on="USUBJID", how="semi")["USUBJID"]
        .to_list()
    )
    unscored = fit_tubes.filter(
        ~pl.col("SampleID").is_in(list(has_event | kept)) & (pl.col("_call") == "mismatch")
    )
    is_majority = pl.col("USUBJID").is_in(list(true_owner)) & ~pl.col("SampleID").is_in(
        flagged["SampleID"].to_list()
    )
    parts += [
        _event_rows(unscored.filter(is_majority), "majority_foreign", "flagged", "_pdonor"),
        # ── pQTL-only: a mismatch Lp does not see ───────────────────────
        _event_rows(
            pl.concat(
                [
                    unscored.filter(~is_majority),
                    fit_tubes.filter(
                        pl.col("SampleID").is_in(list(kept)) & (pl.col("_call") == "mismatch")
                    ),
                ]
            ),
            "pqtl_only",
            "flagged",
            "_pdonor",
        ),
    ]
    # ── A mixture the other rules do not see (WARN) ──────────────────────
    if rule.mixture_warn is not None and pqtl_on:
        has_event = set(pl.concat(parts)["SampleID"].to_list())
        parts.append(
            _event_rows(
                fit_tubes.filter(
                    ~pl.col("SampleID").is_in(list(has_event | kept))
                    & (pl.col("pqtl_mixture_z") >= rule.mixture_warn).fill_null(False)
                ),
                "pqtl_mixture",
                "flagged",
                None,
            )
        )

    events = pl.concat(parts).sort("SampleID")
    assert events["SampleID"].is_unique().all(), "a SampleID landed in two event classes"
    unknown = set(events["event_type"].to_list()) - set(IDENTITY_EVENT_TYPES)
    assert not unknown, f"event types outside IDENTITY_EVENT_TYPES: {sorted(unknown)}"
    events = _with_gap(events, scored, records, resolved_by)
    for kind, column in (("unplaceable", "unplaced_reason"), ("vetoed", "veto_reason")):
        of_kind = events.filter(pl.col("event_type") == kind)
        logger.info(
            "identity check: %d %s (by reason: %s)",
            of_kind.height,
            kind,
            ", ".join(
                f"{n} {r}" for r, n in of_kind.group_by(column).len().sort(column).iter_rows()
            )
            or "none",
        )

    n_any = scored.group_by("USUBJID").agg(pl.col("fit").sum().alias("_n_fit"))
    unchecked = set(n_any.filter(pl.col("_n_fit") < 2)["USUBJID"].to_list())
    with_event = set(
        scored.filter(pl.col("SampleID").is_in(events["SampleID"].to_list()))["USUBJID"].to_list()
    )
    logger.info(
        "identity check: %d labeled subjects never checked (fewer than two fit samples, no event)",
        len(unchecked - with_event),
    )
    return events, scored.drop("_call", "_pdonor")


def _with_gap(
    events: pl.DataFrame,
    scored: pl.DataFrame,
    records: pl.DataFrame,
    resolved_by: dict[str, str],
) -> pl.DataFrame:
    """Returns ``events`` in ``_PROPOSE_SCHEMA``, with ``resolved_by`` and the date gap.

    ``resolved_by`` comes from the ``resolved_by`` map (``none`` for every other tube);
    ``date_gap_days`` is the days from the tube's ``ADT`` to the inferred subject's
    nearest record.
    """
    from analysis.qc.placement import date_gap

    dated = events.join(
        scored.select("SampleID", "ADT"),
        on="SampleID",
        how="left",
        maintain_order="left",
    )
    return _cast_schema(
        date_gap(dated, "USUBJID_INFERRED", records).with_columns(
            pl.col("SampleID")
            .replace_strict(resolved_by, default="none", return_dtype=pl.String)
            .alias("resolved_by")
        ),
        _PROPOSE_SCHEMA,
    )


def evaluate(events: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Scores reassignments against the truth table.

    Only ``action == "reassigned"`` rows of ``events`` count.

    Returns:
        A dict of counts: ``events``, the reassignable (reciprocal or one_way) events
        of the truth table; ``correct_events``, those whose every sample was reassigned
        correctly; ``wrong_samples``, reassigned samples with the wrong subject or
        visit; ``missed_samples``, reassignable truth samples not reassigned correctly.
    """
    j = events.filter(pl.col("action") == "reassigned").join(truth, on="SampleID", how="left")
    correct = j.filter(
        (
            (pl.col("USUBJID_INFERRED") == pl.col("true_USUBJID"))
            & (pl.col("VISITNUM_INFERRED") == pl.col("true_VISITNUM"))
        ).fill_null(False)
    )
    reassignable = truth.filter(pl.col("event_type").is_in(REASSIGNED_EVENT_TYPES))
    ok = reassignable.join(
        correct.select("SampleID", pl.lit(True).alias("ok")),
        on="SampleID",
        how="left",
    ).with_columns(pl.col("ok").fill_null(False))
    per_event = ok.group_by("event_id").agg(pl.col("ok").all())
    return {
        "events": per_event.height,
        "correct_events": int(per_event["ok"].sum()),
        "wrong_samples": j.height - correct.height,
        "missed_samples": int((~ok["ok"]).sum()),
    }


# ── Fit mask ─────────────────────────────────────────────────────────────────


def _build_fit_mask(qc_instance, sample_ids: list[str]) -> np.ndarray:
    """Returns a mask that is True for samples carrying no ``UNFIT_FLAGS`` flag."""
    sid = qc_instance.sample_id_col
    flagged_ids: set[str] = set()
    for qc_df in qc_instance.sample_qc_actions:
        for col in set(qc_df.columns) & UNFIT_FLAGS:
            flagged_ids.update(qc_df.filter(pl.col(col))[sid].cast(pl.Utf8).to_list())
    return np.array([str(s) not in flagged_ids for s in sample_ids])


# ── Top-level orchestrator ───────────────────────────────────────────────────


def _empty_events() -> pl.DataFrame:
    """Returns an empty events frame with ``IDENTITY_EVENTS_SCHEMA``."""
    return pl.DataFrame(schema=IDENTITY_EVENTS_SCHEMA)


def fill_pqtl_only(dna_link: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Fills ``n_pqtl_only`` in ``dna_link`` with each subject's ``pqtl_only`` events.

    Events count under ``USUBJID_LABELED``, the label the tube was scored under, which
    is the ``USUBJID`` of the link table; a subject with none gets 0.

    Where ``proteome_sex_vs_genome`` is ``opposite_own_dna`` and the subject has an
    identity event on any tube, ``sex_note`` is null. ``proteome_sex_vs_genome`` is kept.
    """
    counts = (
        events.filter(pl.col("event_type") == "pqtl_only")
        .group_by("USUBJID_LABELED")
        .agg(pl.len().cast(pl.Int64).alias("_n"))
        .rename({"USUBJID_LABELED": "USUBJID"})
    )
    flagged = events["USUBJID_LABELED"].drop_nulls().unique().to_list()
    return (
        dna_link.join(counts, on="USUBJID", how="left", maintain_order="left")
        .with_columns(
            pl.col("_n").fill_null(0).alias("n_pqtl_only"),
            pl.when(
                (pl.col("proteome_sex_vs_genome") == "opposite_own_dna")
                & pl.col("USUBJID").is_in(flagged)
            )
            .then(None)
            .otherwise(pl.col("sex_note"))
            .alias("sex_note"),
        )
        .select([pl.col(c).cast(t) for c, t in DNA_LINK_SCHEMA.items()])
    )


def write_identity_outputs(qc_instance, save_prefix: str) -> None:
    """Writes the identity stage's parquet files under ``save_prefix``.

    - ``.identity_events.parquet``, in ``schemas.IDENTITY_EVENTS_SCHEMA``: the events,
      empty with the schema when identity found nothing or did not run.
    - ``.unfit_identity.parquet``: ``qc_instance.unfit_identity``; empty with its
      schema when identity did not run.
    - When pQTL ran, ``.pqtl_scores.parquet`` (``qc_instance.pqtl_scores``, in
      ``schemas.PQTL_SCORES_SCHEMA``) and ``.dna_link.parquet``
      (``qc_instance.pqtl_evidence.dna_link`` with ``n_pqtl_only`` filled by
      ``fill_pqtl_only``, left null when identity did not run). With pQTL off neither
      is written and one log line says so.

    Raises:
        ValueError: If only one of the pQTL scores and evidence is set.
    """
    events = qc_instance.identity_events
    if events is None:
        events = pl.DataFrame(schema=IDENTITY_EVENTS_SCHEMA)
    events = events.select([pl.col(c).cast(t) for c, t in IDENTITY_EVENTS_SCHEMA.items()])
    events_file = f"{save_prefix}.identity_events.parquet"
    events.write_parquet(events_file)
    logger.info("saved: %s (%d events)", events_file, events.height)
    # The unfit-tube identity annotation, in its own file: empty when identity did not
    # run.
    unfit = getattr(qc_instance, "unfit_identity", None)
    if unfit is None:
        unfit = pl.DataFrame(schema=UNFIT_IDENTITY_SCHEMA)
    unfit_file = f"{save_prefix}.unfit_identity.parquet"
    unfit.select([pl.col(c).cast(t) for c, t in UNFIT_IDENTITY_SCHEMA.items()]).write_parquet(
        unfit_file
    )
    logger.info(
        "saved: %s (%d technically failed tubes; %s)",
        unfit_file,
        unfit.height,
        ", ".join(
            f"{n} {c}"
            for c, n in unfit.group_by("unfit_identity").len().sort("unfit_identity").iter_rows()
        )
        or "none",
    )

    scores = getattr(qc_instance, "pqtl_scores", None)
    evidence = getattr(qc_instance, "pqtl_evidence", None)
    if scores is None and evidence is None:
        logger.info(
            "pQTL is off: %s.pqtl_scores.parquet and %s.dna_link.parquet not written",
            save_prefix,
            save_prefix,
        )
        return
    if scores is None or evidence is None:
        raise ValueError(
            "pqtl_scores and pqtl_evidence must be set together; got "
            f"pqtl_scores {'set' if scores is not None else 'None'}, "
            f"pqtl_evidence {'set' if evidence is not None else 'None'}"
        )
    scores = scores.select([pl.col(c).cast(t) for c, t in PQTL_SCORES_SCHEMA.items()])
    link = evidence.dna_link
    link = (
        link.select([pl.col(c).cast(t) for c, t in DNA_LINK_SCHEMA.items()])
        if qc_instance.identity_events is None
        else fill_pqtl_only(link, events)
    )
    scores_file = f"{save_prefix}.pqtl_scores.parquet"
    link_file = f"{save_prefix}.dna_link.parquet"
    scores.write_parquet(scores_file)
    link.write_parquet(link_file)
    logger.info("saved: %s (%d tubes)", scores_file, scores.height)
    logger.info("saved: %s (%d genotype links)", link_file, link.height)


def _pqtl_pair_evidence(events: pl.DataFrame, evidence) -> pl.DataFrame:
    """Add ``pqtl_inferred_z``, ``genotype_kinship`` and ``genotype_unrelated``.

    ``pqtl_inferred_z`` is the sample's robust z against the inferred
    subject's genotype.  ``genotype_kinship`` is the KING kinship between the
    labeled and inferred subjects' genotypes (``evidence.kinship``), null when the
    pair is not in the KING table;
    ``genotype_unrelated`` is True when both are genotyped and that kinship is below
    ``KING_UNRELATED`` or unlisted (two distinct, unrelated people), False for
    relatives or duplicates.  Both are null where either subject is not genotyped or
    no subject was inferred.
    """
    from analysis.qc.pqtl import KING_UNRELATED

    row = {s: i for i, s in enumerate(evidence.sample_ids)}
    col = {u: j for j, u in enumerate(evidence.geno_usubjids)}
    inferred_z: list[float | None] = []
    kinship: list[float | None] = []
    unrelated: list[bool | None] = []
    for sample, labeled, inferred in zip(
        events["SampleID"].to_list(),
        events["USUBJID_LABELED"].to_list(),
        events["USUBJID_INFERRED"].to_list(),
        strict=True,
    ):
        j = col.get(inferred)
        i = row.get(sample)
        z = evidence.S[i, j] if i is not None and j is not None else np.nan
        inferred_z.append(float(z) if np.isfinite(z) else None)
        k = col.get(labeled)
        if j is None or k is None:
            kinship.append(None)
            unrelated.append(None)
            continue
        kin = evidence.kinship[k, j]
        kinship.append(float(kin) if np.isfinite(kin) else None)
        # NaN is a pair the KING table leaves out, read as unrelated.
        unrelated.append(not kin >= KING_UNRELATED)
    return events.with_columns(
        pl.Series("pqtl_inferred_z", inferred_z, dtype=pl.Float64),
        pl.Series("genotype_kinship", kinship, dtype=pl.Float64),
        pl.Series("genotype_unrelated", unrelated, dtype=pl.Boolean),
    )


@dataclass(frozen=True, eq=False)
class IdentityInputs:
    """Everything the identity stage reads from the QC instance.

    Fixed before any scoring key applies (``identity_inputs``), so the stage can be
    rerun at other ``[identity]`` and pQTL settings without the driver
    (``identity_from_inputs``).

    Attributes:
        X: The n x p identity matrix (``build_identity_matrix``).
        samples: The labels of the rows of ``X``, with the plate and the technical
            ``fit`` mask (``_build_fit_mask``).
        protein_ids: The column labels of ``X``.
        adsl: Collected ADSL, with the ``expected_sex`` exception column when the
            config has one.
        accession: One ``AccessionNumber`` per ``SampleID``.
        sex_oof: ``SampleID``, ``Predicted_Male_Probability``; None when the sex model
            did not run.
        age_oof: ``SampleID``, ``Predicted_Age`` (plus ``Age_Residual_Corrected`` when
            the age model wrote it); None when the age model did not run.
        records: The dated ADaM visit records, collected.
        plate_col: The plate column of ``samples``.
        sample_id_col: The QC instance's sample ID column.
        seed: The run's random seed.
        platform_name: The platform's name.
        wells: ``SampleID``, ``well_position`` for triage only; None when the platform
            metadata has no well column.
        residuals: Residuals shared with the pQTL stage, or None.
    """

    X: np.ndarray
    samples: pl.DataFrame
    protein_ids: list[str]
    adsl: pl.DataFrame
    accession: pl.DataFrame
    sex_oof: pl.DataFrame | None
    age_oof: pl.DataFrame | None
    records: pl.DataFrame
    plate_col: str
    sample_id_col: str
    seed: int
    platform_name: str
    wells: pl.DataFrame | None
    residuals: ResidualCache | None = None


def _to_sample_id(frame: pl.DataFrame, sid: str) -> pl.DataFrame:
    return frame.rename({sid: "SampleID"}) if sid != "SampleID" else frame


def identity_inputs(qc_instance) -> IdentityInputs:
    """Collects the identity stage's inputs from ``qc_instance``.

    Raises:
        ValueError: If the plate column (from ``sample_metadata``), ADSL ``TRT01A`` or
            the visit records (``qc.visit_records``) are missing.
    """
    sid = qc_instance.sample_id_col
    plate_col = qc_instance.plate_id_col

    # The plate comes from sample_metadata, the arm from ADSL.
    missing = []
    meta_cols = qc_instance._ensure_lazy(qc_instance.sample_metadata).collect_schema().names()
    if plate_col not in meta_cols:
        missing.append(plate_col)
    if "TRT01A" not in qc_instance._ensure_lazy(qc_instance.adsl).collect_schema().names():
        missing.append("TRT01A")
    if missing:
        raise ValueError(f"Identity check needs columns {missing}, which are missing")
    records = getattr(qc_instance, "visit_records", None)
    if records is None:
        raise ValueError("identity check needs the dated visit records (qc.visit_records)")
    records = qc_instance._ensure_lazy(records).collect()

    # Feature matrix and the fit mask from technical QC.
    X, samples_df, protein_ids, residuals = matrix_for(qc_instance)
    fit = _build_fit_mask(qc_instance, samples_df[sid].to_list())
    samples_df = samples_df.with_columns(pl.Series("fit", fit))

    adsl = qc_instance._ensure_lazy(qc_instance.adsl).collect()
    if qc_instance.expected_sex is not None:
        # The expected_sex exceptions ride along as a column; propose_scored's sex
        # evidence reads it before ADSL SEX, which stays as recorded. Absent (the frame
        # unchanged) when there are none.
        adsl = adsl.join(
            qc_instance.expected_sex.rename({qc_instance.usubjid_col: "USUBJID"}),
            on="USUBJID",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    accession = (
        qc_instance._ensure_lazy(qc_instance.manifest)
        .select(
            pl.col(sid).cast(pl.String).alias("SampleID"),
            pl.col("AccessionNumber").cast(pl.String),
        )
        .collect()
        .sort("SampleID", "AccessionNumber", nulls_last=True)
        .unique(subset=["SampleID"], keep="first", maintain_order=True)
    )

    sex_oof = None
    if qc_instance.sex_oof_predictions is not None:
        sex_oof = _to_sample_id(qc_instance.sex_oof_predictions, sid).select(
            ["SampleID", "Predicted_Male_Probability"]
        )
    age_oof = None
    if qc_instance.age_oof_predictions is not None:
        age_df = _to_sample_id(qc_instance.age_oof_predictions, sid)
        age_cols = ["SampleID", "Predicted_Age"]
        if AGE_RESIDUAL_CORRECTED_COL in age_df.columns:
            age_cols.append(AGE_RESIDUAL_CORRECTED_COL)
        age_oof = age_df.select(age_cols)

    # Well position from the sample metadata (triage-only; never used for
    # automated decisions).
    meta = qc_instance._ensure_lazy(qc_instance.sample_metadata).collect()
    if qc_instance.platform.name == "olink":
        well_col = "WellID" if "WellID" in meta.columns else None
    else:
        well_col = None
        if "SlideId" in meta.columns and "Subarray" in meta.columns:
            meta = meta.with_columns(
                (
                    pl.col("SlideId").cast(pl.String) + ":" + pl.col("Subarray").cast(pl.String)
                ).alias("_well_position")
            )
            well_col = "_well_position"
    wells = None
    if well_col is not None:
        wells = (
            meta.select([sid, well_col])
            .sort([sid, well_col], nulls_last=True)
            .unique(subset=[sid], keep="first", maintain_order=True)
            .rename({sid: "SampleID", well_col: "well_position"})
            .with_columns(pl.col("SampleID").cast(pl.String))
        )

    return IdentityInputs(
        X=X,
        samples=samples_df,
        protein_ids=protein_ids,
        adsl=adsl,
        accession=accession,
        sex_oof=sex_oof,
        age_oof=age_oof,
        records=records,
        plate_col=plate_col,
        sample_id_col=sid,
        seed=qc_instance.random_seed,
        platform_name=qc_instance.platform.name,
        wells=wells,
        residuals=residuals,
    )


def identity_rule(config) -> Rule:
    """Builds the decision ``Rule`` of a ``TrialConfig``.

    ``veto_unplaceable_best_min`` stays None when the config leaves
    ``identity_veto_unplaceable_best_min`` unset; the rule then reads ``best_min``, the
    value ``TrialConfig.veto_unplaceable_best_min`` falls back to, so the decision is
    the same and a rule rebuilt with another ``best_min`` keeps tracking it.
    """
    return Rule(
        self_max=config.identity_self_max,
        best_min=config.best_min,
        margin_min=config.margin_min,
        sex_veto=config.identity_sex_veto,
        age_veto_years=config.age_veto_years,
        tiebreak=config.identity_tiebreak,
        date_tolerance_days=config.identity_reassign_date_tolerance_days,
        pqtl_veto=config.identity_pqtl_veto,
        sex_prob=config.identity_sex_prob,
        age_discordant_mads=config.identity_age_discordant_mads,
        age_fit_mads=config.identity_age_fit_mads,
        age_veto_residual=config.identity_age_veto_residual,
        veto_unplaceable=config.identity_veto_unplaceable,
        veto_unplaceable_best_min=config.identity_veto_unplaceable_best_min,
        sex_from_anchors=config.identity_sex_from_anchors,
        ungenotyped_naming=config.pqtl_ungenotyped_naming,
        split_tiebreak=config.identity_split_tiebreak,
        mixture_warn=config.pqtl_mixture_warn,
    )


# The pqtl_scores columns the decision rule reads (lp_scored joins them; rescoring
# replaces them): the call, the named donors, the label's own z (null: ungenotyped
# label) and the mixture score. The first two are in every pQTL scores frame; the rest
# may be absent from scores computed without them, and propose_scored raises only when
# a key that reads one is set.
PQTL_SCORED_COLUMNS: tuple[str, ...] = (
    "pqtl_call",
    "pqtl_best_nonrel_USUBJID",
    "pqtl_relative_USUBJID",
    "pqtl_self_z",
    "pqtl_mixture_z",
)
PQTL_SCORED_REQUIRED: tuple[str, ...] = PQTL_SCORED_COLUMNS[:2]


def lp_scored(
    inp: IdentityInputs, config, pqtl_scores: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, np.ndarray]:
    """Returns ``(scored, Wz)``: the pass-1 scored frame and the Lp z matrix.

    Scored at ``config``'s Lp settings (``LpParams.from_config``). ``scored`` is what
    ``propose_scored`` receives: ``inp.samples`` with ``subject_scores`` against every
    fit sample, the accession, the sex and age OOF predictions and, with
    ``pqtl_scores``, the ``PQTL_SCORED_COLUMNS`` it has.
    """
    sid = inp.sample_id_col
    Wz = lp_matrix(
        inp.X,
        inp.samples,
        inp.adsl,
        seed=inp.seed,
        arm_col="TRT01A",
        plate_col=inp.plate_col,
        params=LpParams.from_config(config),
        residuals=inp.residuals,
    )
    # Subject scores against every fit sample (propose rescores on anchors).
    usubjid = inp.samples["USUBJID"].to_numpy()
    scores = subject_scores(Wz, usubjid, inp.samples["fit"].to_numpy())
    scored = pl.concat([inp.samples, scores.drop("sample_idx")], how="horizontal")
    if sid != "SampleID":
        scored = scored.rename({sid: "SampleID"})
    scored = scored.with_columns(ensure_date_column(scored, "ADT"))
    scored = scored.join(inp.accession, on="SampleID", how="left", maintain_order="left")
    if inp.sex_oof is not None:
        scored = scored.join(inp.sex_oof, on="SampleID", how="left", maintain_order="left")
    if inp.age_oof is not None:
        scored = scored.join(inp.age_oof, on="SampleID", how="left", maintain_order="left")
    if pqtl_scores is not None:
        scored = scored.join(
            pqtl_scores.select(
                "SampleID",
                *PQTL_SCORED_REQUIRED,
                *(c for c in PQTL_SCORED_COLUMNS[2:] if c in pqtl_scores.columns),
            ),
            on="SampleID",
            how="left",
            maintain_order="left",
        )
    return scored, Wz


def unfit_failures(
    flags: pl.DataFrame, stats: pl.DataFrame | None, n_std: float | None, platform: str
) -> pl.DataFrame:
    """Lists the technically failed tubes of ``flags`` that no ``exclude`` removed.

    Args:
        flags: ``SampleID`` and one Boolean column per flag of ``UNFIT_FLAGS``.
        stats: ``SampleID`` and the sample summary's outlier statistics (``PCk_std``,
            ``Z_Median``, ``Z_IQR``); None only when no failed tube fails PCA or
            Median/IQR.
        n_std: The outlier cut the statistics were judged at; None under the same
            condition.
        platform: The platform's key in ``UNRELIABLE_FAILURES``.

    Returns:
        One row per tube: ``SampleID``, ``technical_flags``, ``failure_kind``,
        ``z_over_cut`` and ``score_reliable``. ``z_over_cut`` is the largest absolute
        statistic of the tube's failing outlier steps minus ``n_std`` (null for RowCheck
        and Sample QC alone). A score is unreliable when the tube carries every flag of
        one of the platform's ``UNRELIABLE_FAILURES`` combinations or is more than
        ``UNFIT_MAX_Z_OVER_CUT`` z over its cut.

    Raises:
        KeyError: If ``platform`` has no entry.
        ValueError: If a flag column is missing, or a tube fails PCA or Median/IQR
            without its outlier statistics.
    """
    unreliable = UNRELIABLE_FAILURES[platform]
    tech = list(UNFIT_FAILURE_KINDS)
    missing = [f for f in [*tech, CORRECTION_EXCLUDE_FLAG] if f not in flags.columns]
    if missing:
        raise ValueError(f"the unfit-tube annotation needs the flag columns {missing}")
    failed = flags.filter(pl.any_horizontal(tech) & ~pl.col(CORRECTION_EXCLUDE_FLAG))
    stat_flags = list(_OUTLIER_STATISTICS)
    needs = failed.filter(pl.any_horizontal(stat_flags))
    if needs.height and (stats is None or n_std is None):
        raise ValueError(
            f"{needs.height} technically failed tubes fail PCA or Median/IQR, but the outlier statistics "
            f"({'none' if stats is None else 'present'}) or the cut ({n_std}) are missing"
        )
    if stats is not None:
        failed = failed.join(
            stats, on="SampleID", how="left", maintain_order="left", validate="1:1"
        )
        over = [
            pl.when(pl.col(flag)).then(pl.max_horizontal(pl.col(pattern).abs()))
            for flag, pattern in _OUTLIER_STATISTICS.items()
        ]
        z_over_cut = (
            pl.max_horizontal(over) - n_std if n_std is not None else pl.lit(None, dtype=pl.Float64)
        )
    else:
        z_over_cut = pl.lit(None, dtype=pl.Float64)
    failed = failed.with_columns(z_over_cut.cast(pl.Float64).alias("z_over_cut"))
    unmeasured = failed.filter(pl.any_horizontal(stat_flags) & pl.col("z_over_cut").is_null())
    if unmeasured.height:
        raise ValueError(
            f"{unmeasured.height} tubes fail PCA or Median/IQR with no outlier statistics row: "
            f"{unmeasured['SampleID'].head(5).to_list()}"
        )
    unreliable_kind = pl.any_horizontal(
        [pl.all_horizontal(sorted(combo)) for combo in unreliable] or [pl.lit(False)]
    )
    return failed.select(
        pl.col("SampleID").cast(pl.String),
        pl.concat_str(
            [pl.when(pl.col(f)).then(pl.lit(f)) for f in sorted(tech)],
            separator=";",
            ignore_nulls=True,
        ).alias("technical_flags"),
        pl.concat_str(
            [pl.when(pl.col(f)).then(pl.lit(k)) for f, k in UNFIT_FAILURE_KINDS.items()],
            separator="+",
            ignore_nulls=True,
        ).alias("failure_kind"),
        pl.col("z_over_cut"),
        (~unreliable_kind & ~(pl.col("z_over_cut") > UNFIT_MAX_Z_OVER_CUT).fill_null(False)).alias(
            "score_reliable"
        ),
    ).sort("SampleID")


def unfit_identity(
    scored: pl.DataFrame,
    failures: pl.DataFrame,
    adsl: pl.DataFrame,
    *,
    self_max: float,
    margin_min: float,
    relatives: frozenset[tuple[str, str]] = frozenset(),
    pqtl_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Builds the unfit-tube identity annotation, one row per tube of ``failures``.

    The rows are in ``schemas.UNFIT_IDENTITY_SCHEMA``. The annotation reads the pass-1
    Lp scores of ``scored`` (``self_z`` against the labeled subject's fit tubes,
    ``pass1_best_subject``, ``pass1_best_z`` and ``pass1_second_z`` against every other
    subject's fit tubes; ``identity_from_inputs`` with ``with_pass1``), so a failed tube
    is never a reference and nothing is retrained. ``unfit_identity`` is
    ``no_comparator`` without a self z; ``consistent_with_label`` at a self z of at
    least ``self_max`` on a reliable failure kind; ``names_other_subject`` at a self z
    below it, a best z of at least ``UNFIT_BEST_MIN`` (not the rule's ``best_min``), a
    margin over the second of at least ``margin_min`` and a best subject that is no
    KING relative of the label (``relatives``); ``uninformative`` otherwise. The pQTL
    call stands beside it (``pqtl_interpretable`` False on a tube failing PCA or
    Median/IQR). Every row says ``NO_SEX_AGE_EVIDENCE``.

    Raises:
        ValueError: If a failed tube carries a sex or age prediction (the models skip
            them) or is missing from ``scored``.
    """
    p1 = scored.select(
        "SampleID",
        "USUBJID",
        pl.col("VISITNUM").cast(pl.Float64).alias("VISITNUM_LABELED"),
        "self_z",
        pl.col("pass1_best_subject").alias("best_USUBJID"),
        pl.col("pass1_best_z").alias("best_z"),
        pl.col("pass1_second_z").alias("second_z"),
        *[c for c in ("Predicted_Male_Probability", "Predicted_Age") if c in scored.columns],
    )
    t = failures.join(p1, on="SampleID", how="left", maintain_order="left", validate="1:1")
    absent = t.filter(pl.col("USUBJID").is_null())
    if absent.height:
        raise ValueError(
            f"{absent.height} technically failed tubes have no identity scores: {absent['SampleID'].head(5).to_list()}"
        )
    predicted = [c for c in ("Predicted_Male_Probability", "Predicted_Age") if c in t.columns]
    with_prediction = t.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in predicted] or [pl.lit(False)])
    )
    if with_prediction.height:
        raise ValueError(
            f"{with_prediction.height} technically failed tubes carry a sex or age prediction, which the models skip; "
            f"the annotation says {NO_SEX_AGE_EVIDENCE!r}"
        )
    pairs = sorted({(a, b) for a, b in relatives} | {(b, a) for a, b in relatives})
    rel = pl.DataFrame(
        {
            "USUBJID": [a for a, _ in pairs],
            "best_USUBJID": [b for _, b in pairs],
            "_rel": [True] * len(pairs),
        },
        schema={"USUBJID": pl.String, "best_USUBJID": pl.String, "_rel": pl.Boolean},
    )
    t = t.join(rel, on=["USUBJID", "best_USUBJID"], how="left", maintain_order="left").with_columns(
        pl.when(pl.col("best_USUBJID").is_not_null())
        .then(pl.col("_rel").fill_null(False))
        .alias("best_related")
    )
    if "SITEID" in adsl.columns:
        site = (
            adsl.select("USUBJID", pl.col("SITEID").cast(pl.String))
            .unique(subset=["USUBJID"])
            .sort("USUBJID")
        )
        t = (
            t.join(site, on="USUBJID", how="left", maintain_order="left")
            .join(
                site.rename({"USUBJID": "best_USUBJID", "SITEID": "_best_site"}),
                on="best_USUBJID",
                how="left",
                maintain_order="left",
            )
            .with_columns((pl.col("SITEID") == pl.col("_best_site")).alias("same_site"))
        )
    else:
        t = t.with_columns(pl.lit(None, dtype=pl.Boolean).alias("same_site"))
    if pqtl_scores is not None:
        t = t.join(
            pqtl_scores.select("SampleID", "pqtl_call", "pqtl_self_z", "pqtl_best_USUBJID"),
            on="SampleID",
            how="left",
            maintain_order="left",
            validate="1:1",
        ).with_columns(
            # No finite score: no evidence, as in the events.
            pl.col("pqtl_call").fill_null("none"),
            ~pl.col("technical_flags")
            .str.contains("|".join(sorted(_OUTLIER_STATISTICS)))
            .alias("pqtl_interpretable"),
        )
    else:
        t = t.with_columns(
            pl.lit(None, dtype=pl.String).alias("pqtl_call"),
            pl.lit(None, dtype=pl.Float64).alias("pqtl_self_z"),
            pl.lit(None, dtype=pl.String).alias("pqtl_best_USUBJID"),
            pl.lit(None, dtype=pl.Boolean).alias("pqtl_interpretable"),
        )
    names_other = (
        (pl.col("self_z") < self_max)
        & pl.col("best_USUBJID").is_not_null()
        & (pl.col("best_z") >= UNFIT_BEST_MIN)
        & ((pl.col("best_z") - pl.col("second_z").fill_null(float("-inf"))) >= margin_min)
        & ~pl.col("best_related")
    )
    t = t.with_columns(
        pl.when(pl.col("self_z").is_null())
        .then(pl.lit("no_comparator"))
        .when((pl.col("self_z") >= self_max) & pl.col("score_reliable"))
        .then(pl.lit("consistent_with_label"))
        .when(names_other.fill_null(False))
        .then(pl.lit("names_other_subject"))
        .otherwise(pl.lit("uninformative"))
        .alias("unfit_identity"),
        (pl.col("best_z") - pl.col("second_z")).alias("margin"),
        pl.col("USUBJID").alias("USUBJID_LABELED"),
        pl.lit(NO_SEX_AGE_EVIDENCE).alias("sex_age_evidence"),
    )
    return _cast_schema(t, UNFIT_IDENTITY_SCHEMA).sort("SampleID")


def unfit_identity_for(
    qc_instance, inp: IdentityInputs, scored: pl.DataFrame, config
) -> pl.DataFrame:
    """Builds a run's unfit-tube identity annotation (``unfit_identity``).

    Reads the technical flags of ``qc_instance.sample_qc_actions``, the outlier
    statistics ``qc_instance._pca_results`` judged at ``qc_instance.outlier_n_std``, the
    pQTL scores and the KING relatives (``pqtl_relative_kinship``) when pQTL ran, and
    ``identity_self_max`` and ``margin_min`` of ``config``. ``scored`` is the frame
    ``identity_from_inputs`` returns with ``with_pass1``; it is read, never changed.

    Raises:
        ValueError: If a technically failed tube is fit in ``scored``.
    """
    sid = qc_instance.sample_id_col
    flagged: dict[str, set[str]] = {f: set() for f in UNFIT_FLAGS}
    for qc_df in qc_instance.sample_qc_actions:
        for col in set(qc_df.columns) & UNFIT_FLAGS:
            flagged[col].update(qc_df.filter(pl.col(col))[sid].cast(pl.Utf8).to_list())
    flags = scored.select("SampleID").with_columns(
        pl.col("SampleID").is_in(sorted(flagged[f])).alias(f) for f in sorted(UNFIT_FLAGS)
    )
    stats = getattr(qc_instance, "_pca_results", None)
    if stats is not None:
        stats = (
            _to_sample_id(stats, sid)
            .select("SampleID", pl.col(r"^PC\d+_std$"), pl.col(r"^Z_(Median|IQR)$"))
            .with_columns(pl.col("SampleID").cast(pl.String))
        )
    failures = unfit_failures(flags, stats, qc_instance.outlier_n_std, inp.platform_name)
    fit = scored.filter(pl.col("fit") & pl.col("SampleID").is_in(failures["SampleID"].implode()))
    if fit.height:
        raise ValueError(f"{fit.height} technically failed tubes are fit in the identity frame")
    pqtl_scores = getattr(qc_instance, "pqtl_scores", None)
    pqtl_evidence = getattr(qc_instance, "pqtl_evidence", None)
    relatives = (
        relative_pairs(pqtl_evidence, config.pqtl_relative_kinship)
        if pqtl_scores is not None and pqtl_evidence is not None
        else frozenset()
    )
    return unfit_identity(
        scored,
        failures,
        inp.adsl,
        self_max=config.identity_self_max,
        margin_min=config.margin_min,
        relatives=relatives,
        pqtl_scores=pqtl_scores,
    )


def identity_check(qc_instance, config) -> pl.DataFrame:
    """Runs the full identity check and stores its results on ``qc_instance``.

    Runs ``identity_inputs``, then ``identity_from_inputs`` with
    ``qc_instance.pqtl_scores`` and ``.pqtl_evidence``. Stores the events on
    ``qc_instance.identity_events`` and the final scored frame, with the pass-1 best and
    second scores, on ``qc_instance.identity_proposals``. Then builds the unfit-tube
    annotation (``unfit_identity_for``) from the decision's outputs, never feeding back
    into them, on ``qc_instance.unfit_identity``.

    Returns:
        The identity events.
    """
    inp = identity_inputs(qc_instance)
    events, scored = identity_from_inputs(
        inp,
        config,
        getattr(qc_instance, "pqtl_scores", None),
        getattr(qc_instance, "pqtl_evidence", None),
        with_pass1=True,
    )
    qc_instance.identity_events = events
    qc_instance.identity_proposals = scored
    qc_instance.unfit_identity = unfit_identity_for(qc_instance, inp, scored, config)
    return events


def identity_from_inputs(
    inp: IdentityInputs,
    config,
    pqtl_scores: pl.DataFrame | None = None,
    pqtl_evidence=None,
    *,
    with_pass1: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Runs the identity stage on fixed inputs at ``config``.

    Scores Lp similarity (``lp_scored``), classifies the flagged samples
    (``propose_scored``, which rescores best and second subjects on anchors), and
    enriches the events with the sample flag each type sets (``identity_flag``), the
    scoring evidence, ADaM site, the pQTL evidence and the well position, in
    ``IDENTITY_EVENTS_SCHEMA``. Reads ``TRT01A``, ``USUBJID``, ``SEX`` and ``AGE`` from
    ADSL by those names; ``--group-col``, ``--sex-col`` and ``--age-col`` do not reach
    it.

    Args:
        inp: The stage's inputs (``identity_inputs``).
        config: The ``TrialConfig`` whose identity and pQTL keys apply.
        pqtl_scores: Per-tube pQTL scores (``schemas.PQTL_SCORES_SCHEMA``); None when
            pQTL is off.
        pqtl_evidence: The ``PqtlEvidence``; None when pQTL is off.
        with_pass1: Add the scores before the anchor rescoring to ``scored`` as
            ``pass1_best_subject``, ``pass1_best_z`` and ``pass1_second_z`` (the
            cross-platform step compares them; the decision does not read them).

    Returns:
        ``(events, scored)``: the events, and the frame ``propose_scored`` returns with
        the final scores.
    """
    adsl = inp.adsl
    scored, Wz = lp_scored(inp, config, pqtl_scores)
    p_male_col = None if inp.sex_oof is None else "Predicted_Male_Probability"
    age_col = None if inp.age_oof is None else "Predicted_Age"
    relatives = (
        relative_pairs(pqtl_evidence, config.pqtl_relative_kinship)
        if pqtl_scores is not None and pqtl_evidence is not None
        else frozenset()
    )
    rule = identity_rule(config)
    pass1 = scored.select(
        "SampleID",
        pl.col("best_subject").alias("pass1_best_subject"),
        pl.col("best_z").alias("pass1_best_z"),
        pl.col("second_z").alias("pass1_second_z"),
    )

    # Propose: rescores against anchors, so scored carries the final scores.
    events, scored = propose_scored(
        scored,
        adsl,
        rule,
        p_male_col,
        age_col,
        Wz,
        records=inp.records,
        relatives=relatives,
    )

    # Enrich with scoring evidence, metadata and plausibility columns.
    if events.height > 0:
        scored_join_cols = [
            "SampleID",
            "self_z",
            "best_z",
            "second_z",
            "USUBJID",
            "VISITNUM",
            "ADT",
        ]
        if "ADT_SOURCE" in scored.columns:  # the tube's own source, before any move
            scored_join_cols.append("ADT_SOURCE")
        oof_rename: dict[str, str] = {}
        if p_male_col is not None:
            scored_join_cols.append(p_male_col)
            oof_rename[p_male_col] = "p_male"
        if age_col is not None:
            scored_join_cols.append(age_col)
            oof_rename[age_col] = "pred_age"

        events = (
            events.join(
                scored.select(scored_join_cols),
                on="SampleID",
                how="left",
                maintain_order="left",
            )
            .rename({"USUBJID": "USUBJID_LABELED", "VISITNUM": "VISITNUM_LABELED"} | oof_rename)
            .with_columns(
                pl.col("event_type")
                .replace_strict(EVENT_FLAGS, return_dtype=pl.String)
                .alias("identity_flag"),
                (pl.col("best_z") - pl.col("second_z")).alias("margin"),
                pl.lit(config.study_name).alias("trial"),
                pl.lit(inp.platform_name).alias("platform"),
            )
            .drop("second_z")
        )

        # SITEID from ADSL on the labeled subject; same_site against the inferred one.
        if "SITEID" in adsl.columns:
            site = adsl.select("USUBJID", "SITEID").unique(subset=["USUBJID"]).sort("USUBJID")
            events = (
                events.join(
                    site.rename({"USUBJID": "USUBJID_LABELED"}),
                    on="USUBJID_LABELED",
                    how="left",
                    maintain_order="left",
                )
                .join(
                    site.rename({"USUBJID": "USUBJID_INFERRED", "SITEID": "_inf_site"}),
                    on="USUBJID_INFERRED",
                    how="left",
                    maintain_order="left",
                )
                .with_columns((pl.col("SITEID") == pl.col("_inf_site")).alias("same_site"))
                .drop("_inf_site")
            )

        # pQTL concordance (filled when the pqtl module has run): per-sample scores,
        # the sample's z against the inferred subject's genotype, and kinship between
        # the labeled and inferred subjects' genotypes. A tube with no finite score
        # has no evidence (call none).
        if pqtl_scores is not None:
            events = events.join(
                pqtl_scores.select(
                    "SampleID",
                    "pqtl_self_z",
                    "pqtl_best_z",
                    "pqtl_best_USUBJID",
                    "pqtl_call",
                    "dna_sex_concordant",
                ),
                on="SampleID",
                how="left",
                maintain_order="left",
            ).with_columns(pl.col("pqtl_call").fill_null("none"))
            events = _pqtl_pair_evidence(events, pqtl_evidence)

        # Well position (triage-only; never used for automated decisions).
        if inp.wells is not None:
            events = events.join(inp.wells, on="SampleID", how="left", maintain_order="left")

        # Fill any missing schema columns with typed nulls.
        for col_name, col_type in IDENTITY_EVENTS_SCHEMA.items():
            if col_name not in events.columns:
                events = events.with_columns(pl.lit(None).cast(col_type).alias(col_name))
        events = _cast_schema(events, IDENTITY_EVENTS_SCHEMA)
    else:
        events = _empty_events()

    if with_pass1:
        scored = scored.join(pass1, on="SampleID", how="left", maintain_order="left")
    return events, scored
