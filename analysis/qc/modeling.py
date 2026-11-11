"""Feature matrices and grouped CV splits shared by the sex and age checks."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from analysis.qc.flags import FAIL_DUPLICATE_ASSAY
from analysis.qc.utils import impute_missing_values, wide_frame

logger = logging.getLogger(__name__)

# Defaults of TrialConfig ``model_cv_folds`` and ``model_knn_k``: the grouped CV folds
# of the sex and age models, and the neighbors of the KNN imputation of their
# matrices.
CV_FOLDS: int = 5
KNN_K: int = 5


@dataclass
class FeatureMatrix:
    """A samples x proteins matrix with its sample, subject and feature labels.

    Attributes:
        X: Imputed values, one row per sample.
        sample_ids: Sample id of each row.
        groups: Subject id of each row.
        feature_cols: Protein ids of the columns of ``X``.
        frame: The unimputed wide table joined to the labels.
        fit: True for the rows the models may train on.
    """

    X: np.ndarray
    sample_ids: np.ndarray
    groups: np.ndarray
    feature_cols: list[str]
    frame: pl.DataFrame
    fit: np.ndarray


def duplicate_assay_ids(qc_instance) -> list[str]:
    """Lists the assays flagged ``FAIL_Duplicate_Assay``, the copies of a kept assay."""
    pid = qc_instance.protein_id_col
    return sorted(
        {
            p
            for df in qc_instance.protein_qc_actions
            if FAIL_DUPLICATE_ASSAY in df.columns
            for p in df.filter(pl.col(FAIL_DUPLICATE_ASSAY))[pid]
        }
    )


def build_feature_matrix(
    qc_instance,
    labels: pl.DataFrame,
    label_cols: list[str],
    protein_subset: list[str],
    max_missing: float = 0.1,
    fit_exclude=(),
    knn_k: int = KNN_K,
) -> FeatureMatrix:
    """Builds the samples x proteins matrix of ``protein_subset``, joined to ``labels``.

    Failed samples are dropped. Every assay is used whatever its assay-QC status, except
    ``FAIL_Duplicate_Assay`` copies; an assay missing in more than ``max_missing`` of
    the samples is dropped. Duplicate rows per sample and protein are averaged; rows are
    sorted by subject then sample; missing values are KNN-imputed, with only the fit
    samples as donors.

    Args:
        qc_instance: An OlinkQC instance.
        labels: One row per sample with the sample id, USUBJID and ``label_cols``.
        label_cols: Non-protein columns carried in ``frame`` (excluded from ``X``).
        protein_subset: Protein ids to keep.
        max_missing: Largest fraction of samples an assay may lack.
        fit_exclude: Sample ids the models must not train on (the samples kept out of
            the outlier fits). They stay in the matrix and are predicted; ``fit`` marks
            the rest.
        knn_k: Neighbors of the KNN imputation.

    Returns:
        The feature matrix.

    Raises:
        ValueError: If fewer than 10 samples have labels, no assay is missing in at
            most ``max_missing`` of the samples, or every sample is outside the fit.
    """
    sid, pid, val = (
        qc_instance.sample_id_col,
        qc_instance.protein_id_col,
        qc_instance.data_col,
    )
    usubjid = qc_instance.usubjid_col
    data = qc_instance.samples
    all_ids = set(data.select(sid).unique().collect()[sid].to_list())
    failed_ids = all_ids & set(qc_instance.failed_samples or [])
    if failed_ids:
        data = data.filter(~pl.col(sid).is_in(pl.Series(sorted(failed_ids)).implode()))
        logger.info(f"excluding {len(failed_ids)} failed samples")
    duplicates = duplicate_assay_ids(qc_instance)
    if duplicates:
        data = data.filter(~pl.col(pid).is_in(pl.Series(duplicates, dtype=pl.String).implode()))
        logger.info(f"Excluding {len(duplicates)} duplicate-assay {pid}s")
    data = data.filter(pl.col(pid).is_in(pl.Series(protein_subset, dtype=pl.String).implode()))
    labeled_ids = set(labels[sid].to_list())
    kept_ids = all_ids - failed_ids
    missing_label_ids = kept_ids - labeled_ids
    wide = (
        data.filter(
            pl.col(sid).is_in(pl.Series(sorted(labeled_ids)).implode())
            & pl.col(val).is_not_null()
            & ~pl.col(val).is_nan()
        )
        .select([sid, pid, val])
        .group_by([sid, pid])
        .agg(pl.col(val).mean())
        .collect()
    )
    wide = wide_frame(wide, index=sid, on=pid, values=val)
    no_panel_ids = (kept_ids & labeled_ids) - set(wide[sid].to_list())
    logger.info(
        f"build_feature_matrix dropped samples: failed={len(failed_ids)}, "
        f"missing_label={len(missing_label_ids)}, no_panel_values={len(no_panel_ids)}"
    )
    frame = wide.join(labels, on=sid, how="inner", validate="1:1")
    if len(frame) < 10:
        raise ValueError(f"Insufficient samples with labels: {len(frame)} < 10")
    frame = frame.sort([usubjid, sid])
    excluded_cols = {sid, usubjid, *label_cols}
    feature_cols = sorted(c for c in frame.columns if c not in excluded_cols)
    missing = {c: frame[c].null_count() / frame.height for c in feature_cols}
    sparse = [c for c in feature_cols if missing[c] > max_missing]
    if sparse:
        logger.info(
            f"dropping {len(sparse)} {pid}s missing in more than {max_missing:.0%} of samples"
        )
        feature_cols = [c for c in feature_cols if c not in set(sparse)]
    if not feature_cols:
        raise ValueError(
            f"No assay is missing in at most {max_missing:.0%} of {frame.height} samples"
        )
    fit_excluded = set(fit_exclude)
    fit = np.array([s not in fit_excluded for s in frame[sid].to_list()])
    if not fit.any():
        raise ValueError("Every sample is outside the outlier fit; nothing to train on")
    logger.info(
        f"{int((~fit).sum())} samples outside the outlier fit are predicted but not trained on"
    )
    X = impute_missing_values(frame.select(feature_cols).to_numpy(), donors=fit, n_neighbors=knn_k)
    return FeatureMatrix(
        X=X,
        sample_ids=frame[sid].to_numpy(),
        groups=frame[usubjid].to_numpy(),
        feature_cols=feature_cols,
        frame=frame,
        fit=fit,
    )


# UniProt accession remap applied before matching: an accession -> its current accession.
RETIRED_UNIPROT = {"P04745": "P0DUB6"}


def explode_assay_uniprot(
    assays: pl.DataFrame, protein_id_col: str, uniprot_col: str
) -> pl.DataFrame:
    """Splits multi-UniProt assays into one row per assay and UniProt.

    Separators are ``_``, ``;``, ``,``, ``|`` and whitespace, and retired accessions are
    remapped (``RETIRED_UNIPROT``). Rows are sorted by protein id then UniProt and
    de-duplicated, so "first match wins" is deterministic.

    Args:
        assays: Assay table.
        protein_id_col: Assay id column.
        uniprot_col: UniProt column, possibly with several accessions per row.

    Returns:
        The assay id and single UniProt columns.
    """
    return (
        assays.select(protein_id_col, uniprot_col)
        .drop_nulls()
        .with_columns(pl.col(uniprot_col).str.replace_all(r"[_;,|\s]+", ";").str.split(";"))
        .explode(uniprot_col)
        .with_columns(pl.col(uniprot_col).str.strip_chars().replace(RETIRED_UNIPROT))
        .filter(pl.col(uniprot_col) != "")
        .sort(protein_id_col, uniprot_col)
        .unique(maintain_order=True)
    )


def _subject_digest(subject: str, seed: int, salt: str) -> int:
    """Hashes (seed, salt, subject) to 64 bits, fixed across hosts and hash seeds."""
    key = f"{seed}\x1f{salt}\x1f{subject}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def subject_folds(subjects, n_folds: int, seed: int = 0, salt: str = "outer") -> dict[str, int]:
    """Assigns each subject a fold from a stable hash of its id.

    The fold is ``hash(seed, salt, id) mod n_folds``, so it depends on the subject's
    own id only: adding, removing or relabelling one subject moves no other subject.
    Different ``salt`` values give independent partitions (outer, inner and identity
    folds). When the hash leaves a fold empty (only with a handful of subjects),
    subjects are instead ranked by ``(digest, id)`` and dealt round-robin, which is
    still host-independent.

    Args:
        subjects: Subject ids (repeats allowed).
        n_folds: Number of folds.
        seed: Hash seed.
        salt: Partition name.

    Returns:
        Subject id to fold.

    Raises:
        ValueError: If there are fewer subjects than folds.
    """
    ids = sorted({str(x) for x in subjects})
    if len(ids) < n_folds:
        raise ValueError(f"Insufficient participants: {len(ids)} < {n_folds}")
    digest = {x: _subject_digest(x, seed, salt) for x in ids}
    folds = {x: digest[x] % n_folds for x in ids}
    if len(set(folds.values())) < n_folds:
        ranked = sorted(ids, key=lambda x: (digest[x], x))
        folds = {x: i % n_folds for i, x in enumerate(ranked)}
    return folds


def grouped_splits(
    groups: np.ndarray, n_splits_max: int = CV_FOLDS, seed: int = 0, salt: str = "outer"
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Builds grouped k-fold splits on the group-sorted rows.

    Sorting first makes the splits independent of input row order. Each group's fold
    comes from ``subject_folds``, so it does not depend on the other groups, the host
    or the hash seed. Nested splits use a different ``salt`` from the outer ones.

    Args:
        groups: Group (subject) id of each row.
        n_splits_max: Folds, at most; fewer groups give fewer.
        seed: Hash seed.
        salt: Partition name.

    Returns:
        The row order that sorts by group, and the (train, test) row indices of each
        fold on the sorted rows.

    Raises:
        ValueError: If there are fewer than two groups.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits_max, n_groups)
    if n_splits < 2:
        raise ValueError(f"Insufficient participants: {n_groups} < 2")
    logger.info(
        f"Using {n_splits}-fold grouped cross-validation ({n_groups} unique participants in training set)"
    )
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    fold_of = subject_folds(sorted_groups, n_splits, seed, salt)
    row_fold = np.array([fold_of[str(g)] for g in sorted_groups])
    splits = [
        (np.flatnonzero(row_fold != k), np.flatnonzero(row_fold == k)) for k in range(n_splits)
    ]
    return order, splits
