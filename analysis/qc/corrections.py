"""Resolves a trial's ``[[corrections]]`` against the ids in its data.

Corrections name subjects and samples by the SHA-256 hex digest of the id
(``config.Correction``). Resolution hashes every candidate id and needs exactly one
match; zero or several raise.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars as pl

# config imports sex_concordance and identity, which import this module.
if TYPE_CHECKING:
    from analysis.qc.config import Correction

logger = logging.getLogger(__name__)

# Types applied by build_manifest, so a built manifest already carries them
# (MANIFEST_EXCLUDE marks a built one).
BUILD_TIME_TYPES = (
    "rename",
    "subject_from_accession",
    "exclude",
    "reassign",
    "visit_set",
    "bridge_swap",
)


def digest(value: str) -> str:
    """Hashes an id to the SHA-256 hex digest written in the trial TOML."""
    return hashlib.sha256(value.encode()).hexdigest()


def resolve_digest(sha256: str, candidates: Iterable[str], what: str) -> str:
    """Finds the single candidate whose digest is ``sha256``.

    Args:
        sha256: The digest to match.
        candidates: The ids to hash and compare.
        what: The field and table, for the error (for example
            ``"usubjid_sha256 in ADSL"``). The error never names an id.

    Returns:
        The matching candidate.

    Raises:
        ValueError: If zero or several candidates match.
    """
    matches = sorted({c for c in candidates if c is not None and digest(c) == sha256})
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} matches for {what} {sha256[:12]}...; expected exactly 1")
    return matches[0]


def apply_subject_sets(
    corrections: Iterable[Correction], adsl: pl.LazyFrame, usubjid_col: str
) -> pl.LazyFrame:
    """Applies the ``subject_set`` corrections to ADSL.

    Each entry's subject is resolved by digest against ADSL. Other correction types are
    left to the manifest build.

    Args:
        corrections: The trial's corrections.
        adsl: ADSL.
        usubjid_col: Subject id column of ADSL.

    Returns:
        ADSL with the corrected values.

    Raises:
        ValueError: If an entry names a column ADSL lacks, or its values are already
            in ADSL (a stale correction).
    """
    entries = [c for c in corrections if c.type == "subject_set"]
    if not entries:
        return adsl
    frame = adsl.collect()
    subjects = frame[usubjid_col].cast(pl.Utf8).unique().to_list()
    for entry in entries:
        uid = resolve_digest(entry.usubjid_sha256, subjects, "usubjid_sha256 in ADSL")
        absent = sorted(set(entry.set_values) - set(frame.columns))
        if absent:
            raise ValueError(f"subject_set sets {', '.join(absent)}, which ADSL lacks")
        is_subject = pl.col(usubjid_col).cast(pl.Utf8) == uid
        current = frame.filter(is_subject).select(list(entry.set_values)).row(0, named=True)
        if current == entry.set_values:
            raise ValueError(
                f"subject_set for {entry.usubjid_sha256[:12]}... changes nothing: ADSL already "
                f"holds {entry.set_values} (stale correction)"
            )
        frame = frame.with_columns(
            pl.when(is_subject).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            for col, value in entry.set_values.items()
        )
    logger.info(f"Applied {len(entries)} subject_set corrections to ADSL")
    return frame.lazy()


# The expected protein sex column the sex checks compare with.
EXPECTED_SEX_COL = "Expected_Protein_Sex"
# Sample summary column: True for a tube whose subject has an expected_sex exception.
SEX_EXCEPTION_COL = "Sex_Exception"

# The recorded SEX an exception to the expected sex must have: the other of M and F.
OTHER_SEX = {"M": "F", "F": "M"}


def resolve_expected_sex(
    corrections: Iterable[Correction],
    adsl: pl.LazyFrame | pl.DataFrame,
    usubjid_col: str,
) -> pl.DataFrame | None:
    """Resolves the ``expected_sex`` exceptions against ADSL.

    ADSL is not changed: ``SEX`` stays as recorded. Each entry's subject is resolved by
    digest against ``adsl``. Errors never name an id.

    Args:
        corrections: The trial's corrections.
        adsl: ADSL.
        usubjid_col: Subject id column of ADSL.

    Returns:
        A frame of ``usubjid_col`` and ``EXPECTED_SEX_COL``, or None when the config has
        no such exception.

    Raises:
        ValueError: If a subject matches zero or several ADSL ids; an entry's expected
            sex equals the recorded ``SEX`` (a stale exception); ADSL has no ``SEX``;
            or the recorded ``SEX`` is anything but the other of M and F (null, blank,
            U: nothing to except).
    """
    entries = [c for c in corrections if c.type == "expected_sex"]
    if not entries:
        return None
    frame = adsl.lazy().collect()
    if "SEX" not in frame.columns:
        raise ValueError("expected_sex corrections need ADSL SEX, which ADSL lacks")
    subjects = frame[usubjid_col].cast(pl.Utf8).unique().to_list()
    rows = []
    for entry in entries:
        uid = resolve_digest(entry.usubjid_sha256, subjects, "usubjid_sha256 in ADSL")
        recorded = (
            frame.filter(pl.col(usubjid_col).cast(pl.Utf8) == uid)["SEX"]
            .cast(pl.Utf8)
            .unique()
            .to_list()
        )
        if recorded == [entry.expected_sex]:
            raise ValueError(
                f"expected_sex for {entry.usubjid_sha256[:12]}... changes nothing: ADSL SEX is already "
                f"{entry.expected_sex} (stale correction)"
            )
        if recorded != [OTHER_SEX[entry.expected_sex]]:
            raise ValueError(
                f"expected_sex for {entry.usubjid_sha256[:12]}... needs a recorded ADSL SEX of "
                f"{OTHER_SEX[entry.expected_sex]}, the other of M and F: an exception is an exception to a "
                f"recorded value, and ADSL records {recorded}"
            )
        rows.append((uid, entry.expected_sex))
    logger.info(
        f"{len(rows)} expected_sex exceptions: ADSL SEX kept as recorded, the sex checks read the exception"
    )
    return pl.DataFrame(
        rows, schema={usubjid_col: pl.Utf8, EXPECTED_SEX_COL: pl.Utf8}, orient="row"
    )


def with_expected_sex(
    frame: pl.DataFrame,
    expected: pl.DataFrame | None,
    usubjid_col: str,
    sex_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Adds each subject's expected protein sex to a frame.

    Args:
        frame: Rows with ``usubjid_col`` and ``sex_col``.
        expected: ``resolve_expected_sex`` output keyed by ``usubjid_col``; None adds
            ``sex_col`` unchanged.
        usubjid_col: Subject id column.
        sex_col: Recorded sex column.
        out_col: Name of the new column: the subject's exception in ``expected``, else
            ``sex_col``.

    Returns:
        ``frame`` with ``out_col``.
    """
    if expected is None:
        return frame.with_columns(pl.col(sex_col).alias(out_col))
    lookup = dict(expected.select(usubjid_col, EXPECTED_SEX_COL).iter_rows())
    return frame.with_columns(
        pl.col(usubjid_col)
        .cast(pl.Utf8)
        .replace_strict(lookup, default=pl.col(sex_col), return_dtype=frame.schema[sex_col])
        .alias(out_col)
    )
