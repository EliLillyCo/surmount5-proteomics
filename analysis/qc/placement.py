"""Decides whether a tube may move to another subject, and into which visit.

A tube that identity names as belonging to another subject moves only if that subject
has its own ADaM record at exactly one visit dated within ``tolerance_days`` of the
tube's draw date. There is no fallback to the label's visit code and no wider search.
``place`` is pure: it reads proposal rows and the destination's visit records and
returns, per row, the visit, the reason it could not be placed and the gap in days to
the destination's nearest record.

Reasons (``unplaced_reason``), checked in this order:

- ``no_date``: the tube has no draw date.
- ``label_date``: the tube's ADT was filled from the labeled subject's own visit
  record (``ADT_SOURCE`` of ``adlb_visit`` or ``advs``), so it is no evidence of the
  draw.
- ``no_record``: no record of the destination within the tolerance.
- ``multi_visit``: records of several visits within the tolerance. The exception is
  when every record in range is dated exactly on the tube's date and exactly one of the
  visits is scheduled (an integer below 100): that visit is taken, as the manifest
  build does for an unscheduled code.
- ``accession``: the destination's record at the placed visit carries a tube reference
  that matches neither this tube's accession nor the partner tube's. A record with no
  reference passes.
"""

from __future__ import annotations

import logging

import polars as pl

from analysis.qc.manifest import accession_matches, accession_tokens
from analysis.qc.utils import ensure_date_column

logger = logging.getLogger(__name__)

# ADT_SOURCE values that copy a date from the labeled subject's own visit record.
LABEL_FILLED_SOURCES = ("adlb_visit", "advs")


def _scheduled(visit: pl.Expr) -> pl.Expr:
    """Flags a scheduled visit code: an integer below 100."""
    return (visit == visit.floor()) & (visit < 100)


def _gaps(base: pl.DataFrame, target_col: str, records: pl.DataFrame) -> pl.DataFrame:
    """Lists every record of each row's target with its gap in days to the tube.

    Args:
        base: Rows with ``_row``, ``target_col`` and ``_adt`` (Date).
        target_col: Column holding the destination USUBJID.
        records: The destination's dated records.

    Returns:
        ``_row``, ``VISITNUM``, ``REF`` and ``_gap`` for each record.
    """
    return (
        base.select("_row", pl.col(target_col).alias("USUBJID"), "_adt")
        .join(
            records.select("USUBJID", "VISITNUM", pl.col("ADT").alias("_rec_adt"), "REF"),
            on="USUBJID",
            how="inner",
        )
        .with_columns((pl.col("_adt") - pl.col("_rec_adt")).dt.total_days().abs().alias("_gap"))
        .filter(pl.col("_gap").is_not_null())
    )


def date_gap(rows: pl.DataFrame, target_col: str, records: pl.DataFrame) -> pl.DataFrame:
    """Adds the day gap from each tube's ``ADT`` to the target's nearest record.

    The gap is to any visit of the subject in ``target_col``. It is null when the row
    has no target, no ``ADT``, or the target has no dated record.

    Args:
        rows: One row per tube, with ``ADT``.
        target_col: Column holding the destination USUBJID.
        records: The destination's dated records.

    Returns:
        ``rows`` with ``date_gap_days`` (Int64).
    """
    base = rows.with_row_index("_row").with_columns(ensure_date_column(rows, "ADT").alias("_adt"))
    gap = (
        _gaps(base, target_col, records)
        .group_by("_row")
        .agg(pl.col("_gap").min().cast(pl.Int64).alias("date_gap_days"))
    )
    return base.join(gap, on="_row", how="left", maintain_order="left").select(
        *rows.columns, "date_gap_days"
    )


def place(
    rows: pl.DataFrame,
    target_col: str,
    records: pl.DataFrame,
    tolerance_days: int,
    partner_acc: str | None = None,
    acc_col: str = "AccessionNumber",
) -> pl.DataFrame:
    """Places each tube in a visit of its destination subject, or says why not.

    Args:
        rows: One row per tube to place, with ``ADT`` (Date, or ISO strings), the
            ``target_col`` column and, when the manifest was built, ``ADT_SOURCE``.
            Without ``ADT_SOURCE`` every ADT is read as the tube's own date and one
            WARNING line counts the placements that relied on it.
        target_col: Column holding the destination USUBJID.
        records: The destination's dated records (``manifest.visit_records``), with
            ``USUBJID``, ``VISITNUM``, ``ADT`` and ``REF``.
        tolerance_days: Whole days, 0 or more
            (``identity_reassign_date_tolerance_days``).
        partner_acc: Column holding the accession of the tube leaving the destination's
            slot in the same reciprocal move, null when there is none.
        acc_col: Column holding this tube's accession; absent means no accession.

    Returns:
        ``rows`` with ``_inf_visit`` (Float64, the placed visit, null when not
        placed), ``unplaced_reason`` and ``date_gap_days`` (Int64, the minimum gap to
        any record of the destination whatever the outcome, null when it has no dated
        record).

    Raises:
        ValueError: If ``tolerance_days`` is not an integer >= 0, ``rows`` lacks the
            target or ``ADT`` column, or some row has an ``ADT`` but no
            ``ADT_SOURCE``.
    """
    if (
        isinstance(tolerance_days, bool)
        or not isinstance(tolerance_days, int)
        or tolerance_days < 0
    ):
        raise ValueError(f"tolerance_days must be an integer >= 0; got {tolerance_days!r}")
    missing = [c for c in (target_col, "ADT") if c not in rows.columns]
    if missing:
        raise ValueError(f"place: rows lack {', '.join(missing)}")
    has_source = "ADT_SOURCE" in rows.columns
    if (
        has_source
        and rows.filter(pl.col("ADT").is_not_null() & pl.col("ADT_SOURCE").is_null()).height
    ):
        raise ValueError("place: some rows have an ADT but no ADT_SOURCE")

    base = rows.with_row_index("_row").with_columns(ensure_date_column(rows, "ADT").alias("_adt"))
    near = _gaps(base, target_col, records)
    gap = near.group_by("_row").agg(pl.col("_gap").min().cast(pl.Int64).alias("date_gap_days"))
    within = near.filter(pl.col("_gap") <= tolerance_days)

    per_row = within.group_by("_row").agg(
        pl.col("VISITNUM").unique().sort().alias("_visits"),
        pl.col("VISITNUM").filter(pl.col("_gap") == 0).unique().sort().alias("_exact_visits"),
    )
    per_row = per_row.with_columns(
        pl.col("_visits").list.len().alias("_n_visits"),
        pl.col("_visits").list.eval(pl.element().filter(_scheduled(pl.element()))).alias("_sched"),
    ).with_columns(
        # Same-day rule: only when every record in range is on the tube's date.
        pl.when(pl.col("_n_visits") == 1)
        .then(pl.col("_visits").list.first())
        .when(
            (pl.col("_n_visits") > 1)
            & (pl.col("_exact_visits").list.len() == pl.col("_n_visits"))
            & (pl.col("_sched").list.len() == 1)
        )
        .then(pl.col("_sched").list.first())
        .alias("_visit")
    )
    chosen = per_row.select("_row", "_n_visits", "_visit")

    # References of the records at the placed visit.
    refs = (
        within.join(chosen.select("_row", "_visit"), on="_row")
        .filter((pl.col("VISITNUM") == pl.col("_visit")) & pl.col("REF").is_not_null())
        .select("_row", "REF")
        .unique()
    )
    tube = base.select(
        "_row",
        (pl.col(acc_col) if acc_col in base.columns else pl.lit(None, pl.String)).alias("_acc"),
        (pl.col(partner_acc) if partner_acc is not None else pl.lit(None, pl.String)).alias(
            "_partner"
        ),
    )
    matched = (
        accession_matches(
            pl.concat(
                [
                    accession_tokens(tube.select("_row", pl.col("_acc").alias("ACC")), "_row"),
                    accession_tokens(tube.select("_row", pl.col("_partner").alias("ACC")), "_row"),
                ]
            ),
            refs,
            "_row",
        )
        .select("_row")
        .unique()
        .with_columns(pl.lit(True).alias("_ref_ok"))
    )
    has_ref = refs.select("_row").unique().with_columns(pl.lit(True).alias("_has_ref"))

    out = (
        base.join(gap, on="_row", how="left", maintain_order="left")
        .join(chosen, on="_row", how="left", maintain_order="left")
        .join(has_ref, on="_row", how="left", maintain_order="left")
        .join(matched, on="_row", how="left", maintain_order="left")
    )
    label_filled = (
        pl.col("ADT_SOURCE").is_in(list(LABEL_FILLED_SOURCES)).fill_null(False)
        if has_source
        else pl.lit(False)
    )
    n_visits = pl.col("_n_visits").fill_null(0)
    reason = (
        pl.when(pl.col("_adt").is_null())
        .then(pl.lit("no_date"))
        .when(label_filled)
        .then(pl.lit("label_date"))
        .when(n_visits == 0)
        .then(pl.lit("no_record"))
        .when(pl.col("_visit").is_null())
        .then(pl.lit("multi_visit"))
        .when(pl.col("_has_ref").fill_null(False) & ~pl.col("_ref_ok").fill_null(False))
        .then(pl.lit("accession"))
    )
    out = out.with_columns(reason.alias("unplaced_reason")).with_columns(
        pl.when(pl.col("unplaced_reason").is_null())
        .then(pl.col("_visit"))
        .cast(pl.Float64)
        .alias("_inf_visit")
    )
    if not has_source:
        n_relied = out.filter(pl.col("unplaced_reason").is_null()).height
        logger.warning(
            "placement: the manifest has no ADT_SOURCE, so %d placement(s) read the ADT as the "
            "tube's own date",
            n_relied,
        )
    return out.select(*rows.columns, "_inf_visit", "unplaced_reason", "date_gap_days")
