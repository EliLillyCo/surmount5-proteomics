"""Builds the sample manifest from the raw vendor key and ADaM.

``build_manifest`` reads the vendor key named in the trial TOML's ``[manifest]`` block
(``[manifest.columns]`` maps logical names to the key's columns; there is no per-trial
code), links each tube to its tube record (SDTM LB, or ADaM ADLB and ADIG), assigns
``VISITNUM`` and ``ADT``, applies the manifest corrections, and writes
``manifest.parquet`` plus a counts-only build report.

Every key row becomes one manifest row, in key order. A row is a control (sample type
``NEG_CTRL``), a bridge row (a ``bridge_flag`` value, an empty subject, or a subject
matching ``bridge_subject_pattern``) or a study row; only study rows get a subject, an
accession, a visit and a date. ``bridge_swap`` corrections apply first. ADT is the first
available of the tube record's date, the vendor date, the ADLB date and the ADVS date at
the tube's visit.

FAILs that a later step acts on (null ADT, a null visit, a subject not in ADSL) are
counted in the report and the row is kept. FAILs that would make the manifest wrong
raise: a duplicate SampleID, an uncovered owner mismatch, an ambiguous tube link and a
stale correction. ``visit_records``, ``accession_tokens`` and ``accession_matches``
give identity the same readings of the ADaM records.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime
from pathlib import Path

import openpyxl
import polars as pl

from analysis.qc.config import (
    Correction,
    ManifestColumns,
    ManifestConfig,
    TrialConfig,
)
from analysis.qc.corrections import resolve_digest
from analysis.qc.flags import (
    FAIL_NO_ADSL_SUBJECT,
    FAIL_NULL_ADT,
    FAIL_TWO_SCHEDULED_VISITS_SAME_DATE,
    WARN_ADLB_VISIT_MULTIPLE_DATES,
    WARN_ADT_FILL_ADLB_VISIT,
    WARN_ADT_FILL_ADVS,
    WARN_ADVS_VISIT_MULTIPLE_DATES,
    WARN_DATE_MATCH_NO_HIT,
    WARN_DATE_MATCH_UNSCHEDULED_ONLY,
    WARN_DUPLICATE_SUBJECT_VISIT,
    WARN_PLANNED_VISIT_DIFFERS,
    WARN_SITE_TRANSFER,
    WARN_TUBE_ADLB_VISIT_DATE_DISAGREE,
    WARN_TUBE_RECORDS_DISAGREE,
    WARN_VENDOR_DATE_UNPARSED,
    WARN_VENDOR_TUBE_DATE_DISAGREE,
    WARN_VISIT_DATE_ORDER,
    WARN_VISIT_LABEL_UNPARSED,
)
from analysis.qc.utils import reject_short_years

MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "SampleID": pl.Utf8,
    "USUBJID": pl.Utf8,
    "BridgeSample": pl.Boolean,
    "SampleType": pl.Utf8,
    "AccessionNumber": pl.Utf8,
    "VISITNUM": pl.Float64,
    "VISITNUM_PLANNED": pl.Float64,
    "VISITNUM_SOURCE": pl.Utf8,
    "ADT": pl.Date,
    "ADT_SOURCE": pl.Utf8,
    "ADT_VENDOR": pl.Date,
    "MANIFEST_EXCLUDE": pl.Utf8,
    "MANIFEST_CORRECTION": pl.Utf8,
}

# Columns a built manifest carries only when the config asks for them: ``VOLUME_NOTE``
# is the key's sample-volume note (``[manifest.columns] volume_note``), read by sample
# QC's WARN_Low_Volume.
OPTIONAL_MANIFEST_COLUMNS: dict[str, pl.DataType] = {"VOLUME_NOTE": pl.Utf8}

# ADT_SOURCE values whose date is the labeled visit's own lab date, not a draw date; a
# visit the date match returns for such a tube is the label's, and VISITNUM_SOURCE says
# so.
LABEL_DATE_SOURCES = ("adlb_visit", "advs")
VENDOR_LABEL_DATE = "vendor_label_date"
NULL_TOKENS = ["NA", "N/A", "#N/A"]
CONTROL_SAMPLE_TYPES = ["NEG_CTRL"]
LIMS_SAMPLE_ID = "Container Barcode"
# Tube-record source -> the column holding the tube's accession (CDISC --REFID), in the
# order the sources are consulted.
TUBE_REF_COLUMNS = {"lb": "LBREFID", "adlb": "LBREFID", "adig": "ISREFID"}
VENDOR_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%d/%b/%Y")
# Corrections the build applies; QC applies subject_set to ADSL.
MANIFEST_CORRECTIONS = (
    "rename",
    "subject_from_accession",
    "exclude",
    "reassign",
    "visit_set",
    "bridge_swap",
)
# Key-row fields a bridge_swap exchanges between two rows; SampleID and sample type stay.
SWAP_FIELDS = ("_kind", "_subj", "ACC", "_label", "_date_raw", "_note")

_VISIT_LABEL = r"^[Vv]?0*(\d+(?:\.\d+)?)$"
_NOTED_VISIT = r"\bV0*(\d+(?:\.\d+)?)\b"


# ---------------------------------------------------------------- reading


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _xlsx_rows(path: str) -> list[list[str]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [[_cell(v) for v in row] for row in wb.worksheets[0].iter_rows(values_only=True)]
    wb.close()
    return rows


def _csv_rows(path: str, header_cell: str) -> list[list[str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    for sep in ("\t", ","):
        # A newline-preserving reader: a quoted newline stays in its cell, and only
        # CR/LF end a record.
        rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=sep)]
        if any(header_cell in r for r in rows):
            return rows
    return rows


def _unique_names(header: list[str]) -> list[str]:
    names, seen = [], {}
    for i, name in enumerate(header):
        name = name or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_duplicated_{seen[name] - 1}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def _read_table(path: str, header_cell: str) -> pl.DataFrame:
    """Reads a vendor key or LIMS table with every cell as Utf8.

    Cells are stripped. The text separator is a tab or a comma, whichever finds the
    header. A repeated header name gets the suffix ``_duplicated_<n>``, as in
    ``polars.read_csv``.

    Args:
        path: An ``.xlsx`` workbook (first sheet) or a tab- or comma-separated text
            file.
        header_cell: A cell of the header row; the header is the first row holding it.

    Returns:
        The rows below the header, one Utf8 column per header cell; empty cells and NA
        tokens are null, and a short row is padded with empty cells.

    Raises:
        ValueError: If no row holds ``header_cell``, or a row has a non-empty cell
            beyond the header's width (an unquoted delimiter inside a cell shifts every
            later cell). Trailing empty cells are allowed.
    """
    rows = _xlsx_rows(path) if path.lower().endswith(".xlsx") else _csv_rows(path, header_cell)
    idx = next((i for i, r in enumerate(rows) if header_cell in r), None)
    if idx is None:
        raise ValueError(f"{Path(path).name}: no row holds the header {header_cell!r}")
    names = _unique_names(rows[idx])
    wide = [
        i for i, r in enumerate(rows[idx + 1 :], start=1) if any(c != "" for c in r[len(names) :])
    ]
    if wide:
        raise ValueError(
            f"{Path(path).name}: {len(wide)} row{' has' if len(wide) == 1 else 's have'} more cells than the "
            f"header ({len(names)}), first at data row {wide[0]}; quote any cell that holds the delimiter"
        )
    data = [(r + [""] * len(names))[: len(names)] for r in rows[idx + 1 :]]
    frame = pl.DataFrame(data, schema={n: pl.Utf8 for n in names}, orient="row")
    return frame.with_columns(
        pl.when(pl.col(n).is_in(["", *NULL_TOKENS])).then(None).otherwise(pl.col(n)).alias(n)
        for n in names
    )


def _read_key(m: ManifestConfig) -> tuple[pl.DataFrame, dict]:
    """Reads the vendor key, joined to the LIMS file when one is set.

    The LIMS file joins on ``Container Barcode``; a LIMS column whose name the key
    already uses takes the suffix ``_lims``.

    Returns:
        The key without its empty rows and rows repeating the header, and the counts of
        each dropped kind.

    Raises:
        ValueError: If the LIMS file repeats a barcode or ``[manifest.columns]`` names a
            column the key lacks.
    """
    c = m.columns
    key = _read_table(m.raw_key, c.sample_id)
    if m.lims:
        lims = _read_table(m.lims, LIMS_SAMPLE_ID).filter(pl.col(LIMS_SAMPLE_ID).is_not_null())
        n_dup = lims.height - lims[LIMS_SAMPLE_ID].n_unique()
        if n_dup:
            raise ValueError(f"{Path(m.lims).name}: {n_dup} repeated {LIMS_SAMPLE_ID} values")
        key = key.join(
            lims,
            left_on=c.sample_id,
            right_on=LIMS_SAMPLE_ID,
            how="left",
            suffix="_lims",
            maintain_order="left",
        )
    configured = [v for v in vars(c).values() if v is not None]
    missing = [v for v in configured if v not in key.columns]
    if missing:
        raise ValueError(
            f"{Path(m.raw_key).name}: [manifest.columns] names columns not in the key: "
            + ", ".join(repr(v) for v in missing)
        )
    empty = pl.all_horizontal(pl.all().is_null())
    repeated = pl.col(c.sample_id) == c.sample_id
    if c.bridge_flag:
        repeated = repeated | (pl.col(c.bridge_flag) == c.bridge_flag)
    repeated = repeated.fill_null(False)
    counts = {
        "dropped_empty": key.filter(empty).height,
        "dropped_repeated_header": key.filter(repeated).height,
    }
    return key.filter(~empty & ~repeated), counts


# ---------------------------------------------------------------- ADaM


def _norm_subjid(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.strip_chars().str.replace(r"^0+(\d)", "${1}")


def _as_date(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.Date:
        return col
    if isinstance(dtype, pl.Datetime):
        return col.dt.date()
    if dtype == pl.Utf8:
        return reject_short_years(
            col.str.strip_chars().replace("", None).str.slice(0, 10).str.to_date("%Y-%m-%d")
        )
    raise ValueError(f"ADT has unsupported type {dtype}")


def _lab(path: str, ref: str | None = None) -> pl.LazyFrame:
    """Reads USUBJID, VISITNUM, ADT and the ``ref`` column as REF from an ADaM file.

    Only rows with an empty DTYPE are read.
    """
    return _lab_frame(pl.scan_parquet(path), ref)


def _empty_dtype(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the recorded rows of an ADaM frame: those with an empty ``DTYPE``."""
    if "DTYPE" not in lf.collect_schema():
        return lf
    return lf.filter(
        pl.col("DTYPE").is_null() | (pl.col("DTYPE").cast(pl.Utf8).str.strip_chars() == "")
    )


def _lab_frame(lf: pl.LazyFrame, ref: str | None = None) -> pl.LazyFrame:
    """Reads a frame as ``_lab`` reads a file; shared by ADLB, ADIG and ADVS."""
    schema = lf.collect_schema()
    lf = _empty_dtype(lf)
    cols = [
        pl.col("USUBJID").cast(pl.Utf8),
        pl.col("VISITNUM").cast(pl.Float64),
        _as_date("ADT", schema["ADT"]).alias("ADT"),
    ]
    if ref is not None:
        cols.append(pl.col(ref).cast(pl.Utf8).str.strip_chars().replace("", None).alias("REF"))
    return lf.select(cols)


def _visit_dates(path: str) -> pl.LazyFrame:
    """Returns the sorted distinct dates per (USUBJID, VISITNUM)."""
    return (
        _lab(path)
        .drop_nulls(["USUBJID", "VISITNUM", "ADT"])
        .group_by("USUBJID", "VISITNUM")
        .agg(pl.col("ADT").unique().sort().alias("DATES"))
    )


def _scheduled(v: pl.Expr) -> pl.Expr:
    """Returns whether a visit code is scheduled: an integer below 100."""
    return (v == v.floor()) & (v < 100)


def _ref_column(frame: pl.LazyFrame) -> pl.Expr:
    """Returns ``REF``, or all null when the dataset has no tube reference column."""
    return (
        pl.col("REF") if "REF" in frame.collect_schema() else pl.lit(None, pl.String).alias("REF")
    )


def visit_records(
    adlb: pl.LazyFrame,
    advs: pl.LazyFrame | None,
    adig: pl.LazyFrame | None = None,
    lb: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Returns the subjects' dated visit records, one row per distinct record.

    The primary records are the tube-record sources the build consults
    (``_tube_records``): LB and ADLB with ``lb``, else ADLB and ADIG. LB is read as
    ``_lb_records`` reads it, but only rows carrying an ``LBREFID`` count: a row without
    one is skipped.
    ADaM rows count with an empty DTYPE. ADVS rows are added only for (USUBJID,
    VISITNUM) pairs with no primary record. Rows missing a subject, a visit or a date
    are dropped.

    Args:
        adlb: ADaM ADLB.
        advs: ADaM ADVS, or None (a run on a mapping manifest reads ADLB only).
        adig: ADaM ADIG, or None; must be None when ``lb`` is given.
        lb: SDTM LB, or None.

    Returns:
        USUBJID, VISITNUM (Float64), ADT (Date), REF (the tube reference, null when the
        dataset has none) and SOURCE (``lb``, ``adlb``, ``adig`` or ``advs``), sorted.

    Raises:
        ValueError: If both ``lb`` and ``adig`` are given.
    """
    if lb is not None and adig is not None:
        raise ValueError("visit_records reads ADIG only without LB, as the manifest build does")
    primary = []
    if lb is not None:
        frame = _lb_frame(lb, "lb").filter(pl.col("REF").is_not_null())
        primary.append(frame.with_columns(pl.lit("lb").alias("SOURCE")))
    for source, lf, ref in (("adlb", adlb, "LBREFID"), ("adig", adig, "ISREFID")):
        if lf is not None:
            frame = _lab_frame(lf, ref if ref in lf.collect_schema() else None)
            primary.append(frame.with_columns(_ref_column(frame), pl.lit(source).alias("SOURCE")))
    parts = [pl.concat(primary, how="vertical").drop_nulls(["USUBJID", "VISITNUM", "ADT"])]
    if advs is not None:
        vs = (
            _lab_frame(advs)
            .with_columns(pl.lit(None, pl.String).alias("REF"), pl.lit("advs").alias("SOURCE"))
            .drop_nulls(["USUBJID", "VISITNUM", "ADT"])
            .join(
                parts[0].select("USUBJID", "VISITNUM").unique(),
                on=["USUBJID", "VISITNUM"],
                how="anti",
            )
        )
        parts.append(vs)
    return (
        pl.concat(parts, how="vertical")
        .select("USUBJID", "VISITNUM", "ADT", "REF", "SOURCE")
        .unique()
        .sort("USUBJID", "VISITNUM", "ADT", "SOURCE", "REF", nulls_last=True)
    )


# ---------------------------------------------------------------- tube links


def accession_tokens(samples: pl.DataFrame, key: str, acc: str = "ACC") -> pl.DataFrame:
    """Returns one row per accession label: the ``key`` column and ``TOKEN``.

    The ``acc`` column may hold several labels separated by commas; each is stripped and
    empty ones are dropped (the reading ``_link`` uses).
    """
    return (
        samples.select(key, pl.col(acc).str.split(",").alias("TOKEN"))
        .explode("TOKEN")
        .with_columns(pl.col("TOKEN").str.strip_chars())
        .filter(pl.col("TOKEN").str.len_chars() > 0)
    )


def accession_matches(tokens: pl.DataFrame, refs: pl.DataFrame, key: str) -> pl.DataFrame:
    """Returns the (``key``, REF) rows of ``refs`` that a label in ``tokens`` carries.

    A label matches only references of the same ``key``. A reference may be a prefix of
    a label, as in the build's tube link; an equal label matches too.
    """
    return (
        tokens.join(refs, on=key, how="inner")
        .filter(pl.col("TOKEN").str.starts_with(pl.col("REF")))
        .select(key, "REF")
        .unique()
    )


def _lb_records(path: str) -> pl.LazyFrame:
    """Reads USUBJID, VISITNUM, ADT and REF from SDTM LB: every row, no DTYPE filter.

    The accession is ``LBREFID`` (blank reads as null), the owner ``USUBJID``, the visit
    ``VISITNUM`` and the date the first 10 characters of ``LBDTC`` read as ``%Y-%m-%d``;
    a partial date raises. Rows are not filtered on ``LBSTAT``.
    """
    return _lb_frame(pl.scan_parquet(path), path)


def _lb_frame(lf: pl.LazyFrame, name: str) -> pl.LazyFrame:
    """Reads an LB frame as ``_lb_records`` does; ``name`` labels it in the error."""
    missing = [
        c for c in ("USUBJID", "VISITNUM", "LBDTC", "LBREFID") if c not in lf.collect_schema()
    ]
    if missing:
        raise ValueError(f"[manifest] lb {name} lacks {', '.join(missing)}")
    return lf.select(
        pl.col("USUBJID").cast(pl.Utf8),
        pl.col("VISITNUM").cast(pl.Float64),
        reject_short_years(
            pl.col("LBDTC")
            .cast(pl.Utf8)
            .str.strip_chars()
            .replace("", None)
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d")
        ).alias("ADT"),
        pl.col("LBREFID").cast(pl.Utf8).str.strip_chars().replace("", None).alias("REF"),
    )


def _link(samples: pl.DataFrame, recs: pl.DataFrame, dataset: str) -> tuple[pl.DataFrame, set[str]]:
    """Links each sample to the ``recs`` row whose reference its accession carries.

    Each label is cut to every reference length in ``recs``, so a reference that is a
    prefix of the accession links. An accession whose labels reach two tube
    records keeps the record dated on the vendor date.

    Args:
        samples: Study rows with SampleID, ACC and VENDOR_DATE.
        recs: Tube records with USUBJID, VISITNUM, ADT and REF.
        dataset: The source's name, for errors.

    Returns:
        SampleID, OWNER, TUBE_VISIT and TUBE_DATE per linked sample, and the SampleIDs
        resolved by the vendor date.

    Raises:
        ValueError: If a sample still links to more than one reference, owner, visit or
            date.
    """
    lengths = sorted(recs["REF"].str.len_chars().unique().to_list())
    tokens = accession_tokens(samples, "SampleID")
    cand = (
        pl.concat(
            [
                tokens.filter(pl.col("TOKEN").str.len_chars() >= n).select(
                    "SampleID", pl.col("TOKEN").str.slice(0, n).alias("REF")
                )
                for n in lengths
            ]
        )
        if lengths
        else tokens.select("SampleID", pl.col("TOKEN").alias("REF")).clear()
    )
    hits = cand.unique().join(recs, on="REF", how="inner")
    multi = hits.group_by("SampleID").agg(pl.col("REF").n_unique().alias("n_ref"))
    hits = hits.join(multi, on="SampleID").join(
        samples.select("SampleID", "VENDOR_DATE"), on="SampleID", how="left"
    )
    by_date = (pl.col("n_ref") > 1) & (pl.col("ADT") == pl.col("VENDOR_DATE")).fill_null(False)
    resolved = hits.filter(by_date).group_by("SampleID").agg(pl.col("REF").n_unique().alias("n"))
    resolved = resolved.filter(pl.col("n") == 1)["SampleID"]
    hits = hits.filter(~pl.col("SampleID").is_in(resolved.implode()) | by_date)
    per = hits.group_by("SampleID").agg(
        pl.col("REF").n_unique().alias("n_ref"),
        pl.col("USUBJID").n_unique().alias("n_owner"),
        pl.col("VISITNUM").drop_nulls().n_unique().alias("n_visit"),
        pl.col("ADT").drop_nulls().n_unique().alias("n_date"),
        pl.col("USUBJID").first().alias("OWNER"),
        pl.col("VISITNUM").drop_nulls().first().alias("TUBE_VISIT"),
        pl.col("ADT").drop_nulls().first().alias("TUBE_DATE"),
    )
    for what, col in (
        ("tube references", "n_ref"),
        ("owners", "n_owner"),
        ("visits", "n_visit"),
        ("dates", "n_date"),
    ):
        n = per.filter(pl.col(col) > 1).height
        if n:
            raise ValueError(f"{n} study samples link to more than one {dataset.upper()} {what}")
    return per.select("SampleID", "OWNER", "TUBE_VISIT", "TUBE_DATE"), set(resolved.to_list())


def _tube_records(study: pl.DataFrame, m: ManifestConfig) -> tuple[pl.DataFrame, dict]:
    """Returns each study sample's tube owner, visit and date from its first source.

    Sources in order: SDTM LB (``[manifest] lb``), ADaM ADLB ``LBREFID``, then ADaM ADIG
    ``ISREFID``, the last only when the config has no LB. Two sources that link one tube
    with a different owner raise. A different visit or date is counted in
    ``counts["disagree"]`` and the first source still wins.
    """
    fields = ("OWNER", "TUBE_VISIT", "TUBE_DATE")
    out = study.select("SampleID")
    linked = []
    counts = {}
    by_date: set[str] = set()  # a sample resolved in two sources counts once
    for dataset, ref_col in TUBE_REF_COLUMNS.items():
        path = getattr(m, dataset)
        if path is None or (dataset == "adig" and m.lb is not None):
            continue
        if dataset == "lb":
            recs = _lb_records(path)
        elif ref_col not in pl.scan_parquet(path).collect_schema():
            continue
        else:
            recs = _lab(path, ref_col)
        link, resolved = _link(study, recs.drop_nulls("REF").unique().collect(), dataset)
        counts[dataset] = link.height
        by_date |= resolved
        out = out.join(
            link.rename({f: f"{f}_{dataset}" for f in fields}),
            on="SampleID",
            how="left",
            maintain_order="left",
        )
        linked.append(dataset)
    picks = {f: pl.lit(None) for f in fields}
    for dataset in reversed(linked):
        has = pl.col(f"OWNER_{dataset}").is_not_null()
        picks = {
            f: pl.when(has).then(pl.col(f"{f}_{dataset}")).otherwise(e) for f, e in picks.items()
        }
    disagree = pl.lit(False)
    for i, a in enumerate(linked):
        for b in linked[i + 1 :]:
            both = pl.col(f"OWNER_{a}").is_not_null() & pl.col(f"OWNER_{b}").is_not_null()
            n_owner = out.filter(both & pl.col(f"OWNER_{a}").ne(pl.col(f"OWNER_{b}"))).height
            if n_owner:
                raise ValueError(
                    f"{n_owner} study samples link to different owners in {a.upper()} and {b.upper()}"
                )
            disagree = disagree | (
                both
                & pl.any_horizontal(
                    pl.col(f"{f}_{a}").ne_missing(pl.col(f"{f}_{b}")) for f in fields[1:]
                )
            )
    counts["disagree"] = out.filter(disagree).height
    counts["resolved_by_vendor_date"] = len(by_date)
    out = out.select(
        "SampleID",
        picks["OWNER"].cast(pl.Utf8).alias("OWNER"),
        picks["TUBE_VISIT"].cast(pl.Float64).alias("TUBE_VISIT"),
        picks["TUBE_DATE"].cast(pl.Date).alias("TUBE_DATE"),
    )
    return out, counts


# ---------------------------------------------------------------- corrections


def _apply_renames(
    rows: pl.DataFrame, corrections: tuple[Correction, ...]
) -> tuple[pl.DataFrame, int]:
    """Renames a vendor SampleID by appending ``strip_suffix`` (rename entries).

    The entry's digest resolves against the vendor SampleIDs plus ``strip_suffix``.
    """
    renames = [c for c in corrections if c.type == "rename"]
    key_ids = rows.filter(pl.col("_kind") != "control")["SampleID"].to_list()
    for corr in renames:
        assay_id = resolve_digest(
            corr.sampleid_sha256,
            [k + corr.strip_suffix for k in key_ids],
            "sampleid_sha256 of a vendor SampleID plus strip_suffix",
        )
        vendor_id = assay_id.removesuffix(corr.strip_suffix)
        if assay_id in key_ids:
            raise ValueError(
                f"rename correction {corr.sampleid_sha256[:12]}...: the suffixed SampleID is "
                "already a vendor key row"
            )
        rows = rows.with_columns(
            pl.when(pl.col("SampleID") == vendor_id)
            .then(pl.lit(assay_id))
            .otherwise(pl.col("SampleID"))
            .alias("SampleID")
        )
        key_ids = [assay_id if k == vendor_id else k for k in key_ids]
    return rows, len(renames)


def _apply_bridge_swaps(rows: pl.DataFrame, corrections: tuple[Correction, ...]) -> pl.DataFrame:
    """Applies bridge_swap entries: exchanges ``SWAP_FIELDS`` between two key rows.

    The entry's label goes to ``_swap_label``; ``_apply_corrections`` labels the other
    row and checks its participant.

    Raises:
        ValueError: If a well is named by more than one entry, or the named wells are not
            a bridge row and a study row (a stale entry).
    """
    swaps = [c for c in corrections if c.type == "bridge_swap"]
    rows = rows.with_columns(pl.lit(None, pl.Utf8).alias("_swap_label"))
    kinds = dict(rows.filter(pl.col("_kind") != "control").select("SampleID", "_kind").iter_rows())
    pairs = [
        (
            corr,
            resolve_digest(corr.sampleid_sha256, kinds, "sampleid_sha256 in the vendor key"),
            resolve_digest(
                corr.study_sampleid_sha256, kinds, "study_sampleid_sha256 in the vendor key"
            ),
        )
        for corr in swaps
    ]
    named = [sid for _, bridge_id, study_id in pairs for sid in (bridge_id, study_id)]
    if len(set(named)) != len(named):
        raise ValueError("a well is named by more than one bridge_swap correction")
    for corr, bridge_id, study_id in pairs:
        if (kinds[bridge_id], kinds[study_id]) != ("bridge", "study"):
            raise ValueError(
                f"stale bridge_swap correction {corr.sampleid_sha256[:12]}...: it names a "
                f"{kinds[bridge_id]} row and a {kinds[study_id]} row; expected a bridge row "
                "and a study row"
            )
        is_bridge = pl.col("SampleID") == bridge_id
        is_study = pl.col("SampleID") == study_id
        bridge_row = rows.filter(is_bridge).select(SWAP_FIELDS).row(0, named=True)
        study_row = rows.filter(is_study).select(SWAP_FIELDS).row(0, named=True)
        rows = rows.with_columns(
            *(
                pl.when(is_bridge)
                .then(pl.lit(study_row[f], dtype=rows.schema[f]))
                .when(is_study)
                .then(pl.lit(bridge_row[f], dtype=rows.schema[f]))
                .otherwise(pl.col(f))
                .alias(f)
                for f in SWAP_FIELDS
            ),
            pl.when(is_study)
            .then(pl.lit(corr.label))
            .otherwise(pl.col("_swap_label"))
            .alias("_swap_label"),
        )
    return rows


def _check_labels(corrections: tuple[Correction, ...], identifiers: set[str]) -> None:
    """Raises when a correction's label holds an ID of the key or ADSL.

    Labels must be identifier-free. An ID counts when it appears in the label
    delimited by anything but a letter or digit, case ignored, so a short numeric SUBJID
    matches only as a whole number. The error names the correction type and a count,
    never an ID.
    """
    for corr in corrections:
        if corr.label is None:
            continue
        low = corr.label.lower()
        hits = sum(
            1
            for ident in identifiers
            if ident.lower() in low
            and re.search(rf"(?<![a-z0-9]){re.escape(ident.lower())}(?![a-z0-9])", low)
        )
        if hits:
            raise ValueError(
                f"{corr.type} correction {corr.sampleid_sha256[:12]}...: the label holds {hits} "
                "identifier(s) of the vendor key or ADSL; a label must be identifier-free"
            )


def _apply_corrections(
    study: pl.DataFrame,
    corrections: tuple[Correction, ...],
    adsl_ids: list[str],
    n_renames: int,
    identifiers: set[str],
) -> tuple[pl.DataFrame, dict]:
    """Applies the subject_from_accession, bridge_swap, reassign, visit_set and exclude entries.

    A ``bridge_swap`` has already exchanged its rows (``_apply_bridge_swaps``); here its
    study row gets the label once its subject is checked against the entry's.
    Raises on an uncovered owner mismatch, a stale entry, a label that holds an ID, or
    two entries that set one tube's subject or one tube's visit. ``reassign`` and
    ``visit_set`` leave ``_corr_visit`` (the visit they set), ``_corr_type`` and
    ``_pre_USUBJID`` (the subject before a ``reassign``) for the date and visit lookups
    and for ``_check_corrections``.
    """
    _check_labels(corrections, identifiers)
    key_ids = study["SampleID"].to_list()
    applied: dict[str, int] = {"rename": n_renames} if n_renames else {}
    skipped: dict[str, int] = {}
    study = study.with_columns(
        pl.lit(False).alias("_covered"),
        pl.lit(None, pl.Utf8).alias("MANIFEST_EXCLUDE"),
        pl.lit(None, pl.Utf8).alias("MANIFEST_CORRECTION"),
        pl.lit(None, pl.Float64).alias("_corr_visit"),
        pl.lit(None, pl.Utf8).alias("_corr_type"),
    )

    def row_of(sid: str) -> dict:
        return study.filter(pl.col("SampleID") == sid).row(0, named=True)

    for corr in corrections:
        if corr.type not in MANIFEST_CORRECTIONS:
            skipped[corr.type] = skipped.get(corr.type, 0) + 1

    sets_subject: dict[str, int] = {}
    sets_visit: dict[str, int] = {}
    for corr in corrections:
        if corr.type in ("subject_from_accession", "reassign", "visit_set"):
            sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
            if corr.type != "visit_set":
                sets_subject[sid] = sets_subject.get(sid, 0) + 1
            if corr.type != "subject_from_accession":
                sets_visit[sid] = sets_visit.get(sid, 0) + 1
        elif corr.type == "bridge_swap":
            sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
            sets_subject[sid] = sets_subject.get(sid, 0) + 1
            sets_visit[sid] = sets_visit.get(sid, 0) + 1
    if any(n > 1 for n in sets_subject.values()):
        raise ValueError("more than one correction sets the subject of one tube")
    if any(n > 1 for n in sets_visit.values()):
        raise ValueError("more than one correction sets the visit of one tube")

    for corr in (c for c in corrections if c.type == "subject_from_accession"):
        sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
        frm = resolve_digest(corr.from_usubjid_sha256, adsl_ids, "from_usubjid_sha256 in ADSL")
        to = resolve_digest(corr.to_usubjid_sha256, adsl_ids, "to_usubjid_sha256 in ADSL")
        row = row_of(sid)
        if not (row["_mismatch"] and row["USUBJID"] == frm and row["OWNER"] == to):
            raise ValueError(
                f"stale subject_from_accession correction {corr.sampleid_sha256[:12]}...: the "
                "sample's vendor subject and accession owner do not match its from and to subjects"
            )
        hit = pl.col("SampleID") == sid
        study = study.with_columns(
            pl.when(hit).then(pl.lit(to)).otherwise(pl.col("USUBJID")).alias("USUBJID"),
            (pl.col("_covered") | hit).alias("_covered"),
        )
        applied[corr.type] = applied.get(corr.type, 0) + 1
    study = study.with_columns(pl.col("USUBJID").alias("_pre_USUBJID"))

    for corr in (c for c in corrections if c.type == "bridge_swap"):
        sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
        uid = resolve_digest(corr.usubjid_sha256, adsl_ids, "usubjid_sha256 in ADSL")
        if row_of(sid)["USUBJID"] != uid:
            raise ValueError(
                f"stale bridge_swap correction {corr.sampleid_sha256[:12]}...: the study well "
                "is not labeled for the subject the entry names"
            )
        study = study.with_columns(
            pl.when(pl.col("SampleID") == sid)
            .then(pl.lit(corr.label))
            .otherwise(pl.col("MANIFEST_CORRECTION"))
            .alias("MANIFEST_CORRECTION")
        )
        applied[corr.type] = applied.get(corr.type, 0) + 1

    for corr in (c for c in corrections if c.type == "reassign"):
        sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
        to = resolve_digest(corr.to_usubjid_sha256, adsl_ids, "to_usubjid_sha256 in ADSL")
        if corr.from_usubjid_sha256 is not None:
            frm = resolve_digest(corr.from_usubjid_sha256, adsl_ids, "from_usubjid_sha256 in ADSL")
            if row_of(sid)["USUBJID"] != frm:
                raise ValueError(
                    f"stale reassign correction {corr.sampleid_sha256[:12]}...: the sample no "
                    "longer belongs to the subject the entry moves it from"
                )
        hit = pl.col("SampleID") == sid
        study = study.with_columns(
            pl.when(hit).then(pl.lit(to)).otherwise(pl.col("USUBJID")).alias("USUBJID"),
            pl.when(hit)
            .then(pl.lit(corr.to_visitnum))
            .otherwise(pl.col("_corr_visit"))
            .alias("_corr_visit"),
            pl.when(hit)
            .then(pl.lit("reassign"))
            .otherwise(pl.col("_corr_type"))
            .alias("_corr_type"),
            pl.when(hit)
            .then(pl.lit(corr.label))
            .otherwise(pl.col("MANIFEST_CORRECTION"))
            .alias("MANIFEST_CORRECTION"),
            (pl.col("_covered") | hit).alias("_covered"),
        )
        applied[corr.type] = applied.get(corr.type, 0) + 1

    uncovered = study.filter(pl.col("_mismatch") & ~pl.col("_covered")).height
    if uncovered:
        raise ValueError(
            f"{uncovered} study sample(s) whose vendor subject differs from the accession owner "
            "(by SUBJID) and no subject_from_accession or reassign correction covers them"
        )

    for corr in (c for c in corrections if c.type == "visit_set"):
        sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
        if corr.usubjid_sha256 is not None:
            uid = resolve_digest(corr.usubjid_sha256, adsl_ids, "usubjid_sha256 in ADSL")
            if row_of(sid)["USUBJID"] != uid:
                raise ValueError(
                    f"stale visit_set correction {corr.sampleid_sha256[:12]}...: the sample no "
                    "longer belongs to the subject the entry names"
                )
        hit = pl.col("SampleID") == sid
        study = study.with_columns(
            pl.when(hit)
            .then(pl.lit(corr.visitnum))
            .otherwise(pl.col("_corr_visit"))
            .alias("_corr_visit"),
            pl.when(hit)
            .then(pl.lit("visit_set"))
            .otherwise(pl.col("_corr_type"))
            .alias("_corr_type"),
            pl.when(hit)
            .then(pl.lit(corr.label))
            .otherwise(pl.col("MANIFEST_CORRECTION"))
            .alias("MANIFEST_CORRECTION"),
        )
        applied[corr.type] = applied.get(corr.type, 0) + 1

    for corr in (c for c in corrections if c.type == "exclude"):
        sid = resolve_digest(corr.sampleid_sha256, key_ids, "sampleid_sha256 in the vendor key")
        row = row_of(sid)
        if corr.usubjid_sha256 is not None:
            uid = resolve_digest(corr.usubjid_sha256, adsl_ids, "usubjid_sha256 in ADSL")
            if row["USUBJID"] != uid:
                raise ValueError(
                    f"stale exclude correction {corr.sampleid_sha256[:12]}...: the sample no "
                    "longer belongs to the subject the entry names"
                )
        study = study.with_columns(
            pl.when(pl.col("SampleID") == sid)
            .then(pl.lit(corr.label))
            .otherwise(pl.col("MANIFEST_EXCLUDE"))
            .alias("MANIFEST_EXCLUDE")
        )
        applied[corr.type] = applied.get(corr.type, 0) + 1
    report = {
        "applied": dict(sorted(applied.items())),
        "not_applied_by_build": dict(sorted(skipped.items())),
    }
    return study.drop("_covered"), report


def _check_corrections(study: pl.DataFrame, natural: pl.DataFrame | None) -> None:
    """Raises on a stale ``reassign`` or ``visit_set`` and on a clash.

    ``study`` has the final subjects and visits; ``natural`` is the same rows built as
    if no ``reassign`` or ``visit_set`` had applied (None when none did). Stale: the
    tube already sits at the entry's subject and visit. Clash: a corrected tube
    (a ``reassign``, ``visit_set`` or ``bridge_swap`` row) shares (USUBJID, VISITNUM)
    with another study tube that is not excluded. A ``bridge_swap`` sets no visit, so
    only this second check covers it.
    """
    if natural is not None:
        stale = study.join(
            natural.select("SampleID", pl.col("VISITNUM").alias("_natural_visit")),
            on="SampleID",
            how="left",
        ).filter(
            pl.col("_corr_visit").is_not_null()
            & (pl.col("USUBJID") == pl.col("_pre_USUBJID"))
            & (pl.col("_natural_visit") == pl.col("_corr_visit"))
        )
        if stale.height:
            kind = sorted(stale["_corr_type"].unique().to_list())[0]
            raise ValueError(
                f"{stale.height} stale {kind} correction(s): the tube already sits at the subject "
                "and visit the entry names"
            )
    live = study.filter(
        pl.col("MANIFEST_EXCLUDE").is_null()
        & pl.col("USUBJID").is_not_null()
        & pl.col("VISITNUM").is_not_null()
    )
    clash = live.filter(
        (pl.len().over("USUBJID", "VISITNUM") > 1) & pl.col("MANIFEST_CORRECTION").is_not_null()
    )
    if clash.height:
        raise ValueError(
            f"{clash.height} corrected tube(s) share a subject and visit with another study tube "
            "that is not excluded"
        )


# ---------------------------------------------------------------- dates and visits


def _vendor_date(raw: pl.Expr) -> pl.Expr:
    token = raw.str.extract(r"^([^ T]+)", 1)
    return reject_short_years(
        pl.coalesce([token.str.to_date(f, strict=False) for f in VENDOR_DATE_FORMATS])
    )


def _date_digit_difference(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Returns whether two dates differ in 1 or 2 digits within one date part.

    The part is the year, the month or the day.
    """
    sa, sb = a.dt.strftime("%Y%m%d"), b.dt.strftime("%Y%m%d")
    digits = pl.sum_horizontal(
        (sa.str.slice(i, 1) != sb.str.slice(i, 1)).cast(pl.Int32) for i in range(8)
    )
    fields = pl.sum_horizontal(
        (sa.str.slice(i, n) != sb.str.slice(i, n)).cast(pl.Int32)
        for i, n in ((0, 4), (4, 2), (6, 2))
    )
    return (fields == 1) & digits.is_between(1, 2)


def _assign_adt(study: pl.DataFrame, m: ManifestConfig) -> pl.DataFrame:
    """Assigns ADT and ADT_SOURCE, with the columns the report counts."""
    lb = _visit_dates(m.adlb).rename({"DATES": "LB_DATES"})
    vs = _visit_dates(m.advs).rename({"DATES": "VS_DATES"})
    agree = pl.col("LB_DATES").list.set_intersection(
        pl.col("VS_DATES").fill_null(pl.lit([], dtype=pl.List(pl.Date)))
    )
    visit = (
        lb.join(vs, on=["USUBJID", "VISITNUM"], how="full", coalesce=True)
        .with_columns(
            pl.when(pl.col("LB_DATES").list.len() == 1)
            .then(pl.col("LB_DATES").list.first())
            .when(agree.list.len() == 1)
            .then(agree.list.first())
            .otherwise(pl.col("LB_DATES").list.min())
            .alias("LB_DATE"),
            (pl.col("LB_DATES").list.len() > 1).alias("LB_MULTI"),
            (agree.list.len() == 1).alias("LB_AGREE"),
            pl.col("VS_DATES").list.min().alias("VS_DATE"),
            (pl.col("VS_DATES").list.len() > 1).alias("VS_MULTI"),
        )
        .rename({"VISITNUM": "FILL_VISIT"})
    )
    frame = (
        study.lazy()
        .with_columns(
            pl.coalesce("_corr_visit", "TUBE_VISIT", "VISITNUM_PLANNED").alias("FILL_VISIT")
        )
        .join(visit, on=["USUBJID", "FILL_VISIT"], how="left", maintain_order="left")
    )
    source = (
        pl.when(pl.col("TUBE_DATE").is_not_null())
        .then(pl.lit("tube"))
        .when(pl.col("VENDOR_DATE").is_not_null())
        .then(pl.lit("vendor"))
        .when(pl.col("LB_DATE").is_not_null())
        .then(pl.lit("adlb_visit"))
        .when(pl.col("VS_DATE").is_not_null())
        .then(pl.lit("advs"))
    )
    disagree = (
        pl.col("TUBE_DATE").is_not_null()
        & pl.col("VENDOR_DATE").is_not_null()
        & (pl.col("TUBE_DATE") != pl.col("VENDOR_DATE"))
    )
    return frame.with_columns(
        pl.coalesce("TUBE_DATE", "VENDOR_DATE", "LB_DATE", "VS_DATE").alias("ADT"),
        source.alias("ADT_SOURCE"),
        pl.when(disagree).then(pl.col("VENDOR_DATE")).alias("ADT_VENDOR"),
        pl.when(disagree)
        .then((pl.col("VENDOR_DATE") - pl.col("TUBE_DATE")).dt.total_days())
        .alias("_vendor_gap"),
        (disagree & _date_digit_difference(pl.col("VENDOR_DATE"), pl.col("TUBE_DATE"))).alias(
            "_date_digit_difference"
        ),
        (
            pl.col("TUBE_DATE").is_not_null()
            & pl.col("LB_DATES").is_not_null()
            & ~pl.col("LB_DATES").list.contains(pl.col("TUBE_DATE"))
        )
        .fill_null(False)
        .alias("_tube_vs_adlb"),
    ).collect()


def _assign_visit(study: pl.DataFrame, m: ManifestConfig) -> pl.DataFrame:
    """Assigns VISITNUM and VISITNUM_SOURCE."""
    day = (
        _lab(m.adlb)
        .drop_nulls(["USUBJID", "VISITNUM", "ADT"])
        .group_by("USUBJID", "ADT")
        .agg(pl.col("VISITNUM").unique().sort().alias("DAY"))
        .with_columns(
            pl.col("DAY")
            .list.eval(pl.element().filter(_scheduled(pl.element())))
            .alias("DAY_SCHED")
        )
    )
    tv = pl.col("TUBE_VISIT")
    # A reassign or visit_set sets the visit.
    corrected = pl.col("_corr_visit").is_not_null()
    # An ADT filled from the ADLB or ADVS date at the vendor-labeled visit is no draw
    # date: the tube keeps the visit that filled it and is never date-matched, since
    # another visit drawn that day says nothing about this tube.
    label_date = (
        ~corrected
        & tv.is_null()
        & pl.col("ADT_SOURCE").is_in(list(LABEL_DATE_SOURCES)).fill_null(False)
    )
    matched = ~corrected & tv.is_null() & ~label_date
    n_day, n_sched = pl.col("DAY").list.len(), pl.col("DAY_SCHED").list.len()
    same_day = ~corrected & tv.is_not_null() & ~_scheduled(tv) & (n_sched == 1)
    two_sched = matched & (n_sched > 1)
    unsched_only = matched & (n_day > 1) & (n_sched == 0)
    visit = (
        pl.when(corrected)
        .then(pl.col("_corr_visit"))
        .when(same_day)
        .then(pl.col("DAY_SCHED").list.first())
        .when(tv.is_not_null())
        .then(tv)
        .when(label_date)
        .then(pl.col("FILL_VISIT"))
        .when(n_day == 1)
        .then(pl.col("DAY").list.first())
        .when(n_sched == 1)
        .then(pl.col("DAY_SCHED").list.first())
        .when(unsched_only)
        .then(pl.col("DAY").list.min())
    )
    source = (
        pl.when(corrected)
        .then(pl.lit("correction"))
        # A tube record exists: its ADT is the tube visit's, no label's.
        .when(same_day)
        .then(pl.lit("date_match"))
        .when(tv.is_not_null())
        .then(pl.lit("tube"))
        .when(label_date)
        .then(pl.lit(VENDOR_LABEL_DATE))
        .when(visit.is_not_null())
        .then(pl.lit("date_match"))
        .otherwise(pl.lit("none"))
    )
    return (
        study.lazy()
        .join(day, on=["USUBJID", "ADT"], how="left", maintain_order="left")
        .with_columns(
            visit.alias("VISITNUM"),
            source.alias("VISITNUM_SOURCE"),
            same_day.fill_null(False).alias("_same_day"),
            two_sched.fill_null(False).alias("_two_sched"),
            unsched_only.fill_null(False).alias("_unsched_only"),
            (matched & pl.col("ADT").is_not_null() & pl.col("DAY").is_null()).alias("_no_hit"),
        )
        .collect()
    )


# ---------------------------------------------------------------- build


def _counts(series: pl.Series) -> dict[str, int]:
    vc = series.fill_null("none").value_counts()
    return {str(k): int(n) for k, n in sorted(vc.rows())}


def build_manifest(config: TrialConfig, out_dir: Path) -> tuple[pl.DataFrame, dict]:
    """Builds the manifest and writes it with its build report to ``out_dir``.

    Writes ``manifest.parquet`` and ``manifest_build_report.json``.

    Args:
        config: The trial config; its ``[manifest]`` block names the inputs.
        out_dir: Output directory, created if missing.

    Returns:
        The manifest and the report. The manifest has one row per vendor key row,
        bridge and control wells included, in key order; its columns and types are
        ``MANIFEST_SCHEMA``, then ``VOLUME_NOTE`` (``OPTIONAL_MANIFEST_COLUMNS``) when
        ``[manifest.columns] volume_note`` is set. The report holds counts and the
        inputs' full paths, never IDs.

    Raises:
        ValueError: If the config has no ``[manifest]``, or the key, ADSL or a
            correction would make the manifest wrong (see the module docstring).
    """
    m = config.manifest
    if m is None:
        raise ValueError("TrialConfig has no [manifest] block; nothing to build")
    c: ManifestColumns = m.columns
    key, dropped = _read_key(m)

    control = pl.col(c.sample_type).is_in(CONTROL_SAMPLE_TYPES).fill_null(False)
    flagged = pl.col(c.bridge_flag).is_not_null() if c.bridge_flag else pl.col(c.subject).is_null()
    if m.bridge_subject_pattern is not None:
        flagged = flagged | pl.col(c.subject).str.contains(
            f"^(?:{m.bridge_subject_pattern})$"
        ).fill_null(False)
    bridge = ~control & flagged
    by_usubjid = m.subject_join == "USUBJID"
    subject_key = (
        pl.col(c.subject).str.strip_chars() if by_usubjid else _norm_subjid(pl.col(c.subject))
    )
    sample_id = pl.col(c.sample_id)
    if c.bridge_flag:
        sample_id = (
            pl.when(bridge).then(pl.coalesce(c.sample_id, c.bridge_flag)).otherwise(sample_id)
        )
    rows = key.select(
        pl.int_range(pl.len()).alias("_row"),
        sample_id.alias("SampleID"),
        pl.when(control)
        .then(pl.lit("control"))
        .when(bridge)
        .then(pl.lit("bridge"))
        .otherwise(pl.lit("study"))
        .alias("_kind"),
        pl.col(c.sample_type).alias("SampleType"),
        subject_key.alias("_subj"),
        pl.col(c.accession).alias("ACC"),
        (pl.col(c.visit_label) if c.visit_label else pl.lit(None, pl.Utf8)).alias("_label"),
        (pl.col(c.date) if c.date else pl.lit(None, pl.Utf8)).alias("_date_raw"),
        (pl.col(c.volume_note) if c.volume_note else pl.lit(None, pl.Utf8)).alias("_note"),
    )
    n_null = rows.filter(pl.col("SampleID").is_null()).height
    if n_null:
        raise ValueError(f"{n_null} vendor key rows have no SampleID")
    study_like = pl.col("ACC").is_not_null() | pl.col("_label").str.contains(
        _VISIT_LABEL
    ).fill_null(False)
    n_bad_bridge = rows.filter((pl.col("_kind") == "bridge") & study_like).height
    if n_bad_bridge:
        raise ValueError(
            f"{n_bad_bridge} vendor key row(s) classed as bridge carry an accession or a visit "
            "label"
        )
    non_control = rows.filter(pl.col("_kind") != "control")
    n_dup = non_control.filter(pl.col("SampleID").is_duplicated())["SampleID"].n_unique()
    if n_dup:
        raise ValueError(f"{n_dup} SampleID(s) appear on more than one row of the vendor key")
    rows, n_renames = _apply_renames(rows, config.corrections)
    rows = _apply_bridge_swaps(rows, config.corrections)

    # The join key is held in the SUBJID column of this frame: ADSL SUBJID, or USUBJID
    # itself (subject_join).
    join_key = (
        pl.col("USUBJID").cast(pl.Utf8).str.strip_chars()
        if by_usubjid
        else _norm_subjid(pl.col("SUBJID"))
    )
    adsl = (
        pl.scan_parquet(m.adsl)
        .select(pl.col("USUBJID").cast(pl.Utf8), join_key.alias("SUBJID"))
        .collect()
    )
    if adsl["SUBJID"].drop_nulls().n_unique() != adsl["SUBJID"].drop_nulls().len():
        raise ValueError(
            f"ADSL {m.subject_join} is not unique; the vendor subject cannot be resolved"
        )
    if m.bridge_subject_pattern is not None:
        n_adsl = rows.filter(
            (pl.col("_kind") == "bridge")
            & pl.col("_subj").is_in(adsl["SUBJID"].drop_nulls().implode())
        ).height
        if n_adsl:
            raise ValueError(
                f"[manifest] bridge_subject_pattern matches {n_adsl} vendor key row(s) whose subject is an ADSL "
                "subject: the pattern must name bridge donors only"
            )

    label_visit = pl.col("_label").str.extract(_VISIT_LABEL, 1).cast(pl.Float64)
    vendor_date = _vendor_date(pl.col("_date_raw"))
    noted_visit = pl.col("_date_raw").str.extract(_NOTED_VISIT, 1).cast(pl.Float64)
    study = (
        rows.filter(pl.col("_kind") == "study")
        .join(
            adsl.rename({"SUBJID": "_subj"}),
            on="_subj",
            how="left",
            maintain_order="left",
        )
        .with_columns(
            vendor_date.alias("VENDOR_DATE"),
            label_visit.alias("_label_visit"),
        )
        .with_columns(
            pl.coalesce(
                "_label_visit",
                pl.when(pl.col("VENDOR_DATE").is_null()).then(noted_visit),
            ).alias("VISITNUM_PLANNED"),
        )
    )
    tubes, link_counts = _tube_records(study, m)
    # The sources consulted, in order, and the count of tubes where two of them differ.
    link_sources = ("lb", "adlb") if m.lb is not None else ("adlb", "adig")
    tube_links = {
        **{src: link_counts.get(src, 0) for src in link_sources},
        f"{'_'.join(link_sources)}_disagree": link_counts["disagree"],
    }
    study = study.join(tubes, on="SampleID", how="left", maintain_order="left")
    owner_subj = (
        study.select("_row", "OWNER")
        .join(
            adsl.rename({"USUBJID": "OWNER", "SUBJID": "_owner_subj"}),
            on="OWNER",
            how="left",
        )
        .with_columns(
            pl.coalesce(
                "_owner_subj",
                pl.col("OWNER").str.strip_chars()
                if by_usubjid
                else _norm_subjid(pl.col("OWNER").str.split("-").list.last()),
            ).alias("_owner_subj")
        )
        .select("_row", "_owner_subj")
    )
    study = (
        study.join(owner_subj, on="_row", how="left", maintain_order="left")
        .with_columns(
            (
                pl.col("OWNER").is_not_null() & pl.col("_owner_subj").ne_missing(pl.col("_subj"))
            ).alias("_mismatch"),
        )
        .with_columns(
            (
                pl.col("OWNER").is_not_null()
                & ~pl.col("_mismatch")
                & pl.col("OWNER").ne_missing(pl.col("USUBJID"))
            ).alias("_site_transfer"),
        )
    )
    identifiers = {
        *rows.filter(pl.col("_kind") != "control")["SampleID"].to_list(),
        *(t.strip() for acc in rows["ACC"].drop_nulls().to_list() for t in acc.split(",")),
        *adsl["USUBJID"].drop_nulls().to_list(),
        *adsl["SUBJID"].drop_nulls().to_list(),
        *study["_subj"].drop_nulls().to_list(),
    } - {""}
    study, corr_report = _apply_corrections(
        study, config.corrections, adsl["USUBJID"].to_list(), n_renames, identifiers
    )
    natural = None
    if study["_corr_visit"].is_not_null().any():
        # The same rows as if no reassign or visit_set had applied: the stale check
        # compares with them.
        natural = _assign_visit(
            _assign_adt(
                study.with_columns(
                    pl.col("_pre_USUBJID").alias("USUBJID"),
                    pl.lit(None, pl.Float64).alias("_corr_visit"),
                ),
                m,
            ),
            m,
        )
    study = _assign_visit(_assign_adt(study, m), m)
    _check_corrections(study, natural)

    sv = study.filter(pl.col("USUBJID").is_not_null() & pl.col("VISITNUM").is_not_null())
    dup_visit = sv.filter(pl.len().over("USUBJID", "VISITNUM") > 1).height
    order = (
        sv.filter(_scheduled(pl.col("VISITNUM")) & pl.col("ADT").is_not_null())
        .sort("USUBJID", "VISITNUM", "ADT")
        .with_columns(pl.col("ADT").cum_max().shift(1).over("USUBJID").alias("_prev"))
        .filter(pl.col("ADT") < pl.col("_prev"))
        .height
    )

    def is_src(source: str) -> pl.Expr:
        return (pl.col("ADT_SOURCE") == source).fill_null(False)

    warn = {
        WARN_ADLB_VISIT_MULTIPLE_DATES: study.filter(
            is_src("adlb_visit") & pl.col("LB_MULTI") & ~pl.col("LB_AGREE")
        ).height,
        WARN_ADT_FILL_ADLB_VISIT: study.filter(is_src("adlb_visit")).height,
        WARN_ADT_FILL_ADVS: study.filter(is_src("advs")).height,
        WARN_ADVS_VISIT_MULTIPLE_DATES: study.filter(is_src("advs") & pl.col("VS_MULTI")).height,
        WARN_DATE_MATCH_NO_HIT: study.filter(pl.col("_no_hit")).height,
        WARN_DATE_MATCH_UNSCHEDULED_ONLY: study.filter(pl.col("_unsched_only")).height,
        WARN_DUPLICATE_SUBJECT_VISIT: dup_visit,
        WARN_PLANNED_VISIT_DIFFERS: study.filter(
            pl.col("VISITNUM").ne(pl.col("VISITNUM_PLANNED")).fill_null(False)
        ).height,
        WARN_SITE_TRANSFER: study.filter(pl.col("_site_transfer")).height,
        WARN_TUBE_ADLB_VISIT_DATE_DISAGREE: study.filter(pl.col("_tube_vs_adlb")).height,
        WARN_TUBE_RECORDS_DISAGREE: link_counts["disagree"],
        WARN_VENDOR_DATE_UNPARSED: study.filter(
            pl.col("_date_raw").is_not_null() & pl.col("VENDOR_DATE").is_null()
        ).height,
        WARN_VENDOR_TUBE_DATE_DISAGREE: study.filter(pl.col("ADT_VENDOR").is_not_null()).height,
        WARN_VISIT_DATE_ORDER: order,
        WARN_VISIT_LABEL_UNPARSED: study.filter(
            pl.col("_label").is_not_null() & pl.col("_label_visit").is_null()
        ).height,
    }
    fail = {
        FAIL_NO_ADSL_SUBJECT: study.filter(pl.col("USUBJID").is_null()).height,
        FAIL_NULL_ADT: study.filter(pl.col("ADT").is_null()).height,
        FAIL_TWO_SCHEDULED_VISITS_SAME_DATE: study.filter(pl.col("_two_sched")).height,
    }

    manifest = (
        rows.select("_row", "SampleID", "_kind", "SampleType")
        .join(
            study.select(
                "_row",
                *[
                    k
                    for k in MANIFEST_SCHEMA
                    if k
                    not in (
                        "SampleID",
                        "SampleType",
                        "BridgeSample",
                        "AccessionNumber",
                    )
                ],
                "ACC",
            ),
            on="_row",
            how="left",
        )
        .sort("_row")
        .join(rows.select("_row", "_note", "_swap_label"), on="_row", how="left")
        .with_columns(
            (pl.col("_kind") == "bridge").alias("BridgeSample"),
            pl.col("ACC").alias("AccessionNumber"),
            pl.col("_note").alias("VOLUME_NOTE"),
            pl.coalesce("MANIFEST_CORRECTION", "_swap_label").alias("MANIFEST_CORRECTION"),
        )
        .select(
            [
                pl.col(k).cast(t)
                for k, t in {
                    **MANIFEST_SCHEMA,
                    **(OPTIONAL_MANIFEST_COLUMNS if c.volume_note else {}),
                }.items()
            ]
        )
    )
    report = {
        "study_name": config.study_name,
        "inputs": {k: str(v) for k, v in m.files().items()},
        "rows": {
            "key_rows": rows.height,
            "study": study.height,
            "bridge": rows.filter(pl.col("_kind") == "bridge").height,
            "control": rows.filter(pl.col("_kind") == "control").height,
            **dropped,
        },
        "visitnum_source": _counts(study["VISITNUM_SOURCE"]),
        "adt_source": _counts(study["ADT_SOURCE"]),
        "visitnum": _counts(study["VISITNUM"].cast(pl.Utf8)),
        "tube_links": tube_links,
        "tube_links_resolved_by_vendor_date": link_counts.get("resolved_by_vendor_date", 0),
        "tube_unscheduled_same_day_scheduled": study.filter(pl.col("_same_day")).height,
        "adlb_visit_two_dates_advs_agree": study.filter(
            is_src("adlb_visit") & pl.col("LB_MULTI") & pl.col("LB_AGREE")
        ).height,
        "vendor_tube_date_digit_difference": study.filter(pl.col("_date_digit_difference")).height,
        "warn": {k: v for k, v in warn.items() if v},
        "fail": {k: v for k, v in fail.items() if v},
        "corrections": corr_report,
    }
    if by_usubjid or m.bridge_subject_pattern is not None:
        report["settings"] = {
            "subject_join": m.subject_join,
            "bridge_subject_pattern": m.bridge_subject_pattern,
        }
    if c.volume_note:
        report["volume_notes"] = rows.filter(pl.col("_note").is_not_null()).height
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(out_dir / "manifest.parquet")
    (out_dir / "manifest_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return manifest, report
