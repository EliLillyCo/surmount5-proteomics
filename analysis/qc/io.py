"""Shared file scanner that keeps identifier columns as strings."""

import os
from pathlib import Path

import polars as pl

# Identifiers must stay text: numeric-looking values lose leading zeros as integers
# and then fail to join against the string IDs used everywhere else.
ID_COLUMNS = ("SampleID", "SampleId", "PatientID", "USUBJID")

_SUPPORTED = ".parquet, .csv, .tsv, .csv.gz, .tsv.gz"


def _scan_delimited(path: str, separator: str, **kwargs) -> pl.LazyFrame:
    """Scans a delimited file with a schema that does not depend on row order.

    Types come from every row (``infer_schema_length=None``), so shuffling the
    file cannot change them. Identifier columns are read as String, which keeps
    leading zeros on numeric-looking IDs. A column that mixes types across rows
    infers as String and fails at its first numeric use.

    Args:
        path: A csv or tsv file, gzipped or not.
        separator: The field separator.
        **kwargs: Passed to ``pl.scan_csv``.

    Returns:
        The scanned file.
    """
    # The header names pick the ID columns present, since an override for an absent
    # column is an error.
    names = pl.scan_csv(path, separator=separator, has_header=True).collect_schema().names()
    id_overrides = {c: pl.String for c in ID_COLUMNS if c in names}
    return pl.scan_csv(
        path,
        separator=separator,
        has_header=True,
        infer_schema_length=None,
        schema_overrides=id_overrides,
        **kwargs,
    )


def scan_file(path: str | Path, **kwargs) -> pl.LazyFrame:
    """Scans a file by its extension (parquet, csv, tsv, or gzipped csv or tsv).

    ID columns (``ID_COLUMNS``) come back as String in every format. Delimited
    files read them as String at parse time, because a cast after the scan comes
    too late when Polars has typed an all-digit column as integer from its first
    rows and then meets a text ID. Parquet carries its own types, so a cast is
    enough.

    Args:
        path: The file to scan.
        **kwargs: Passed to ``pl.scan_parquet`` or ``pl.scan_csv``.

    Returns:
        The scanned file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the extension is not supported.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".gz":
        ext = os.path.splitext(os.path.splitext(path)[0])[1].lower()
        if ext not in (".csv", ".tsv"):
            raise ValueError(f"Unsupported compressed file type: {path}")

    if ext == ".parquet":
        scanned = pl.scan_parquet(path, **kwargs)
        present = [c for c in ID_COLUMNS if c in scanned.collect_schema().names()]
        return scanned.with_columns(pl.col(present).cast(pl.String))
    if ext == ".csv":
        return _scan_delimited(path, ",", **kwargs)
    if ext == ".tsv":
        return _scan_delimited(path, "\t", **kwargs)
    raise ValueError(f"Unsupported file extension: {ext}. Supported: {_SUPPORTED}")
