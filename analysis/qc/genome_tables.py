"""Reads the precomputed genome-wide KING table and ancestry PCs for the pQTL step.

The pQTL step can take two genotype summaries from files:

- **KING kinship** (``pqtl.kinship_table``): a parquet with ``USUBJID_1``,
  ``USUBJID_2`` and ``kinship``. A pair the table does not list is read as unrelated. A table listing every pair is read the same way.
- **Ancestry PCs** (``pqtl.ancestry_pcs``): a TSV with a subject ID column
  (``PCS_ID_COLUMN``) and ``PC1`` to ``PCn``; the ancestry null reads the first
  ``pqtl.null_pcs``.

Both are keyed by USUBJID, matched to the PSAM IIDs.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from analysis.qc.pqtl import KING_UNRELATED

KING_TABLE_FLOOR: float = KING_UNRELATED
"""The smallest kinship the KING table is sure to list (plink2 ``--king-table-filter``,
the third-degree lower bound). A kinship cut below it cannot be read from the table."""
KING_SELF: float = 0.5
"""KING kinship of a genome with itself; the matrix diagonal, as ``kinship_matrix``
gives it, so a genotype is its own duplicate genome."""
KING_COLUMNS: tuple[str, ...] = ("USUBJID_1", "USUBJID_2", "kinship")
PCS_ID_COLUMN: str = "dx_usubjid"
_PC_NAME = re.compile(r"PC(\d+)")


def read_king_table(path: str) -> pl.DataFrame:
    """Reads and checks a KING table.

    Args:
        path: Parquet with ``USUBJID_1``, ``USUBJID_2`` and ``kinship``.

    Returns:
        ``USUBJID_1``, ``USUBJID_2`` (String) and ``kinship`` (Float64), one row per
        listed pair.

    Raises:
        ValueError: If a column is missing, an ID is null, a kinship is not finite, a
            row pairs a genotype with itself or a pair is listed twice (in either
            order).
    """
    table = pl.read_parquet(path)
    missing = [c for c in KING_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"KING table {path} lacks column(s) {', '.join(missing)}")
    table = table.select(
        pl.col("USUBJID_1").cast(pl.String),
        pl.col("USUBJID_2").cast(pl.String),
        pl.col("kinship").cast(pl.Float64),
    )
    if table.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError(f"KING table {path} has a null ID or kinship")
    if not table["kinship"].is_finite().all():
        raise ValueError(f"KING table {path} has a non-finite kinship")
    if (table["USUBJID_1"] == table["USUBJID_2"]).any():
        raise ValueError(f"KING table {path} pairs a genotype with itself")
    pair = table.select(
        pl.min_horizontal("USUBJID_1", "USUBJID_2").alias("a"),
        pl.max_horizontal("USUBJID_1", "USUBJID_2").alias("b"),
    )
    if pair.is_duplicated().any():
        raise ValueError(f"KING table {path} lists a pair twice")
    return table


def kinship_from_table(table: pl.DataFrame, usubjids: list[str], *, source: str) -> np.ndarray:
    """Returns the genotypes' KING kinship matrix from a KING table.

    A listed pair gets its kinship (both orders), the diagonal is ``KING_SELF``, and a
    pair the table does not list is NaN: below ``KING_TABLE_FLOOR``, so every reader
    treats it as unrelated (the relative and duplicate cuts and the identity step's
    ``genotype_unrelated``).

    Args:
        table: ``read_king_table`` output.
        usubjids: The genotypes (PSAM IIDs), in the matrix's order.
        source: The table's path, for messages.

    Raises:
        ValueError: If a table genotype identifier is absent from the PSAM IIDs in
            ``usubjids``; check that the table and PSAM use matching genotype identifiers.
    """
    index = {u: j for j, u in enumerate(usubjids)}
    ids = set(table["USUBJID_1"]) | set(table["USUBJID_2"])
    unknown = ids - index.keys()
    if unknown:
        raise ValueError(
            f"KING table {source}: {len(unknown)} of its {len(ids)} genotypes are not in the PSAM "
            f"({len(usubjids)} genotypes); check that the table and PSAM use matching genotype identifiers"
        )
    n = len(usubjids)
    K = np.full((n, n), np.nan)
    np.fill_diagonal(K, KING_SELF)
    i = np.array([index[u] for u in table["USUBJID_1"]], dtype=np.int64)
    j = np.array([index[u] for u in table["USUBJID_2"]], dtype=np.int64)
    k = table["kinship"].to_numpy()
    K[i, j] = k
    K[j, i] = k
    return K


def read_ancestry_pcs(path: str) -> pl.DataFrame:
    """Reads and checks a projected-PC file.

    Args:
        path: Tab-separated, with ``dx_usubjid`` and ``PC1`` to ``PCn``.

    Returns:
        ``USUBJID`` (String) and ``PC1`` to ``PCn`` (Float64), in that order.

    Raises:
        ValueError: If the ID column is missing, an ID is null or repeats, no ``PC``
            column is present, the ``PC`` columns are not ``PC1`` to ``PCn`` in order,
            or a PC value is not finite.
    """
    pcs = pl.read_csv(path, separator="\t", infer_schema=False)
    if PCS_ID_COLUMN not in pcs.columns:
        raise ValueError(f"ancestry PC file {path} lacks column {PCS_ID_COLUMN}")
    names = [c for c in pcs.columns if _PC_NAME.fullmatch(c)]
    if not names or names != [f"PC{k}" for k in range(1, len(names) + 1)]:
        raise ValueError(
            f"ancestry PC file {path} must have columns PC1 to PCn in order; got {names}"
        )
    out = pcs.select(
        pl.col(PCS_ID_COLUMN).alias("USUBJID"),
        *(pl.col(c).cast(pl.Float64) for c in names),
    )
    if out["USUBJID"].null_count() or not out["USUBJID"].is_unique().all():
        raise ValueError(f"ancestry PC file {path} has a null or repeated ID")
    # An empty field reads as null, which is_finite leaves null and all() would skip.
    finite = pl.col(names).is_finite().fill_null(False)
    if not out.select(pl.all_horizontal(finite).all()).item():
        raise ValueError(f"ancestry PC file {path} has a missing or non-finite PC")
    return out


def pcs_for(pcs: pl.DataFrame, usubjids: list[str], *, source: str) -> np.ndarray:
    """Returns every PC of the genotypes, rows in ``usubjids`` order.

    Args:
        pcs: ``read_ancestry_pcs`` output.
        usubjids: The genotypes (PSAM IIDs).
        source: The file's path, for messages.

    Returns:
        A genotypes x n_pcs float64 array.

    Raises:
        ValueError: If a PSAM genotype identifier has no PC row or the PC file has
            a genotype identifier absent from the PSAM IIDs in ``usubjids``; check
            that the PC file and PSAM use matching genotype identifiers.
    """
    have = set(pcs["USUBJID"])
    panel = set(usubjids)
    absent = panel - have
    if absent:
        raise ValueError(
            f"ancestry PC file {source}: {len(absent)} of the {len(panel)} genotypes in the PSAM have no PCs; check that the PC file and PSAM use matching genotype identifiers"
        )
    extra = have - panel
    if extra:
        raise ValueError(
            f"ancestry PC file {source}: {len(extra)} of its {len(have)} genotypes are not in the PSAM; "
            "check that the PC file and PSAM use matching genotype identifiers"
        )
    order = pl.DataFrame({"USUBJID": usubjids}, schema={"USUBJID": pl.String})
    return (
        order.join(pcs, on="USUBJID", how="left", maintain_order="left")
        .drop("USUBJID")
        .to_numpy()
        .astype(np.float64)
    )


def check_projected_pcs(pcs: np.ndarray, k: int, *, study: str) -> np.ndarray:
    """Returns the first ``k`` projected PCs after checking they can carry the fit.

    Projected PCs are numbers read from a file, so the singular-value gap that guards
    the in-data PCs (``pqtl_reference.check_singular_gap``, a LAPACK-path question)
    does not apply. What can still go wrong is a file too short for ``k`` or columns
    that cannot be regressed on: a non-finite value, a constant PC over the panel, or
    PCs collinear with each other or the intercept.

    Args:
        pcs: Genotypes x n_pcs (``pcs_for``).
        k: ``pqtl_null_pcs``.
        study: Study name, for messages.

    Raises:
        ValueError: If the file has fewer than ``k`` PCs, a value is not finite, or
            ``[1, PC1..PCk]`` is not of full column rank over the panel.
    """
    if pcs.shape[1] < k:
        raise ValueError(
            f"{study}: pqtl.null_pcs = {k} but the ancestry PC file has {pcs.shape[1]} PCs"
        )
    top = np.asarray(pcs[:, :k], dtype=np.float64)
    if not np.isfinite(top).all():
        raise ValueError(f"{study}: a projected ancestry PC is not finite")
    X = np.column_stack([np.ones(top.shape[0]), top])
    if np.linalg.matrix_rank(X) < k + 1:
        raise ValueError(
            f"{study}: the first {k} projected ancestry PCs are constant or collinear over the "
            f"{top.shape[0]} genotypes of the panel; the ancestry location fit needs {k + 1} independent columns"
        )
    return top
