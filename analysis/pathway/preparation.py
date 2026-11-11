"""Data preparation: MMRM raw outputs -> per-contrast statistics -> gene-level ranking.

The ranking statistic is the t-like ratio ``z = estimate / SE`` of the emmeans estimate on
the log2 scale: for the treatment contrasts the
TZP - SEMA ``diff`` of ``rawOutput/*_chg_diff.csv``. Multi-gene assays are exploded and a gene's
statistic is the equal-weight mean of z over the assays that map to it. There is no QC
gating beyond the analysis cohort (the engine already dropped vendor-failed Olink points).
Genes whose statistic comes from the same assays (``gene_source_assays``: the paralog genes of one
accession, the genes of one multi-target assay) carry the same z; cameraPR and ORA count them once
per gene set (``source_groups``). The ORA query is gene-level: a gene's P is the Simes combination of
its assays' P values, BH over genes, query = genes at FDR < 0.05 (``simes_gene_fdr``). Olink assays in
``OLINK_EXCLUDED_ASSAYS`` are dropped by assay id before ranking and from the ORA query
and background.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from analysis.mmrm_outputs import chg_diff_csv_in
from analysis.statistics import FDR_THRESHOLD, bh_fdr

logger = logging.getLogger(__name__)


def _with_fdr(df: pl.DataFrame) -> pl.DataFrame:
    """Add BH ``fdr`` over the proteins of one contrast."""
    return df.with_columns(pl.Series("fdr", bh_fdr(df["pVal"].to_numpy())))


def contrast_statistics(mmrm_dir: Path) -> dict[str, pl.DataFrame]:
    """Per-contrast statistics read from an MMRM ``covar_*`` directory's ``rawOutput``.

    Args:
        mmrm_dir: ``<root>/<study>_<platform>/covar_<label>``.

    Returns:
        ``{label: frame}`` with columns ``marker, estimate, StdErr, pVal, fdr``; the label
        ``"TZP15mgorMTDVSSEMA2.4mgorMTD@24"`` names the same-visit treatment contrast
        ``trt - _trt`` (TZP - SEMA) at a visit. ``fdr`` is BH within the contrast.

    Raises:
        FileNotFoundError: When the diff file is not exactly one.
    """
    diff = pl.read_csv(chg_diff_csv_in(mmrm_dir), schema_overrides={"time": pl.String, "_time": pl.String})

    out: dict[str, pl.DataFrame] = {}
    across = diff.filter((pl.col("trt") != pl.col("_trt")) & (pl.col("time") == pl.col("_time")))
    for (trt, ref, time), grp in across.group_by(["trt", "_trt", "time"], maintain_order=True):
        out[f"{trt}VS{ref}@{time}"] = _with_fdr(
            grp.select("marker", pl.col("diff").alias("estimate"), "StdErr", "pVal").sort("marker")
        )
    return out


def assay_gene_pairs(stats: pl.DataFrame, mapping: dict[str, list[str]]) -> pl.DataFrame:
    """Long ``marker, gene, z, fdr`` frame: one row per (usable assay, mapped gene).

    The shared assay -> gene expansion of the ranking statistic (``ranked_genes``), the ORA
    query (``significant_genes``) and the disease-signature reversal
    (``analysis.disease_reversal.gene_level_z``). Assays without a positive finite SE, a
    finite estimate or a gene are dropped; a gene repeated for one assay counts once.

    Args:
        stats: Frame with ``marker, estimate, StdErr, fdr``.
        mapping: Assay id -> gene symbols (several for a multi-gene assay).

    Returns:
        The pairs in ``stats`` row order, ``z = estimate / StdErr``.
    """
    usable = stats.filter(
        pl.col("StdErr").is_not_null() & pl.col("StdErr").is_finite() & (pl.col("StdErr") > 0)
        & pl.col("estimate").is_finite()
    )
    rows = [
        (marker, gene, estimate / se, fdr)
        for marker, estimate, se, fdr in usable.select("marker", "estimate", "StdErr", "fdr").iter_rows()
        for gene in dict.fromkeys(g for g in mapping.get(marker, []) if g)
    ]
    schema = {"marker": pl.String, "gene": pl.String, "z": pl.Float64, "fdr": pl.Float64}
    return pl.DataFrame(rows, schema=schema, orient="row")


def gene_source_assays(
    stats: pl.DataFrame, mapping: dict[str, list[str]], excluded_assays: tuple[str, ...] = ()
) -> dict[str, frozenset[str]]:
    """Gene -> the usable assays its statistic comes from (``assay_gene_pairs`` after ``excluded_assays``).

    Genes with the same source-assay set carry the same value (paralog genes of one accession, the
    genes of one multi-target assay); gene sets and disease signatures count such genes once.
    """
    pairs = assay_gene_pairs(stats.filter(~pl.col("marker").is_in(excluded_assays)), mapping)
    out: dict[str, set[str]] = {}
    for marker, gene in pairs.select("marker", "gene").iter_rows():
        out.setdefault(gene, set()).add(marker)
    return {gene: frozenset(markers) for gene, markers in sorted(out.items())}


def source_groups(sources: dict[str, frozenset[str]]) -> dict[str, str]:
    """Gene -> a key shared exactly by the genes with the same source-assay set (``gene_source_assays``)."""
    return {gene: "|".join(sorted(markers)) for gene, markers in sources.items()}


def ranked_genes(stats: pl.DataFrame, mapping: dict[str, list[str]]) -> dict[str, float]:
    """Gene -> z = estimate / SE, equal-weight mean over the assays of a gene.

    Args:
        stats: Frame with ``marker, estimate, StdErr, fdr``.
        mapping: Assay id -> gene symbols (several for a multi-gene assay).

    Returns:
        Gene symbols (sorted, so tied statistics keep one rank order run to run) to z.
    """
    agg = (
        assay_gene_pairs(stats, mapping)
        .group_by("gene")
        .agg(pl.col("z").mean())
        .sort("gene")
    )
    return dict(zip(agg["gene"].to_list(), agg["z"].to_list()))


def simes_gene_fdr(stats: pl.DataFrame, mapping: dict[str, list[str]]) -> dict[str, float]:
    """Gene -> BH FDR of its Simes P over the usable assays that map to it (the ORA query statistic).

    A gene's P is Simes' combination of its assays' P values, min_i(m p_(i) / i) over its m assays in
    ascending order; Benjamini-Hochberg then runs over the genes, each same-source group once
    (``source_groups``: genes measured by the identical assays share one P and are one test). Usable assays as in ``assay_gene_pairs``.

    Args:
        stats: Frame with ``marker, estimate, StdErr, pVal, fdr`` (excluded assays already dropped).
        mapping: Assay id -> gene symbols.
    """
    pairs = assay_gene_pairs(stats, mapping).join(stats.select("marker", "pVal"), on="marker", how="left")
    by_gene: dict[str, list[float]] = {}
    for gene, p in pairs.select("gene", "pVal").iter_rows():
        by_gene.setdefault(gene, []).append(p)
    simes = {}
    for gene, ps in sorted(by_gene.items()):
        ordered = np.sort(np.asarray(ps, dtype=float))
        simes[gene] = float(np.min(len(ordered) * ordered / np.arange(1, len(ordered) + 1)))
    groups = source_groups(gene_source_assays(stats, mapping))
    keys = sorted({groups[g] for g in simes})
    group_p = {groups[g]: p for g, p in simes.items()}
    group_fdr = dict(zip(keys, bh_fdr(np.array([group_p[k] for k in keys]))))
    return {g: float(group_fdr[groups[g]]) for g in simes}


def significant_genes(
    stats: pl.DataFrame, mapping: dict[str, list[str]], threshold: float = FDR_THRESHOLD
) -> list[str]:
    """The ORA query: genes whose gene-level Simes FDR (``simes_gene_fdr``) is below ``threshold``."""
    return sorted(g for g, q in simes_gene_fdr(stats, mapping).items() if q < threshold)


def prepare_data(
    stats: pl.DataFrame, mapping: dict[str, list[str]], excluded_assays: tuple[str, ...] = ()
) -> tuple[dict[str, float], list[str]]:
    """Ranked genes and significant genes for one contrast.

    Args:
        stats: One ``contrast_statistics`` frame.
        mapping: Assay id -> gene symbols for the frame's platform.
        excluded_assays: Assay ids dropped before ranking and from the ORA query and
            background (``OLINK_EXCLUDED_ASSAYS`` for Olink, empty for SomaScan).

    Returns:
        ``(ranked_genes, significant_genes)``.
    """
    n_before = stats.height
    stats = stats.filter(~pl.col("marker").is_in(excluded_assays))
    logger.info(f"Excluded {n_before - stats.height} assays by id ({', '.join(excluded_assays) or 'none'})")
    ranked = ranked_genes(stats, mapping)
    sig = significant_genes(stats, mapping)
    logger.info(f"Mapped {stats.height} assays -> {len(ranked)} genes; {len(sig)} significant (FDR<{FDR_THRESHOLD})")
    return ranked, sig
