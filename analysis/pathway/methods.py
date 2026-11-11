"""Pathway analysis methods: cameraPR (limma) and ORA (Enrichr format)."""

from __future__ import annotations

import logging
import tempfile

import gseapy as gp
import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from analysis.statistics import FDR_THRESHOLD

from .constants import CAMERA_INTER_GENE_COR, CAMERA_MAX_SIZE, CAMERA_MIN_SIZE

logger = logging.getLogger(__name__)

# Columns of every cameraPR result table, in output order.
CAMERA_COLUMNS = [
    "Term",
    "Name",
    "Direction",
    "NOM p-val",
    "FDR q-val",
    "N_Genes",
    "Total_Genes",
    "Lead_genes",
    "Genes",
    "inter_gene_cor",
]


def collapse_same_source(members, universe, groups: dict[str, str] | None) -> list[str]:
    """A set's measured members, each name once and each same-source group once.

    Genes in one group (``preparation.source_groups``: identical source-assay sets) carry the
    same statistic; the set keeps the first of them in its own member order. Without ``groups``
    only repeated names are dropped.
    """
    seen: set[str] = set()
    out = []
    for g in members:
        if g not in universe:
            continue
        key = groups.get(g, g) if groups else g
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# ORA (enrichr)
# ---------------------------------------------------------------------------


def run_ora(
    significant_genes: list[str],
    ranked_genes: dict[str, float],
    gene_sets: str,
    gene_set_dict: dict,
    groups: dict[str, str] | None = None,
) -> pl.DataFrame | None:
    """Run ORA (Enrichr statistics) for one gene-set library.

    With ``groups``, one measurement is one draw: the query, the background and every set hold each
    same-source group once (under one name, the group's first gene alphabetically), so genes of one
    assay are one overlap, one set member, one query gene and one background gene. In ``Genes`` each
    group is named by the set's own first member of it (``collapse_same_source``). ``Genes`` lists the
    overlap alphabetically.

    Args:
        significant_genes: Query genes (``preparation.significant_genes``: gene-level Simes FDR < 0.05, both
            platforms).
        ranked_genes: Gene -> statistic; its keys are the background.
        gene_sets: Library name (for logging).
        gene_set_dict: Gene-set library as ``{term: [genes]}``, already translated to our symbols
            (``cache.GeneSetCache`` with the alias table); there is no fallback to an untranslated library.
        groups: Gene -> same-source key (``preparation.source_groups``).

    Returns:
        gseapy's result table, or None when there is no query gene or no result.
    """
    background = [str(g) for g in ranked_genes]
    sig = [str(g) for g in significant_genes]
    if not sig:
        logger.warning(f"ORA {gene_sets}: skipped — no significant genes (FDR<{FDR_THRESHOLD})")
        return None
    own_names: dict[str, dict[str, str]] = {}
    if groups is not None:
        universe = set(background)
        rep = {}
        for g in sorted(universe):
            rep.setdefault(groups.get(g, g), g)
        canonical = {g: rep[groups.get(g, g)] for g in universe}
        sets = {t: collapse_same_source(m, universe, groups) for t, m in gene_set_dict.items()}
        own_names = {t: {canonical[g]: g for g in members} for t, members in sets.items()}
        gene_set_dict = {t: [canonical[g] for g in members] for t, members in sets.items()}
        background = list(dict.fromkeys(canonical[g] for g in background))
        sig = list(dict.fromkeys(canonical[g] for g in sig if g in canonical))
    with tempfile.TemporaryDirectory() as tmpdir:
        res = gp.enrichr(
            gene_list=sig,
            gene_sets=gene_set_dict,
            background=background,
            outdir=tmpdir,
            cutoff=1.0,
            no_plot=True,
            verbose=False,
        )
        df = res.res2d
    if df is None or len(df) == 0:
        return None
    logger.info(
        f"ORA {gene_sets}: {len(df)} pathways, {(df['Adjusted P-value'] < FDR_THRESHOLD).sum()} sig"
    )
    out = pl.from_pandas(df)
    # gseapy returns each overlap as a set, so its order follows the string hash; list it alphabetically.
    return out.with_columns(
        pl.Series(
            "Genes",
            [
                ";".join(sorted(own_names.get(t, {}).get(g, g) for g in genes.split(";") if g))
                for t, genes in out.select("Term", "Genes").iter_rows()
            ],
            dtype=pl.String,
        )
    )


# ---------------------------------------------------------------------------
# cameraPR (limma)
# ---------------------------------------------------------------------------


def run_camera(
    ranked_genes: dict[str, float],
    gene_sets_name: str,
    gene_set_dict: dict,
    groups: dict[str, str] | None = None,
) -> pl.DataFrame | None:
    """Run limma cameraPR for one gene-set library.

    cameraPR does not estimate an inter-gene correlation from the statistic vector (that
    is ``camera``'s, which takes an expression matrix); it applies the fixed
    ``CAMERA_INTER_GENE_COR`` as a variance-inflation constant, passed explicitly and
    recorded on every row.

    Test sets within ``CAMERA_MIN_SIZE`` and ``CAMERA_MAX_SIZE`` after overlap with
    the ranked vector and deduplication of set members.

    With ``groups``, genes whose statistic comes from the same assays count once per set (the
    set's first such member stands for them, ``collapse_same_source``), in the index, the size
    bounds and the ``Genes``/``Lead_genes`` lists.

    Args:
        ranked_genes: Gene symbol -> ranking statistic (t-like z = estimate / SE).
        gene_sets_name: Enrichr library name.
        gene_set_dict: Library as ``{term: [genes]}``, already translated to our symbols
            (``cache.GeneSetCache`` with the alias table); there is no fallback to an untranslated library.
        groups: Gene -> same-source key (``preparation.source_groups``); None counts every name.

    Returns:
        One row per tested set with the ``CAMERA_COLUMNS``; None when there is no
        statistic or no set of testable size.

    Raises:
        ValueError: When the library holds no gene sets.
    """
    if not ranked_genes:
        return None
    limma = importr("limma")
    if not gene_set_dict:
        raise ValueError(f"Empty gene-set dict for {gene_sets_name}")

    # Descending statistic; Python's sort is stable, so tied genes keep the input order.
    genes = sorted(ranked_genes, key=ranked_genes.__getitem__, reverse=True)
    universe = set(genes)

    # Deduplicate genes within a set (a seen set, preserving order) before mapping to
    # positions: a symbol repeated inside one term would otherwise pass a duplicate 1-based
    # index to limma, double-weighting that gene in the statistic and inflating the overlap
    # count the size filter tests.
    stat_pos = {g: i for i, g in enumerate(genes)}
    pathway_indices: dict[str, list[int]] = {}
    for name, members in gene_set_dict.items():
        idx = [stat_pos[g] + 1 for g in collapse_same_source(members, universe, groups)]
        if max(2, CAMERA_MIN_SIZE) <= len(idx) <= CAMERA_MAX_SIZE:
            pathway_indices[name] = idx

    if not pathway_indices:
        logger.warning(f"cameraPR {gene_sets_name}: no testable pathways")
        return None
    logger.info(f"cameraPR {gene_sets_name}: testing {len(pathway_indices)} pathways")

    stat_r = ro.FloatVector([ranked_genes[g] for g in genes])
    stat_r.names = ro.StrVector(genes)
    index_list = ro.ListVector({n: ro.IntVector(ix) for n, ix in pathway_indices.items()})

    cam_res = limma.cameraPR(
        statistic=stat_r,
        index=index_list,
        **{"inter.gene.cor": CAMERA_INTER_GENE_COR},
        sort=False,
    )
    # The R data.frame's row names are the set names.
    with (ro.default_converter + pandas2ri.converter).context():
        cam_pd = ro.conversion.get_conversion().rpy2py(cam_res)
    results = pl.from_pandas(cam_pd).with_columns(pl.Series("Term", cam_pd.index.to_list()))

    total, all_genes, lead_genes = [], [], []
    for term, direction in results.select("Term", "Direction").iter_rows():
        pw_genes = list(dict.fromkeys(gene_set_dict[term]))
        total.append(len(pw_genes))
        in_data = [(g, ranked_genes[g]) for g in collapse_same_source(pw_genes, universe, groups)]
        in_data.sort(key=lambda x: x[1], reverse=direction == "Up")
        n_lead = max(5, min(15, int(0.4 * len(in_data))))
        lead_genes.append(";".join(g for g, _ in in_data[:n_lead]))
        all_genes.append(";".join(g for g, _ in in_data))

    results = results.rename(
        {"PValue": "NOM p-val", "FDR": "FDR q-val", "NGenes": "N_Genes"}
    ).with_columns(
        pl.col("N_Genes").cast(pl.Int64),
        pl.lit("cameraPR").alias("Name"),
        pl.Series("Total_Genes", total, dtype=pl.Int64),
        pl.Series("Genes", all_genes, dtype=pl.String),
        pl.Series("Lead_genes", lead_genes, dtype=pl.String),
        pl.lit(CAMERA_INTER_GENE_COR).alias("inter_gene_cor"),
    )
    logger.info(
        f"cameraPR {gene_sets_name}: {results.height} pathways, "
        f"{(results['FDR q-val'] < FDR_THRESHOLD).sum()} sig"
    )
    return results.select(CAMERA_COLUMNS)
