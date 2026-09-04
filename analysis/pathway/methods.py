"""Pathway analysis methods: GSEA (preranked), ORA (enrichr), and cameraPR (limma)."""

from __future__ import annotations

import logging
import tempfile

import gseapy as gp
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import conversion, pandas2ri
from rpy2.robjects.packages import importr

from .cache import _gene_set_cache
from .constants import SEED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSEA (preranked)
# ---------------------------------------------------------------------------


def run_gsea(
    ranked_genes: dict,
    gene_sets: str,
    seed: int = SEED,
    gene_set_dict: dict | None = None,
) -> pd.DataFrame | None:
    """Run GSEA preranked analysis for one gene-set library."""
    if not ranked_genes:
        return None
    ranked_list = pd.Series(ranked_genes).sort_values(ascending=False)
    gs_input = gene_set_dict if gene_set_dict is not None else gene_sets
    with tempfile.TemporaryDirectory() as tmpdir:
        res = gp.prerank(
            rnk=ranked_list,
            gene_sets=gs_input,
            outdir=tmpdir,
            permutation_num=1000,
            min_size=15,
            max_size=500,
            seed=seed,
            no_plot=True,
            verbose=False,
        )
        df = res.res2d
        logger.info(
            f"GSEA {gene_sets}: {len(df)} pathways, "
            f"{(df['FDR q-val'] < 0.05).sum()} sig"
        )
        return df


# ---------------------------------------------------------------------------
# ORA (enrichr)
# ---------------------------------------------------------------------------


def run_ora(
    significant_genes: list,
    ranked_genes: dict,
    gene_sets: str,
    gene_set_dict: dict | None = None,
) -> pd.DataFrame | None:
    """Run ORA (enrichr) for one gene-set library."""
    background = [str(g) for g in ranked_genes if pd.notna(g)]
    sig = [str(g) for g in significant_genes if pd.notna(g)]
    if not sig:
        logger.warning(f"ORA {gene_sets}: skipped — no significant genes (FDR<0.05)")
        return None
    gs_input = gene_set_dict if gene_set_dict is not None else gene_sets
    with tempfile.TemporaryDirectory() as tmpdir:
        res = gp.enrichr(
            gene_list=sig,
            gene_sets=gs_input,
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
            f"ORA {gene_sets}: {len(df)} pathways, "
            f"{(df['Adjusted P-value'] < 0.05).sum()} sig"
        )
        return df


# ---------------------------------------------------------------------------
# cameraPR (limma)
# ---------------------------------------------------------------------------


def run_camera(
    ranked_genes: dict,
    gene_sets_name: str,
    gene_set_dict: dict | None = None,
    skip_genes: list[str] | None = None,
    download_missing: bool = False,
) -> pd.DataFrame | None:
    """Run limma cameraPR for one gene-set library.

    The statistic vector passed to cameraPR is ``-log10(pVal) * sign(effect)``
    derived from the MMRM contrast results.  This serves as the gene-level
    ranking metric for the competitive test.

    If *gene_set_dict* is provided (from the cache) it is used directly,
    avoiding a redundant download.

    *skip_genes*, if given, is a list of gene symbols to exclude from the
    ranked statistic vector before running cameraPR.  This is useful to
    remove proteins with very strong statistics that can bias the
    inter-gene correlation estimate.
    """
    if not ranked_genes:
        return None

    limma = importr("limma")

    if gene_set_dict is None:
        gene_set_dict = _gene_set_cache.get(
            gene_sets_name,
            download_missing=download_missing,
        )
    if not gene_set_dict:
        raise ValueError(f"Empty gene-set dict for {gene_sets_name}")

    stat = pd.Series(ranked_genes).sort_values(ascending=False)
    stat = stat[stat.index.notna()]
    stat.index = stat.index.astype(str)

    if skip_genes:
        before = len(stat)
        stat = stat.drop(labels=[g for g in skip_genes if g in stat.index])
        n_dropped = before - len(stat)
        if n_dropped:
            logger.info(f"cameraPR: skipped {n_dropped} genes from statistic vector")

    universe = set(stat.index)
    stat_list = list(stat.index)

    pathway_indices: dict[str, list] = {}
    for name, genes in gene_set_dict.items():
        idx = [stat_list.index(g) + 1 for g in genes if g in universe]
        if len(idx) >= 2:
            pathway_indices[name] = idx

    if not pathway_indices:
        logger.warning(f"cameraPR {gene_sets_name}: no testable pathways")
        return None
    logger.info(f"cameraPR {gene_sets_name}: testing {len(pathway_indices)} pathways")

    stat_r = ro.FloatVector(stat.values)
    stat_r.names = ro.StrVector(stat.index)
    index_list = ro.ListVector(
        {n: ro.IntVector(ix) for n, ix in pathway_indices.items()}
    )

    cam_res = limma.cameraPR(statistic=stat_r, index=index_list, sort=False)

    with (ro.default_converter + pandas2ri.converter).context():
        results_df = conversion.get_conversion().rpy2py(cam_res)

    results_df.insert(0, "Term", results_df.index)
    results_df = results_df.reset_index(drop=True)
    results_df = results_df.rename(
        columns={
            "PValue": "NOM p-val",
            "FDR": "FDR q-val",
            "NGenes": "N_Genes",
            "Direction": "Direction",
        }
    )
    results_df["N_Genes"] = results_df["N_Genes"].astype(int)

    sorted_genes = stat.sort_values(ascending=False)
    _annotate_camera_genes(results_df, gene_set_dict, universe, sorted_genes)

    results_df.insert(1, "Name", "cameraPR")
    results_df = results_df[
        [
            "Term",
            "Name",
            "Direction",
            "NOM p-val",
            "FDR q-val",
            "N_Genes",
            "Total_Genes",
            "Lead_genes",
            "Genes",
        ]
    ]

    logger.info(
        f"cameraPR {gene_sets_name}: {len(results_df)} pathways, "
        f"{(results_df['FDR q-val'] < 0.05).sum()} sig"
    )
    return results_df


def _annotate_camera_genes(df, gene_set_dict, universe, sorted_genes):
    """Add Total_Genes, Genes, Lead_genes columns to cameraPR results."""
    total, all_genes, lead_genes = [], [], []
    for _, row in df.iterrows():
        pw_genes = gene_set_dict[row["Term"]]
        total.append(len(pw_genes))
        in_data = [(g, sorted_genes[g]) for g in pw_genes if g in universe]
        up = row["Direction"] == "Up"
        in_data.sort(key=lambda x: x[1], reverse=up)
        n_lead = max(5, min(15, int(0.4 * len(in_data))))
        lead_genes.append(";".join(g for g, _ in in_data[:n_lead]))
        all_genes.append(";".join(g for g, _ in in_data))
    df["Total_Genes"] = total
    df["Genes"] = all_genes
    df["Lead_genes"] = lead_genes
