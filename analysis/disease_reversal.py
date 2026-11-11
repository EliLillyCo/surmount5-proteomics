"""Disease-signature reversal (Fig 3b): Spearman ρ of HPA disease signatures with TZP − SEMA.

For each adult disease of the HPA pan-disease blood atlas (``analysis.external.hpa_disease_atlas``),
the signature is the set of proteins with adjusted P < 0.05 against healthy controls, matched one
row per assay: an assay enters once, with the mean HPA log2 fold change of the signature rows its
genes match (an assay of paralog genes that HPA lists as two rows, CKMT1A and CKMT1B, takes their
mean; the IL-12 p70 assay takes IL12A's and IL12B's while p40 takes IL12B's). The statistic is
Spearman ρ (average ranks) across the signature's assays between that log fold change and the
assay's TZP − SEMA z (``assay_level_z``: estimate / SE, ``OLINK_EXCLUDED_ASSAYS`` excluded). The reversal score is
−ρ.

Null: z ~ N(0, C), with C the week-0 correlation of the signature's assays (``baseline_assay_matrix``,
each assay standardized), estimated from the participants complete on those assays. A draw is z = uA with
u ~ N(0, I_n) and A the standardized columns / sqrt(n − 1), so AᵀA = C; ``ROTATION_DRAWS``
draws per disease from a fresh ``default_rng(SEED)``, in chunks of ``ROTATION_CHUNK``, each
draw ranked. Two-sided add-one P = (1 + #{|ρ_null| ≥ |ρ_obs|}) / (1 + B), with Monte Carlo
SE sqrt(P(1 − P) / B); Benjamini–Hochberg FDR across the adult diseases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from analysis.pathway.preparation import assay_gene_pairs
from analysis.output_layout import ANALYSIS_OUTPUTS
from analysis.statistics import SEED, bh_fdr

logger = logging.getLogger(__name__)

# Null draws per disease.
ROTATION_DRAWS = 1_000_000
ROTATION_CHUNK = 500
# HPA's adjusted-P cutoff for each disease signature.
SIGNATURE_PADJ = 0.05
SIGNATURE_CONTROL = "Healthy"
# HPA's class of pediatric diseases, whose signatures are not tested.
PEDIATRIC_CLASS = "Pediatric"
# Absolute slack when counting null draws at least as extreme as the observed ρ.
COUNT_TOL = 1e-12
# Below three matched assays a rank correlation is undefined (one) or always +-1 (two).
MIN_SIGNATURE_ASSAYS = 3
PLATFORM = "olink"


def output_path(week: int) -> Path:
    """``analysis/outputs/disease_reversal_olink_wk<week>.parquet``, read by Fig 3b."""
    return ANALYSIS_OUTPUTS / f"disease_reversal_{PLATFORM}_wk{week}.parquet"


@dataclass(frozen=True, eq=False)
class DiseaseSignature:
    """One adult disease's signature on our assays.

    Attributes:
        Disease: HPA disease name.
        Class: HPA disease class.
        idx: Indices into the assay list of the statistic (``assay_level_z``).
        lfc: The HPA log2 fold changes, aligned with ``idx``.
    """

    Disease: str
    Class: str
    idx: np.ndarray
    lfc: np.ndarray


def assay_level_z(
    stats: pl.DataFrame, mapping: dict[str, list[str]], excluded_assays: tuple[str, ...]
) -> tuple[list[str], dict[str, list[str]], np.ndarray]:
    """The usable assays with a gene, their genes and their z (estimate / SE) for one contrast.

    Args:
        stats: One ``contrast_statistics`` frame (``marker, estimate, StdErr, fdr``).
        mapping: Assay id -> gene symbols.
        excluded_assays: Assay ids dropped first (``EXCLUDED_ASSAYS[platform]``).

    Returns:
        ``(assays, assay_genes, z)``: the assays, sorted (``assay_gene_pairs`` keeps those with a
        positive finite SE, a finite estimate and a gene), each assay's genes (each once), and the
        assays' z in that order.
    """
    pairs = assay_gene_pairs(stats.filter(~pl.col("marker").is_in(excluded_assays)), mapping)
    assay_genes: dict[str, list[str]] = {}
    assay_z: dict[str, float] = {}
    for marker, gene, z in pairs.select("marker", "gene", "z").iter_rows():
        assay_genes.setdefault(marker, []).append(gene)
        assay_z[marker] = z
    assays = sorted(assay_genes)
    return assays, {a: assay_genes[a] for a in assays}, np.array([assay_z[a] for a in assays])


def ensg_alias(annotation: pl.DataFrame, xref: pl.DataFrame) -> pl.DataFrame:
    """``ENSG ID, alias``: the Ensembl gene of each Olink accession that carries one gene, and that gene.

    Only one-to-one pairs give an alias: an accession with several genes of ours (identical proteins
    from paralogous loci) or with several Ensembl genes (paralogous loci, read-through loci such as
    PTGES3L-AARSD1) gives none, because the cross-reference cannot say which ENSG is which gene.

    Args:
        annotation: The assay annotation (``Platform``, ``UniProt``, ``Gene_Symbol``).
        xref: ``ENSG ID, uniprot`` (``analysis.external.ensembl_uniprot.read_uniprot_xref``).
    """
    single = (
        annotation.filter(
            (pl.col("Platform") == PLATFORM)
            & pl.col("Gene_Symbol").is_not_null()
            & ~pl.col("Gene_Symbol").str.contains("|", literal=True)
        )
        .select(pl.col("UniProt").alias("uniprot"), pl.col("Gene_Symbol").alias("alias"))
        .unique(maintain_order=True)
    )
    one_gene = xref.filter(pl.col("ENSG ID").n_unique().over("uniprot") == 1)
    return one_gene.join(single, on="uniprot", how="inner", maintain_order="left").select("ENSG ID", "alias").unique(
        maintain_order=True
    )


def hpa_symbol_map(
    de: pl.DataFrame, olink_alias: pl.DataFrame, our_genes: list[str]
) -> dict[str, str]:
    """HPA gene symbol -> our gene symbol: itself when we measure it, else through its ENSG id.

    Args:
        de: HPA table with ``Gene`` and ``ENSG ID``.
        olink_alias: ``ENSG ID, alias`` pairs (``ensg_alias``).
        our_genes: Gene symbols of the ranked statistic.

    Returns:
        One entry per HPA symbol that reaches one of ``our_genes``; the others are absent.

    Raises:
        ValueError: When an unmatched HPA symbol reaches more than one of our genes through
            its ENSG ids, so the map would depend on row order.
    """
    ours = set(our_genes)
    with_alias = de.select("Gene", "ENSG ID").unique().join(olink_alias, on="ENSG ID", how="left")
    symbol_map: dict[str, str] = {}
    aliases_reached: dict[str, set[str]] = {}
    for gene, alias in with_alias.select("Gene", "alias").iter_rows():
        if gene in ours:
            symbol_map[gene] = gene
        elif alias is not None and alias in ours:
            symbol_map[gene] = alias
            aliases_reached.setdefault(gene, set()).add(alias)
    ambiguous = {gene: sorted(a) for gene, a in aliases_reached.items() if len(a) > 1}
    if ambiguous:
        raise ValueError(f"order-dependent HPA symbol map: {ambiguous}")
    return symbol_map


def disease_signatures(
    de: pl.DataFrame,
    olink_alias: pl.DataFrame,
    assays: list[str],
    assay_genes: dict[str, list[str]],
) -> list[DiseaseSignature]:
    """Adult disease signatures against healthy controls, one entry per assay.

    Pediatric diseases (``Class == PEDIATRIC_CLASS``) are dropped before anything is tested. An
    assay enters a signature when any of its genes matches a signature row (``hpa_symbol_map``),
    once, with the mean HPA log fold change of the rows its genes match.

    Args:
        de: HPA table (``read_disease_de``), every column as text.
        olink_alias: ``ENSG ID, alias`` pairs, as for ``hpa_symbol_map``.
        assays: The assays of the statistic (``assay_level_z``); ``idx`` indexes this list.
        assay_genes: Assay -> its genes.

    Returns:
        One signature per adult disease, in HPA order; the entries in ``assays`` order.

    Raises:
        ValueError: When the symbol map is order-dependent (``hpa_symbol_map``), two HPA
            symbols of one disease map to the same gene of ours, or a disease has fewer
            than ``MIN_SIGNATURE_ASSAYS`` matched assays.
    """
    de = de.with_columns(
        pl.col("logFC").cast(pl.Float64), pl.col("p-value adjusted").cast(pl.Float64).alias("padj")
    )
    our_genes = sorted({g for genes in assay_genes.values() for g in genes})
    symbol_map = hpa_symbol_map(de, olink_alias, our_genes)
    versus_healthy = de.filter(
        pl.col("Control") == SIGNATURE_CONTROL, pl.col("Class") != PEDIATRIC_CLASS
    )
    significant = versus_healthy.filter(pl.col("padj") < SIGNATURE_PADJ)
    signatures = []
    diseases = versus_healthy.select("Disease", "Class").unique(maintain_order=True)
    for disease, disease_class in diseases.iter_rows():
        rows = significant.filter(pl.col("Disease") == disease).select("Gene", "logFC").iter_rows()
        lfc_of = [(symbol_map[gene], lfc) for gene, lfc in rows if gene in symbol_map]
        ours = [g for g, _ in lfc_of]
        if len(set(ours)) != len(ours):
            raise ValueError(f"{disease}: two HPA symbols map to one of our genes")
        by_gene = dict(lfc_of)
        entries = [
            (i, float(np.mean([by_gene[g] for g in assay_genes[a] if g in by_gene])))
            for i, a in enumerate(assays)
            if any(g in by_gene for g in assay_genes[a])
        ]
        if len(entries) < MIN_SIGNATURE_ASSAYS:
            raise ValueError(
                f"{disease}: {len(entries)} of our assays carry a significant HPA protein; "
                f"a rank correlation needs at least {MIN_SIGNATURE_ASSAYS}"
            )
        idx = np.array([i for i, _ in entries], dtype=int)
        lfcs = np.array([v for _, v in entries], dtype=float)
        signatures.append(DiseaseSignature(disease, disease_class, idx, lfcs))
    return signatures


def _unit_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks of ``x``, centered and scaled to unit norm."""
    ranks = rankdata(x)
    ranks = ranks - ranks.mean()
    return ranks / np.linalg.norm(ranks)


class SpearmanBank:
    """Spearman ρ of each signature's HPA log fold change against rows of a z matrix.

    Average ranks, as ``scipy.stats.spearmanr``; the signature ranks are computed once.
    """

    def __init__(self, signatures: list[DiseaseSignature]) -> None:
        self.signatures = signatures
        self.unit_ranks = [_unit_ranks(s.lfc) for s in signatures]

    def rho(self, z: np.ndarray) -> np.ndarray:
        """P x K matrix of ρ for the P rows of ``z`` (P x genes), one column per signature."""
        z = np.atleast_2d(z)
        out = np.empty((z.shape[0], len(self.signatures)))
        for k, (signature, x) in enumerate(zip(self.signatures, self.unit_ranks)):
            y = rankdata(z[:, signature.idx], axis=1)
            y = y - y.mean(1, keepdims=True)
            out[:, k] = (y @ x) / np.linalg.norm(y, axis=1)
        return out


def baseline_assay_matrix(npx: np.ndarray) -> np.ndarray:
    """Participants x assays week-0 NPX, each assay standardized (mean 0, SD 1), NaN where missing."""
    return (npx - np.nanmean(npx, 0)) / np.nanstd(npx, 0, ddof=1)


def complete_case(baseline: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``baseline`` on ``cols`` (in that order), the participants complete on all of them.

    The rows of the matrix without a missing value in the selected columns.
    """
    x = baseline[:, cols]
    return x[np.isfinite(x).all(axis=1)]


def rotation_factor(x_sub: np.ndarray) -> np.ndarray:
    """A with AᵀA = C, the column correlation of ``x_sub``: standardized columns / sqrt(n − 1)."""
    return (x_sub - x_sub.mean(0)) / x_sub.std(0, ddof=1) / np.sqrt(x_sub.shape[0] - 1)


def count_extreme(rho_null: np.ndarray, rho_obs: float) -> int:
    """Null draws with |ρ| at least |ρ_obs| (less ``COUNT_TOL``)."""
    return int((np.abs(rho_null) >= abs(rho_obs) - COUNT_TOL).sum())


def add_one_p(n_extreme: int, n_draws: int) -> float:
    """Two-sided add-one Monte Carlo P, (1 + n_extreme) / (1 + n_draws)."""
    return (1 + n_extreme) / (1 + n_draws)


def rotation_null(
    factor: np.ndarray,
    signature_ranks: np.ndarray,
    rho_obs: float,
    draws: int = ROTATION_DRAWS,
    chunk: int = ROTATION_CHUNK,
    seed: int = SEED,
) -> tuple[int, float]:
    """Count of null ρ at least as extreme as ``rho_obs``, and the null RMS of ρ.

    Args:
        factor: ``rotation_factor`` of the signature's complete-case baseline matrix (n x G).
        signature_ranks: Centered unit-norm ranks of the HPA log fold change, in the column
            order of ``factor``.
        rho_obs: Observed ρ.
        draws: Null draws; a multiple of ``chunk``.
        chunk: Draws per matrix product.
        seed: Seed of the fresh ``np.random.default_rng`` for this signature.

    Returns:
        ``(n_extreme, null_sd)``, with ``null_sd = sqrt(mean(ρ_null²))``.

    Raises:
        ValueError: When ``draws`` is not a multiple of ``chunk``.
    """
    if draws % chunk != 0:
        raise ValueError(f"draws {draws} is not a multiple of chunk {chunk}")
    rng = np.random.default_rng(seed)
    n_extreme, sum_sq = 0, 0.0
    for _ in range(draws // chunk):
        y = rankdata(rng.standard_normal((chunk, factor.shape[0])) @ factor, axis=1)
        y -= y.mean(1, keepdims=True)
        rho_null = (y @ signature_ranks) / np.linalg.norm(y, axis=1)
        n_extreme += count_extreme(rho_null, rho_obs)
        sum_sq += float((rho_null**2).sum())
    return n_extreme, float(np.sqrt(sum_sq / draws))


def reversal_table(
    signatures: list[DiseaseSignature],
    z: np.ndarray,
    baseline: np.ndarray,
    draws: int = ROTATION_DRAWS,
    chunk: int = ROTATION_CHUNK,
) -> pl.DataFrame:
    """One row per signature: −ρ, the rotation-null P and its MC SE, BH FDR across the signatures.

    Progress is logged at INFO, every tenth signature.

    Args:
        signatures: ``disease_signatures`` output (adult diseases only).
        z: Assay-level statistic, indexed like the signatures' ``idx``.
        baseline: ``baseline_assay_matrix`` on the same assays.
        draws: Null draws per signature.
        chunk: Draws per matrix product.

    Returns:
        ``Disease, Class, n_genes, n_baseline_subjects, neg_rho, null_sd, p, mc_se, fdr``.

    Raises:
        ValueError: When a pediatric signature is passed.
    """
    pediatric = [s.Disease for s in signatures if s.Class == PEDIATRIC_CLASS]
    if pediatric:
        raise ValueError(f"pediatric signatures are not tested: {pediatric}")
    bank = SpearmanBank(signatures)
    rho = bank.rho(z)[0]
    rows = []
    for k, signature in enumerate(signatures):
        x_sub = complete_case(baseline, signature.idx)
        n_extreme, null_sd = rotation_null(
            rotation_factor(x_sub), bank.unit_ranks[k], rho[k], draws, chunk
        )
        p = add_one_p(n_extreme, draws)
        rows.append(
            dict(
                Disease=signature.Disease,
                Class=signature.Class,
                n_genes=len(signature.idx),
                n_baseline_subjects=x_sub.shape[0],
                neg_rho=-rho[k],
                null_sd=null_sd,
                p=p,
                mc_se=np.sqrt(p * (1 - p) / draws),
            )
        )
        if k % 10 == 0:
            logger.info(f"rotation null {k + 1}/{len(signatures)}: {signature.Disease}")
    table = pl.DataFrame(rows)
    return table.with_columns(pl.Series("fdr", bh_fdr(table["p"].to_numpy())))
