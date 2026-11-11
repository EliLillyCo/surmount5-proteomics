"""Shared pQTL helpers: genotype reads, genetic sex, kinship and per-sample calls.

Scoring itself is in ``pqtl_reference``. This module reads catalog variants and genetic
sex from per-chromosome PGENs, computes KING kinship from the catalog variants (or reads a
KING table, ``genome_tables``), extracts per-sample
concordance, makes the per-tube calls, applies the positive-identification rule and the
genetic-sex cross-check, and builds the DNA-link table.

Each sample's protein levels should match the pattern its linked participant's genotype
predicts at known cis-pQTLs. A low self score alone is not a mismatch: a mismatch needs positive
identification, another participant's proteome fitting the genotype better than the
linked participant's and at least as well as a typical correct link (Westra et al.
2011, Bioinformatics 27:2104).
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from analysis.qc.identity import robust_z
from analysis.qc.schemas import DNA_LINK_SCHEMA
from analysis.qc.sex_concordance import SEX_PROB_THRESHOLD

logger = logging.getLogger(__name__)


# KING kinship thresholds (Manichaikul et al. 2010).
KING_DUPLICATE: float = 0.354  # MZ twin or duplicate.
KING_SECOND_DEGREE: float = 0.0884  # Lower bound of 2nd degree.
KING_UNRELATED: float = 0.0442  # Lower bound of 3rd degree; below it, unrelated.


# ── Catalog-variant extraction from per-chromosome PGENs ──────────────────────

# hg38 pseudo-autosomal regions, left out of the chrX F statistic and het rate.
_PAR_HG38: tuple[tuple[int, int], ...] = (
    (10_001, 2_781_479),
    (155_701_383, 156_030_895),
)
# SNPs at or below this MAF are not used for F or the het rate.
_CHRX_MAF_MIN: float = 0.05
_CHRX_BLOCK: int = 50_000  # Variants per pgenlib read.

# Genetic sex from two allele-frequency-free axes: the chrX heterozygosity rate (hets
# over called non-PAR SNPs) and the chrY call rate (called genotypes over chrY
# variants). Women have a high X het rate and little Y signal; men have the reverse.
# The cuts below separate the two groups; between each pair of cuts is gray.
SEX_X_HET_ONE: float = 0.05  # Het rate at or below this: one X.
SEX_X_HET_TWO: float = 0.10  # Het rate at or above this: two X.
SEX_Y_ABSENT: float = 0.4  # chrY call rate at or below this: no Y.
SEX_Y_PRESENT: float = 0.6  # chrY call rate at or above this: Y present.


class SexCallThresholds(NamedTuple):
    """The four cuts of the genetic-sex call (``TrialConfig.pqtl_sex_*``)."""

    x_het_one: float = SEX_X_HET_ONE
    x_het_two: float = SEX_X_HET_TWO
    y_absent: float = SEX_Y_ABSENT
    y_present: float = SEX_Y_PRESENT


_COMPLEMENT: dict[str, str] = {"A": "T", "C": "G", "G": "C", "T": "A"}

# "<pre>chr<chrom>[_<start>-<end>]<rest>", e.g. prefix.chr6_1-170805979, prefix_chrX
_PGEN_STEM = re.compile(
    r"^(?P<pre>.*?chr)(?P<chrom>[0-9]+|X|Y|MT|M)"
    r"(?P<chunk>_(?P<start>[0-9]+)-[0-9]+)?(?P<rest>.*)$"
)

# Prints every PVAR data row at a target position: idx, POS, REF, ALT.
# The first file holds the target positions, one per line.
_AWK_SCAN_POSITIONS = (
    'BEGIN{OFS="\\t"} NR==FNR{p[$1]; next} /^#/{next} {idx++} ($2+0) in p {print idx-1, $2, $4, $5}'
)

# Prints idx and POS of every biallelic ACGT SNP row.
_AWK_SCAN_SNPS = (
    'BEGIN{OFS="\\t"} '
    "/^#/{next} "
    "{idx++} "
    "$4 ~ /^[ACGTacgt]$/ && $5 ~ /^[ACGTacgt]$/ {print idx-1, $2}"
)


def normalize_chrom(raw) -> str | None:
    """Returns the bare chromosome name, ``"1"`` to ``"22"`` or ``"X"``.

    Drops a ``chr`` prefix, upper-cases, and maps ``23`` to ``X``. Integer and float
    codes are accepted; None stays None.
    """
    if raw is None:
        return None
    s = str(int(raw)) if isinstance(raw, (int, float)) else str(raw).strip()
    s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).upper()
    return "X" if s == "23" else s


def variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Returns ``"{chrom}:{pos}:{ref}:{alt}"`` with ``chrom`` normalized."""
    return f"{normalize_chrom(chrom)}:{pos}:{ref}:{alt}"


def _chrom_order(chrom: str) -> tuple[int, str]:
    if chrom.isdigit():
        return int(chrom), chrom
    return {"X": 23, "Y": 24}.get(chrom, 25), chrom


def pgen_prefixes(pgen_dir: str | Path, pgen_pattern: str | None = None) -> dict[str, list[str]]:
    """Maps each chromosome to every PGEN prefix for it, sorted by chunk start.

    Files are grouped into naming families such as ``prefix.chr{chrom}_{start}-{end}``
    or ``prefix_chr{chrom}``, written in the ``pgen_pattern`` syntax. One family
    is used: ``pgen_pattern`` when given, otherwise the family that covers the most
    chromosomes (ties go to the lexicographically first).

    Raises:
        FileNotFoundError: If ``pgen_dir`` has no per-chromosome PGEN.
        ValueError: If no PGEN matches ``pgen_pattern``.
    """
    families: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for f in sorted(Path(pgen_dir).glob("*.pgen")):
        m = _PGEN_STEM.match(f.stem)
        if m is None:
            continue
        family = m["pre"] + "{chrom}" + ("_{start}-{end}" if m["chunk"] else "") + m["rest"]
        start = int(m["start"]) if m["start"] else 0
        families.setdefault(family, {}).setdefault(normalize_chrom(m["chrom"]), []).append(
            (start, str(f.with_suffix("")))
        )

    if not families:
        raise FileNotFoundError(f"No per-chromosome PGEN files in {pgen_dir}")
    if pgen_pattern is None:
        family = max(sorted(families), key=lambda k: len(families[k]))
    elif pgen_pattern in families:
        family = pgen_pattern
    else:
        raise ValueError(
            f"No PGEN in {pgen_dir} matches pattern {pgen_pattern!r}; found {sorted(families)}"
        )
    if len(families) > 1:
        logger.info(
            "PGEN family %r used; ignored %s",
            family,
            [k for k in sorted(families) if k != family],
        )
    return {
        chrom: [prefix for _, prefix in sorted(chunks)]
        for chrom, chunks in sorted(families[family].items(), key=lambda kv: _chrom_order(kv[0]))
    }


def scan_pvar_positions(
    pvar_path: str, target_positions: set[int]
) -> list[tuple[int, int, str, str]]:
    """Returns ``(variant_index, POS, REF, ALT)`` of every PVAR row at the positions.

    Runs awk in a subprocess, which streams a large PVAR faster than a Python reader,
    and returns all rows at a position, multiallelic
    ALT lists unsplit, in file order; row choice is the caller's. The positions go to
    awk through a temporary file, so the command line stays short however many there
    are.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".pos") as fh:
        fh.write("\n".join(str(p) for p in sorted(target_positions)) + "\n")
        fh.flush()
        result = subprocess.run(
            ["awk", "-F\t", _AWK_SCAN_POSITIONS, fh.name, pvar_path],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
    matches: list[tuple[int, int, str, str]] = []
    for line in result.stdout.splitlines():
        idx, pos, ref, alt = line.split("\t")
        matches.append((int(idx), int(pos), ref, alt))
    return matches


def read_psam(prefix: str | Path) -> pl.DataFrame:
    """Reads ``IID`` and ``reported_sex`` (1 male, 2 female, 0 unknown) from a PSAM.

    PSAM SEX is the sex supplied with the samples, not a genotype call; genetic sex
    comes from ``infer_genetic_sex``.
    """
    psam = pl.read_csv(
        f"{prefix}.psam",
        separator="\t",
        infer_schema_length=0,
    ).rename({"#IID": "IID"})
    return psam.select(
        "IID",
        pl.col("SEX").cast(pl.Int64, strict=False).fill_null(0).alias("reported_sex"),
    )


def genotype_iids(pgen_dir: str | Path) -> list[str]:
    """Returns the IIDs (USUBJIDs) of the genotype directory's first PSAM, in order."""
    first = next(iter(pgen_prefixes(pgen_dir).values()))[0]
    return read_psam(first)["IID"].to_list()


@dataclass(frozen=True)
class CatalogGenotypes:
    """Trial genotypes at reference-catalog variants.

    Rows of ``dosages`` and ``standardized`` follow ``variant_ids``; columns follow
    ``iids``.

    Attributes:
        dosages: Raw ALT dosages, NaN for missing.
        standardized: The per-variant centered and scaled copy used for scoring.
        variant_ids: ``variant_key`` of each PVAR row.
        alleles: Variant ID to its PVAR ``(REF, ALT)``.
        catalog_variant: One row per catalog ``(chrom, pos, a1, a2)`` that matched a
            PVAR row, with that row's ``variant_id``; ``chrom`` is normalized and
            ``a1``/``a2`` are the catalog's values.
        iids: Genotype IIDs (USUBJIDs).
        genetic_sex: ``infer_genetic_sex`` for ``iids``.
        targets: Every catalog ``(chrom, pos, a1, a2)`` the read looked for, matched or
            not (``catalog_targets``).
    """

    dosages: np.ndarray
    standardized: np.ndarray
    variant_ids: list[str]
    alleles: dict[str, tuple[str, str]]
    catalog_variant: pl.DataFrame
    iids: list[str]
    genetic_sex: pl.DataFrame
    targets: pl.DataFrame | None = None


def standardize_dosages(dosages: np.ndarray) -> np.ndarray:
    """Returns per-variant (row) centered, scaled float32 dosages, missing set to 0."""
    mu = np.nanmean(dosages, axis=1, keepdims=True)
    sd = np.nanstd(dosages, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nan_to_num((dosages - mu) / sd).astype(np.float32)


def catalog_targets(catalog: pl.DataFrame) -> pl.DataFrame:
    """Returns the distinct catalog ``(chrom, pos, a1, a2)`` a genotype read looks for.

    ``chrom`` is normalized (``normalize_chrom``), ``pos`` Int64, ``a1``/``a2`` the
    catalog's strings; rows without a chromosome, position or effect allele are left
    out.
    """
    raw_chroms = [c for c in catalog["chrom"].unique().to_list() if c is not None]
    return (
        catalog.select(
            pl.col("chrom").replace_strict(
                {c: normalize_chrom(c) for c in raw_chroms}, return_dtype=pl.String
            ),
            pl.col("pos").cast(pl.Int64),
            pl.col("a1", "a2").cast(pl.String),
        )
        .filter(
            pl.col("chrom").is_not_null() & pl.col("pos").is_not_null() & pl.col("a1").is_not_null()
        )
        .unique(maintain_order=True)
    )


def _select_pvar_rows(targets: pl.DataFrame, rows: pl.DataFrame, chrom: str) -> pl.DataFrame:
    """Picks the PVAR row for each catalog ``(pos, a1, a2)``.

    Only biallelic rows whose REF and ALT are ``[ACGT]`` strings are candidates, so a
    spanning deletion ``*``, a symbolic ALT or an ``N`` never is. A single-base (SNP)
    row matches a target with ``a2`` when its {REF, ALT} equals {a1, a2} or its strand
    complement. It matches a target without ``a2`` (an effect-allele-only catalog,
    forward strand) only when a1 itself is REF or ALT: with one allele known, a
    complement match is another variant at the position, not a strand flip. A/T and C/G
    rows are dropped for effect-allele-only targets. An indel row matches only a target
    with ``a2``, on exactly {REF, ALT} == {a1, a2}.

    Args:
        targets: Catalog targets of one chromosome (``pos``, ``a1``, ``a2``).
        rows: PVAR rows at the targets' positions (``idx``, ``pos``, ``ref``, ``alt``).
        chrom: The chromosome, for messages.

    Returns:
        One row per placed target with its PVAR row, sorted by ``pos``, ``a1``, ``a2``.
        A target with no candidate is unmatched; an indel or effect-allele-only target
        with two candidates (a split multiallelic site) is dropped with a count logged.

    Raises:
        ValueError: If two PVAR rows match a SNP target with both alleles (a duplicated
            row).
    """
    ref, alt = pl.col("ref").str.to_uppercase(), pl.col("alt").str.to_uppercase()
    biallelic = rows.with_columns(ref.alias("_R"), alt.alias("_A")).filter(
        pl.col("_R").str.contains(r"^[ACGT]+$") & pl.col("_A").str.contains(r"^[ACGT]+$")
    )

    R, A = pl.col("_R"), pl.col("_A")
    is_snp = (R.str.len_chars() == 1) & (A.str.len_chars() == 1)
    a1 = pl.col("a1").str.to_uppercase()
    a2 = pl.col("a2").str.to_uppercase()
    c1, c2 = a1.replace(_COMPLEMENT), a2.replace(_COMPLEMENT)

    def same_pair(x: pl.Expr, y: pl.Expr) -> pl.Expr:
        return ((R == x) & (A == y)) | ((R == y) & (A == x))

    with_a2 = same_pair(a1, a2) | (is_snp & same_pair(c1, c2))
    # One allele known: its complement is another variant here, not a strand flip.
    a1_only = is_snp & ((R == a1) | (A == a1)) & (R != A.replace(_COMPLEMENT))
    keys = ["pos", "a1", "a2"]
    cand = (
        targets.join(biallelic, on="pos", how="inner")
        .filter(pl.when(pl.col("a2").is_null()).then(a1_only).otherwise(with_a2).fill_null(False))
        .with_columns(pl.len().over(keys).alias("_n"))
    )

    multi = cand.filter(pl.col("_n") > 1)
    target_snp = (pl.col("a1").str.len_chars() == 1) & (pl.col("a2").str.len_chars() == 1)
    clash = multi.filter(pl.col("a2").is_not_null() & target_snp)
    if clash.height:
        raise ValueError(
            "several PVAR rows match the catalog alleles at "
            f"{sorted({f'{chrom}:{p}' for p in clash['pos'].to_list()})}"
        )
    n_ambiguous = multi.filter(pl.col("a2").is_null()).select(keys).unique().height
    if n_ambiguous:
        logger.info(
            "chr%s: %d effect-allele-only catalog variants match more than one PVAR row; dropped",
            chrom,
            n_ambiguous,
        )
    # A split multiallelic site can repeat an indel pair in both orientations (G>GA and
    # GA>G), and a duplicated row repeats it outright: either way the catalog row can't
    # be placed.
    n_indel = multi.filter(pl.col("a2").is_not_null()).select(keys).unique().height
    if n_indel:
        logger.info(
            "chr%s: %d indel catalog variants dropped: each matches more than one "
            "PVAR row with the same alleles (split multiallelic site or duplicated row)",
            chrom,
            n_indel,
        )
    return cand.filter(pl.col("_n") == 1).drop("_n", "_R", "_A").sort(keys, nulls_last=True)


def extract_catalog_genotypes(
    catalog: pl.DataFrame,
    pgen_dir: str | Path,
    *,
    pgen_pattern: str | None = None,
    sex_thresholds: SexCallThresholds | None = None,
) -> CatalogGenotypes:
    """Reads the catalog's variants from per-chromosome PGENs.

    Catalog rows are matched on ``chrom``/``pos`` (both hg38) and then on alleles
    (``a1`` the effect allele, ``a2`` the other allele or null) by
    ``_select_pvar_rows``, so the rsID plays no part: null-rsID catalog rows are kept. Every chunk of a
    chromosome in the chosen PGEN family (``pgen_prefixes``) is scanned. Dosages are ALT
    dosages of the matched row; ``pqtl_reference.align_catalog_alleles`` orients the
    betas.

    Args:
        catalog: Catalog rows with ``chrom``, ``pos``, ``a1`` and ``a2``.
        pgen_dir: Directory of per-chromosome PGENs.
        pgen_pattern: PGEN family to use (``pgen_prefixes``).
        sex_thresholds: Cuts of the genetic-sex call; ``SexCallThresholds()`` when
            None.

    Raises:
        ValueError: If two PVAR rows match a SNP target with both alleles, or the
            chunks' PSAM sample orders differ.
        RuntimeError: If no catalog variant is found in the PGENs.
    """
    import pgenlib

    pgens = pgen_prefixes(pgen_dir, pgen_pattern)
    targets_all = catalog_targets(catalog)
    cat_chroms = set(targets_all["chrom"].to_list())
    if cat_chroms - set(pgens):
        logger.info(
            "catalog chromosomes with no PGEN: %s",
            sorted(cat_chroms - set(pgens), key=_chrom_order),
        )
    chroms = sorted(cat_chroms & set(pgens), key=_chrom_order)

    blocks: list[np.ndarray] = []
    variant_ids: list[str] = []
    alleles: dict[str, tuple[str, str]] = {}
    matched: list[pl.DataFrame] = []
    iids: list[str] | None = None
    row_schema = {"idx": pl.Int64, "pos": pl.Int64, "ref": pl.String, "alt": pl.String}

    for chrom in chroms:
        targets = targets_all.filter(pl.col("chrom") == chrom).drop("chrom")
        positions = set(targets["pos"].to_list())
        rows = pl.concat(
            [
                pl.DataFrame(
                    scan_pvar_positions(f"{prefix}.pvar", positions),
                    schema=row_schema,
                    orient="row",
                ).with_columns(pl.lit(k).alias("chunk"))
                for k, prefix in enumerate(pgens[chrom])
            ]
        )
        sel = _select_pvar_rows(targets, rows, chrom)
        if sel.height == 0:
            logger.warning("chr%s: no catalog variants in PVAR", chrom)
            continue

        var = sel.select("chunk", "idx", "pos", "ref", "alt").unique().sort("chunk", "idx")
        vids = [
            variant_key(chrom, p, r, a) for p, r, a in var.select("pos", "ref", "alt").iter_rows()
        ]
        var = var.with_columns(pl.Series("variant_id", vids, dtype=pl.String))
        for k, prefix in enumerate(pgens[chrom]):
            part = var.filter(pl.col("chunk") == k)
            if part.height == 0:
                continue
            chunk_iids = read_psam(prefix)["IID"].to_list()
            if iids is None:
                iids = chunk_iids
            elif chunk_iids != iids:
                raise ValueError(f"{prefix}.psam sample order differs from the first PGEN read")
            reader = pgenlib.PgenReader(bytes(f"{prefix}.pgen", encoding="utf-8"))
            block = np.empty((part.height, reader.get_raw_sample_ct()), dtype=np.float32)
            reader.read_dosages_list(part["idx"].to_numpy().astype(np.uint32), block)
            reader.close()
            block[block < -0.5] = np.nan  # pgenlib's missing sentinel is -9.
            blocks.append(block)

        variant_ids.extend(var["variant_id"].to_list())
        alleles.update(
            (v, (r, a)) for v, r, a in var.select("variant_id", "ref", "alt").iter_rows()
        )
        matched.append(
            sel.join(var.select("chunk", "idx", "variant_id"), on=["chunk", "idx"])
            .select(pl.lit(chrom).alias("chrom"), "pos", "a1", "a2", "variant_id")
            .sort("pos", "a1", "a2", nulls_last=True)
        )
        logger.info(
            "chr%s: %d catalog variants -> %d PVAR rows",
            chrom,
            sel.height,
            var.height,
        )

    if not blocks:
        raise RuntimeError(
            f"No catalog variants found in the PGENs under {pgen_dir}: catalog "
            f"chromosomes {sorted(cat_chroms, key=_chrom_order)}, PGEN chromosomes "
            f"{sorted(pgens, key=_chrom_order)}"
        )

    dosages = np.concatenate(blocks, axis=0)
    standardized = standardize_dosages(dosages)
    logger.info("Catalog genotypes: %d variants x %d samples", *dosages.shape)
    return CatalogGenotypes(
        dosages=dosages,
        standardized=standardized,
        variant_ids=variant_ids,
        alleles=alleles,
        catalog_variant=pl.concat(matched),
        iids=iids,
        genetic_sex=_genetic_sex(pgens.get("X", []), pgens.get("Y", []), iids, sex_thresholds),
        targets=targets_all,
    )


_TARGET_KEYS = ["chrom", "pos", "a1", "a2"]


# ── Genetic sex from chrX heterozygosity and chrY calls ──


def infer_genetic_sex(
    pgen_dir: str | Path,
    iids: list[str],
    *,
    pgen_pattern: str | None = None,
    thresholds: SexCallThresholds | None = None,
) -> pl.DataFrame:
    """Calls genetic sex from the chrX heterozygosity rate and the chrY call rate.

    The call uses two axes that need no allele frequencies, so ancestry cannot bias it.
    ``chrx_het_rate`` is the heterozygous fraction of the called biallelic non-PAR chrX
    SNPs with MAF > 0.05 (hg38 PAR1 10,001-2,781,479 and PAR2 155,701,383-156,030,895
    are excluded; the MAF is the ``iids`` samples' pooled allele frequency, a site
    filter only). ``chry_call_rate`` is the fraction of chrY variants with a called
    genotype. X is one copy at a het rate of ``x_het_one`` or less and two copies at
    ``x_het_two`` or more; Y is absent at ``y_absent`` or less and present at
    ``y_present`` or more.

    ``sex_karyotype`` is XX (two X, no Y: ``genetic_sex`` 2), XY (one X and Y: 1), XXY
    and X0 (``genetic_sex`` null: no sex is claimed for them), or ``undetermined`` (a
    gray zone on either axis, or no chrY call for an IID that has chrX; null genetic
    sex). An IID absent from the chrX PSAM, or a trial with no chrX PGEN, gives null in
    every column. A trial with no chrY PGEN is called from the X axis alone
    (``sex_karyotype`` ``no_chrY``, a warning is logged, XXY and X0 cannot be told from
    XX and XY).

    ``chrx_f`` is the inbreeding coefficient F = 1 - O(het) / E(het) over the same SNPs,
    with allele frequencies pooled over the ``iids`` samples (as plink ``--check-sex``).
    It is reported only; it does not enter the call (pooled frequencies bias it for
    minority ancestries).

    Args:
        pgen_dir: Directory of per-chromosome PGENs.
        iids: Genotype IIDs to call.
        pgen_pattern: PGEN family to use (``pgen_prefixes``).
        thresholds: Cuts of the call; ``SexCallThresholds()`` when None.

    Returns:
        One row per IID of ``iids``, in order: ``IID``, ``genetic_sex`` (1 male, 2
        female, null no call), ``chrx_f``, ``chrx_het_rate``, ``chry_call_rate`` and
        ``sex_karyotype``.
    """
    pgens = pgen_prefixes(pgen_dir, pgen_pattern)
    return _genetic_sex(pgens.get("X", []), pgens.get("Y", []), list(iids), thresholds)


def _chrx_snp_rows(pvar_path: str) -> np.ndarray:
    """Returns the PVAR row indices of biallelic SNPs outside the hg38 PARs."""
    result = subprocess.run(
        ["awk", "-F\t", _AWK_SCAN_SNPS, pvar_path],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    idx_pos = np.array(result.stdout.split(), dtype=np.int64).reshape(-1, 2)
    in_par = np.zeros(idx_pos.shape[0], dtype=bool)
    for lo, hi in _PAR_HG38:
        in_par |= (idx_pos[:, 1] >= lo) & (idx_pos[:, 1] <= hi)
    return idx_pos[~in_par, 0].astype(np.uint32)


SEX_COLUMNS: tuple[str, ...] = (
    "genetic_sex",
    "chrx_f",
    "chrx_het_rate",
    "chry_call_rate",
    "sex_karyotype",
)
"""The per-genotype sex columns ``infer_genetic_sex`` gives beside ``IID``; the DNA-link
table carries them."""

_SEX_SCHEMA = {
    "IID": pl.String,
    "genetic_sex": pl.Int64,
    "chrx_f": pl.Float64,
    "chrx_het_rate": pl.Float64,
    "chry_call_rate": pl.Float64,
    "sex_karyotype": pl.String,
}


def _chry_call_rates(prefixes: list[str], iids: list[str]) -> np.ndarray:
    """Returns each IID's fraction of chrY variants with a called genotype.

    NaN marks an IID that is not in the chrY PSAM.
    """
    import pgenlib

    n_called = np.zeros(len(iids))
    n_variants = np.zeros(len(iids))
    for prefix in prefixes:
        col_of = {iid: j for j, iid in enumerate(read_psam(prefix)["IID"].to_list())}
        rows = np.array([i for i, iid in enumerate(iids) if iid in col_of], dtype=np.int64)
        cols = np.array([col_of[iids[i]] for i in rows], dtype=np.int64)
        reader = pgenlib.PgenReader(bytes(f"{prefix}.pgen", encoding="utf-8"))
        n_var = reader.get_variant_ct()
        buf = np.empty((_CHRX_BLOCK, reader.get_raw_sample_ct()), dtype=np.int8)
        for start in range(0, n_var, _CHRX_BLOCK):
            vi = np.arange(start, min(start + _CHRX_BLOCK, n_var), dtype=np.uint32)
            reader.read_list(vi, buf[: vi.size])
            # pgenlib's missing hard call is -9.
            n_called[rows] += (buf[: vi.size][:, cols] >= 0).sum(axis=0)
        reader.close()
        n_variants[rows] += n_var
    rate = np.full(len(iids), np.nan)
    np.divide(n_called, n_variants, out=rate, where=n_variants > 0)
    return rate


def _call_karyotype(
    het_rate: float, y_rate: float, has_chry: bool, t: SexCallThresholds
) -> tuple[str | None, int | None]:
    """Returns ``(sex_karyotype, genetic_sex)`` from one sample's two rates.

    See ``infer_genetic_sex``.
    """
    if np.isnan(het_rate):
        return None, None
    one, two = het_rate <= t.x_het_one, het_rate >= t.x_het_two
    if not has_chry:
        if one:
            return "no_chrY", 1
        if two:
            return "no_chrY", 2
        return "undetermined", None
    if np.isnan(y_rate):
        return "undetermined", None
    absent, present = y_rate <= t.y_absent, y_rate >= t.y_present
    if two and absent:
        return "XX", 2
    if one and present:
        return "XY", 1
    if two and present:
        return "XXY", None
    if one and absent:
        return "X0", None
    return "undetermined", None


def with_sex_call(
    sex: pl.DataFrame, thresholds: SexCallThresholds, *, has_chry: bool | None = None
) -> pl.DataFrame:
    """Re-calls ``genetic_sex`` and ``sex_karyotype`` from the stored rates.

    The genotype cache stores the rates, so a run whose thresholds differ from the ones
    the cache was built with calls again from them.

    Args:
        sex: ``infer_genetic_sex`` output.
        thresholds: Cuts of the call.
        has_chry: Whether the trial has a chrY PGEN; by default, whether any IID has a
            ``chry_call_rate``.

    Returns:
        ``sex`` with ``genetic_sex`` and ``sex_karyotype`` called from its
        ``chrx_het_rate`` and ``chry_call_rate`` at ``thresholds``.
    """
    if has_chry is None:
        has_chry = sex["chry_call_rate"].null_count() < sex.height
    calls = [
        _call_karyotype(
            h if h is not None else np.nan,
            y if y is not None else np.nan,
            has_chry,
            thresholds,
        )
        for h, y in zip(
            sex["chrx_het_rate"].to_list(), sex["chry_call_rate"].to_list(), strict=True
        )
    ]
    return sex.with_columns(
        pl.Series("genetic_sex", [c[1] for c in calls], dtype=pl.Int64),
        pl.Series("sex_karyotype", [c[0] for c in calls], dtype=pl.String),
    ).select(list(sex.columns))


def _genetic_sex(
    x_prefixes: list[str],
    y_prefixes: list[str],
    iids: list[str],
    thresholds: SexCallThresholds | None = None,
) -> pl.DataFrame:
    """Returns ``infer_genetic_sex`` from the chrX and chrY PGEN prefixes."""
    t = SexCallThresholds() if thresholds is None else thresholds
    if not x_prefixes:
        logger.info("no chrX PGEN; genetic sex is null for all %d samples", len(iids))
        empty = [None] * len(iids)
        return pl.DataFrame(
            {name: iids if name == "IID" else empty for name in _SEX_SCHEMA},
            schema=_SEX_SCHEMA,
        )
    if not y_prefixes:
        logger.warning(
            "no chrY PGEN; genetic sex is called from the chrX heterozygosity rate alone for %d "
            "samples (47,XXY and 45,X cannot be told from XX and XY)",
            len(iids),
        )

    import pgenlib

    o_het = np.zeros(len(iids))
    e_het = np.zeros(len(iids))
    n_used = np.zeros(len(iids))
    n_snps = 0
    for prefix in x_prefixes:
        col_of = {iid: j for j, iid in enumerate(read_psam(prefix)["IID"].to_list())}
        rows = np.array([i for i, iid in enumerate(iids) if iid in col_of], dtype=np.int64)
        cols = np.array([col_of[iids[i]] for i in rows], dtype=np.int64)
        snp_idx = _chrx_snp_rows(f"{prefix}.pvar")
        reader = pgenlib.PgenReader(bytes(f"{prefix}.pgen", encoding="utf-8"))
        buf = np.empty((_CHRX_BLOCK, reader.get_raw_sample_ct()), dtype=np.int8)
        for start in range(0, snp_idx.size, _CHRX_BLOCK):
            vi = snp_idx[start : start + _CHRX_BLOCK]
            reader.read_list(vi, buf[: vi.size])
            g = buf[: vi.size][:, cols]
            called = g >= 0  # pgenlib's missing hard call is -9.
            n_called = called.sum(axis=1)
            p = np.where(called, g, 0).sum(axis=1) / np.maximum(2 * n_called, 1)
            keep = (n_called > 0) & (np.minimum(p, 1 - p) > _CHRX_MAF_MIN)
            g, called, p = g[keep], called[keep], p[keep]
            o_het[rows] += (g == 1).sum(axis=0)
            e_het[rows] += (called * (2 * p * (1 - p))[:, None]).sum(axis=0)
            n_used[rows] += called.sum(axis=0)
            n_snps += int(keep.sum())
        reader.close()

    f = np.full(len(iids), np.nan)
    np.divide(e_het - o_het, e_het, out=f, where=e_het > 0)
    het_rate = np.full(len(iids), np.nan)
    np.divide(o_het, n_used, out=het_rate, where=n_used > 0)
    y_rate = _chry_call_rates(y_prefixes, iids) if y_prefixes else np.full(len(iids), np.nan)

    out = pl.DataFrame(
        {
            "IID": iids,
            "genetic_sex": None,
            "chrx_f": f,
            "chrx_het_rate": het_rate,
            "chry_call_rate": y_rate,
            "sex_karyotype": None,
        },
        schema=_SEX_SCHEMA,
    ).with_columns(pl.col("chrx_f", "chrx_het_rate", "chry_call_rate").fill_nan(None))
    out = with_sex_call(out, t, has_chry=bool(y_prefixes))
    logger.info(
        "genetic sex on %d chrX SNPs and %s: %d male, %d female, %d no call; karyotypes %s",
        n_snps,
        f"{len(y_prefixes)} chrY file(s)" if y_prefixes else "no chrY",
        (out["genetic_sex"] == 1).sum(),
        (out["genetic_sex"] == 2).sum(),
        out["genetic_sex"].null_count(),
        dict(sorted(out["sex_karyotype"].drop_nulls().value_counts().iter_rows())),
    )
    return out


def king_kinship(dosages: np.ndarray, i: int, j: int) -> float:
    """Returns the KING-robust kinship between genotype columns ``i`` and ``j``.

    The kinship is ``(N_het,het - 2 N_opposite-hom) / (N_het_i + N_het_j)`` over
    variants called in both (Manichaikul et al. 2010); NaN when neither has a
    heterozygous call there. Duplicates give ~0.5, first-degree relatives ~0.25,
    unrelated pairs ~0 or below.
    """
    a = np.rint(dosages[:, i])
    b = np.rint(dosages[:, j])
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    het_a, het_b = a == 1, b == 1
    denom = het_a.sum() + het_b.sum()
    if denom == 0:
        return float("nan")
    opposite = ((a == 0) & (b == 2)) | ((a == 2) & (b == 0))
    return float(((het_a & het_b).sum() - 2 * opposite.sum()) / denom)


def kinship_matrix(dosages: np.ndarray) -> np.ndarray:
    """Returns the KING-robust kinship between every pair of genotype columns.

    The (n, n) result equals ``king_kinship`` for every pair (variants called in both;
    NaN when neither has a heterozygous call there) and is computed with
    indicator-matrix products. The diagonal is
    0.5 (NaN for a genotype with no heterozygous call).
    """
    d = np.rint(np.asarray(dosages, dtype=np.float64))
    called = np.isfinite(d)
    het = ((d == 1) & called).astype(np.float32)
    hom0 = ((d == 0) & called).astype(np.float32)
    hom2 = ((d == 2) & called).astype(np.float32)
    called = called.astype(np.float32)
    het_het = (het.T @ het).astype(np.float64)
    opposite = (hom0.T @ hom2).astype(np.float64)
    opposite = opposite + opposite.T
    # Het calls of i among variants called in both i and j, plus those of j.
    denom = (het.T @ called).astype(np.float64)
    denom = denom + denom.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (het_het - 2 * opposite) / denom, np.nan)


# ── Per-sample concordance ────────────────────────────────────────────────────

PQTL_BEST_MIN: float = 4.0
"""Smallest best-other pQTL z that names a subject (``pqtl_best_USUBJID``).

The default of ``extract_concordance(best_min=...)``; ``TrialConfig.pqtl_best_min``
defaults to the platform's instead (``Platform.pqtl_defaults``: 5.0 for both Olink and
SomaScan). Below it the best other genotype is the maximum of the null and identifies no
one."""

_CONCORDANCE_SCHEMA: dict[str, pl.DataType] = {
    "USUBJID": pl.String,
    "pqtl_self_z": pl.Float64,
    "pqtl_best_z": pl.Float64,
    "pqtl_best_USUBJID": pl.String,
    "pqtl_best_nonrel_z": pl.Float64,
    "pqtl_best_nonrel_USUBJID": pl.String,
}


def extract_concordance(
    S: np.ndarray,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str],
    *,
    best_min: float = PQTL_BEST_MIN,
    sample_ids: list[str] | None = None,
    relative: np.ndarray | None = None,
    best_relative: bool = False,
) -> pl.DataFrame:
    """Extracts per-sample self and best-other z-scores from a score matrix.

    Every sample with a finite score is reported, including samples whose labeled
    subject is not genotyped: their ``pqtl_self_z`` is null and their best other
    genotype is the best over all genotypes. ``pqtl_best_USUBJID`` is named only when
    ``pqtl_best_z`` reaches ``best_min``. The calls (match, mismatch, none) are made by
    ``classify_pqtl``.

    ``pqtl_best_nonrel_z`` and ``pqtl_best_nonrel_USUBJID`` are the same best-other
    score restricted to genotypes that ``relative`` does not mark as relatives of the
    sample's labeled genotype; a relative is never named as a donor. Without
    ``relative`` they equal the unrestricted best.

    Args:
        S: Robust z of each sample against each genotype, shape (n_samples,
            n_genotyped).
        sample_usubjids: USUBJID per sample (rows of ``S``).
        geno_usubjids: USUBJID per genotyped participant (columns of ``S``): every
            genotype that can be named, whether or not its participant has proteomics.
            A sample's own genotype is never its own best other.
        best_min: Smallest best-other z that names a subject.
        sample_ids: Sample ID per row of ``S``. When given, the output leads with a
            ``SampleID`` column so callers can join per sample.
        relative: Boolean, shape (n_samples, n_genotyped): True where genotype ``j`` is
            a relative (or duplicate) of sample ``i``'s labeled genotype; all False for
            an ungenotyped label.
        best_relative: Whether to add ``pqtl_best_rel_z``, the highest finite z against
            a genotype that ``relative`` marks, the labeled genotype itself excluded
            (null when there is none, always for an ungenotyped label), and
            ``pqtl_best_rel_USUBJID``, that relative, named only when its z reaches
            ``best_min``. Needs ``relative``. ``classify_pqtl`` reads them when
            ``pqtl_relative_margin`` or ``pqtl_relative_name_margin`` is set.

    Returns:
        One row per scored sample: [``SampleID``], ``USUBJID``, ``pqtl_self_z``,
        ``pqtl_best_z``, ``pqtl_best_USUBJID``, ``pqtl_best_nonrel_z``,
        ``pqtl_best_nonrel_USUBJID`` [, ``pqtl_best_rel_z``, ``pqtl_best_rel_USUBJID``].

    Raises:
        ValueError: If ``relative`` does not have the shape of ``S``, or
            ``best_relative`` is set without ``relative``.
    """
    if relative is not None and relative.shape != S.shape:
        raise ValueError(f"relative has shape {relative.shape}, expected {S.shape}")
    if best_relative and relative is None:
        raise ValueError("best_relative needs the relative mask")
    g_index = {p: i for i, p in enumerate(geno_usubjids)}
    geno_ids = np.array(geno_usubjids)

    def best_of(values: np.ndarray, ids: np.ndarray) -> tuple[float | None, str | None]:
        finite = np.where(np.isfinite(values), values, -np.inf)
        if not np.isfinite(finite).any():
            return None, None
        k = int(np.argmax(finite))
        z = float(finite[k])
        return z, (str(ids[k]) if z >= best_min else None)

    rows: list[dict] = []
    for i in range(len(sample_usubjids)):
        subj = sample_usubjids[i]
        own_j = g_index.get(subj)
        others = np.arange(len(geno_usubjids)) != (-1 if own_j is None else own_j)
        if own_j is None:
            if not np.isfinite(S[i, others]).any():
                continue
            self_z = None
        elif np.isnan(S[i, own_j]):
            continue
        else:
            self_z = float(S[i, own_j])
        best_z, best_other = best_of(S[i, others], geno_ids[others])
        if relative is not None:
            nonrel = others & ~relative[i]
        else:
            nonrel = others
        nonrel_z, nonrel_other = best_of(S[i, nonrel], geno_ids[nonrel])

        row = ({"SampleID": sample_ids[i]} if sample_ids is not None else {}) | {
            "USUBJID": subj,
            "pqtl_self_z": self_z,
            "pqtl_best_z": best_z,
            "pqtl_best_USUBJID": best_other,
            "pqtl_best_nonrel_z": nonrel_z,
            "pqtl_best_nonrel_USUBJID": nonrel_other,
        }
        if best_relative:
            rel = others & relative[i]
            rel_z = S[i, rel]
            rel_z = rel_z[np.isfinite(rel_z)]
            row["pqtl_best_rel_z"] = float(rel_z.max()) if rel_z.size else None
            row["pqtl_best_rel_USUBJID"] = best_of(S[i, rel], geno_ids[rel])[1]
        rows.append(row)

    schema = (
        ({"SampleID": pl.String} if sample_ids is not None else {})
        | _CONCORDANCE_SCHEMA
        | (
            {"pqtl_best_rel_z": pl.Float64, "pqtl_best_rel_USUBJID": pl.String}
            if best_relative
            else {}
        )
    )
    return pl.DataFrame(rows, schema=schema)


def classify_pqtl(
    scores: pl.DataFrame,
    kinship: np.ndarray,
    geno_usubjids: list[str],
    fit: np.ndarray | pl.Series,
    config,
) -> pl.DataFrame:
    """Adds the per-tube pQTL call and its kinship columns to ``scores``.

    The added columns are ``pqtl_best_kinship``, ``pqtl_call`` (match, mismatch, none)
    and ``pqtl_relative_USUBJID``. ``pqtl_best_kinship`` is null when the label or the
    best genotype is not in ``kinship``, or ``kinship`` is NaN for the pair: a pair the
    KING table does not list.

    Thresholds are the ``pqtl_*`` fields of ``config``. A relative is a genotype with
    KING kinship to the labeled genotype of at least ``pqtl_relative_kinship``; an
    ungenotyped label has no kinship, so no candidate counts as a relative. A named
    donor is any genotype, with or without proteomics. A donor that ``kinship`` does
    not cover (not in ``geno_usubjids``) has no kinship to anyone, so it is a
    non-relative, ``pqtl_best_kinship`` is null and the relative block (below) does not
    apply to it.
    The pipeline's kinship covers every genotype it can name, so this is the contract
    for a caller that restricts ``kinship``, not a case the pipeline meets.

    ``pqtl_call`` is one of:

    - ``mismatch``, named: the best non-relative other reaches ``pqtl_best_min``
      and either the label is genotyped and ``best_nonrel_z - self_z`` exceeds
      ``pqtl_margin_min``, or the label is ungenotyped and (under
      ``pqtl_ungenotyped_all_tubes``) every fit tube of the subject names one
      and the same genotype (``pqtl_ungenotyped_naming = "all_tubes"``), or, under
      ``"two_agree"``, two or more fit tubes name one
      genotype, no fit tube names another and this tube names that one.
    - ``mismatch`` naming a relative (only when ``pqtl_relative_name_margin`` is set): the label is genotyped, the best relative (``pqtl_best_rel_z``, needs
      ``extract_concordance(best_relative=True)``) reaches ``pqtl_best_min``, is at
      least ``pqtl_best_nonrel_z``, beats ``pqtl_self_z`` by the margin and is not a
      duplicate genome of the label (kinship below ``pqtl_duplicate_kinship``). The
      relative is ``pqtl_relative_USUBJID`` (null on every other tube); identity reads
      it as the named donor.
    - ``mismatch``, unnamed: the label is genotyped and ``pqtl_self_z`` is below
      ``pqtl_mismatch_floor``.
    - ``match``: the label is genotyped, ``pqtl_self_z`` is at least
      ``pqtl_match_floor``, the tube is no mismatch, and the best other genotype
      is not a relative at z of ``pqtl_best_min`` or more (the relative block).  When
      ``pqtl_relative_margin`` is set a match also needs
      ``pqtl_self_z - pqtl_best_rel_z >= pqtl_relative_margin``, where
      ``pqtl_best_rel_z`` (``extract_concordance(best_relative=True)``) is the
      best z against any relative of the labeled genotype; a label with no
      relative z passes.  This adds to the relative block and never lifts it.  Unset
      (None), the rule is the relative block alone.
    - ``none``: everything else. It never vetoes, resolves, keeps or fails a tube.

    Args:
        scores: ``extract_concordance`` output computed with the relative mask of
            ``kinship``.
        kinship: KING kinship between the genotypes, in ``geno_usubjids`` order (NaN
            for a pair below every cut).
        geno_usubjids: USUBJID per genotype.
        fit: Boolean per row of ``scores``: the tube passed technical QC.
        config: The trial config.

    Raises:
        ValueError: If ``fit`` or ``kinship`` has the wrong size, or a relative margin
            is set without the ``extract_concordance(best_relative=True)`` columns.
    """
    fit = pl.Series("fit", np.asarray(fit, dtype=bool))
    if fit.len() != scores.height:
        raise ValueError(f"fit has {fit.len()} entries for {scores.height} score rows")
    if kinship.shape != (len(geno_usubjids),) * 2:
        raise ValueError(f"kinship has shape {kinship.shape} for {len(geno_usubjids)} genotypes")
    g_index = {u: j for j, u in enumerate(geno_usubjids)}
    own = np.array([g_index.get(u, -1) for u in scores["USUBJID"].to_list()], dtype=np.int64)
    best = np.array(
        [
            g_index.get(u, -1) if u is not None else -1
            for u in scores["pqtl_best_USUBJID"].to_list()
        ],
        dtype=np.int64,
    )
    both = (own >= 0) & (best >= 0)
    best_kinship = np.full(scores.height, np.nan)
    best_kinship[both] = kinship[own[both], best[both]]
    scores = scores.with_columns(
        pl.Series("pqtl_best_kinship", best_kinship).fill_nan(None), fit.alias("_fit")
    )

    donor_col = pl.col("pqtl_best_nonrel_USUBJID")
    two_agree = config.pqtl_ungenotyped_naming == "two_agree"
    if two_agree:
        # two_agree: two or more fit tubes name one genotype and no fit tube names
        # another.
        agree = (
            scores.filter(pl.col("_fit"))
            .group_by("USUBJID")
            .agg(
                (
                    (donor_col.is_not_null().sum() >= 2) & (donor_col.drop_nulls().n_unique() == 1)
                ).alias("_all_same"),
                donor_col.drop_nulls().first().alias("_agreed"),
            )
        )
        scores = scores.join(agree, on="USUBJID", how="left", maintain_order="left")
    else:
        # all_tubes: does every fit tube of an ungenotyped subject name one and the
        # same genotype?
        same = (
            scores.filter(pl.col("_fit"))
            .group_by("USUBJID")
            .agg(((donor_col.null_count() == 0) & (donor_col.n_unique() == 1)).alias("_all_same"))
        )
        scores = scores.join(same, on="USUBJID", how="left", maintain_order="left")

    genotyped = pl.col("pqtl_self_z").is_not_null()
    named = donor_col.is_not_null()
    all_same = (
        pl.col("_all_same").fill_null(False) if config.pqtl_ungenotyped_all_tubes else pl.lit(True)
    )
    if two_agree:
        all_same = all_same & (donor_col == pl.col("_agreed"))
    mismatch = (
        (
            genotyped
            & named
            & ((pl.col("pqtl_best_nonrel_z") - pl.col("pqtl_self_z")) > config.pqtl_margin_min)
        )
        | (genotyped & (pl.col("pqtl_self_z") < config.pqtl_mismatch_floor))
        | (~genotyped & named & all_same)
    ).fill_null(False)
    relative_named = pl.lit(False)
    if config.pqtl_relative_name_margin is not None:
        # Relative naming: the best relative, not a duplicate genome, fits at
        # least best_min, at least as well as any non-relative, and beats the tube's own
        # genotype by the margin.
        missing = {"pqtl_best_rel_z", "pqtl_best_rel_USUBJID"} - set(scores.columns)
        if missing:
            raise ValueError(
                "pqtl_relative_name_margin is set, so the scores need "
                f"{', '.join(sorted(missing))} (extract_concordance(..., best_relative=True))"
            )
        rel = np.array(
            [
                g_index.get(u, -1) if u is not None else -1
                for u in scores["pqtl_best_rel_USUBJID"].to_list()
            ],
            dtype=np.int64,
        )
        rel_kinship = np.full(scores.height, np.nan)
        pair = (own >= 0) & (rel >= 0)
        rel_kinship[pair] = kinship[own[pair], rel[pair]]
        scores = scores.with_columns(pl.Series("_rel_kinship", rel_kinship).fill_nan(None))
        relative_named = (
            genotyped
            & pl.col("pqtl_best_rel_USUBJID").is_not_null()
            & (pl.col("pqtl_best_rel_z") >= config.pqtl_best_min)
            & (
                pl.col("pqtl_best_nonrel_z").is_null()
                | (pl.col("pqtl_best_rel_z") >= pl.col("pqtl_best_nonrel_z"))
            )
            & (
                (pl.col("pqtl_best_rel_z") - pl.col("pqtl_self_z"))
                >= config.pqtl_relative_name_margin
            )
            & (pl.col("_rel_kinship") < config.pqtl_duplicate_kinship)
        ).fill_null(False)
        logger.info(
            "pqtl_relative_name_margin %g: %d tubes name a KING relative (mismatch)",
            config.pqtl_relative_name_margin,
            scores.select(relative_named.sum()).item(),
        )
        mismatch = mismatch | relative_named
    blocked = (
        (pl.col("pqtl_best_z") >= config.pqtl_best_min)
        & (pl.col("pqtl_best_kinship") >= config.pqtl_relative_kinship)
    ).fill_null(False)
    match = genotyped & (pl.col("pqtl_self_z") >= config.pqtl_match_floor) & ~mismatch & ~blocked
    if config.pqtl_relative_margin is not None:
        if "pqtl_best_rel_z" not in scores.columns:
            raise ValueError(
                "pqtl_relative_margin is set, so the scores need pqtl_best_rel_z "
                "(extract_concordance(..., best_relative=True))"
            )
        trails = pl.col("pqtl_best_rel_z").is_null() | (
            (pl.col("pqtl_self_z") - pl.col("pqtl_best_rel_z")) >= config.pqtl_relative_margin
        )
        n_blocked = scores.select((match & ~trails).fill_null(False).sum()).item()
        logger.info(
            "pqtl_relative_margin %g: %d tubes would match but a relative trails self z by less (no match)",
            config.pqtl_relative_margin,
            n_blocked,
        )
        match = match & trails
    match = match.fill_null(False)
    return scores.with_columns(
        pl.when(mismatch)
        .then(pl.lit("mismatch"))
        .when(match)
        .then(pl.lit("match"))
        .otherwise(pl.lit("none"))
        .alias("pqtl_call"),
        pl.when(relative_named)
        .then(pl.col("pqtl_best_rel_USUBJID"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("pqtl_relative_USUBJID")
        if config.pqtl_relative_name_margin is not None
        else pl.lit(None, dtype=pl.String).alias("pqtl_relative_USUBJID"),
    ).drop("_fit", "_all_same", "_agreed", "_rel_kinship", strict=False)


# The rule's rate for the mixture WARN: at most 1 in 1,000 distinct clean tubes would
# WARN.
MIXTURE_WARN_MAX_RATE: float = 1e-3


def mixture_warn_threshold(
    clean_z, max_rate: float = MIXTURE_WARN_MAX_RATE, step: float | None = None
) -> float:
    """Returns the rule's threshold for ``pqtl_mixture_warn``.

    The threshold is the smallest at which at most ``max_rate`` of the clean tubes'
    finite ``pqtl_mixture_z`` values would WARN (z at or above it). With ``step``, it is
    rounded up to a multiple of ``step``, which keeps the rate at or below
    ``max_rate``.

    Args:
        clean_z: ``pqtl_mixture_z`` of the clean tubes; non-finite values are ignored.
        max_rate: Largest fraction of clean tubes that may WARN.
        step: Rounding step, or None for no rounding.

    Raises:
        ValueError: If there is no finite value, ``max_rate`` is outside [0, 1) or
            ``step`` is not positive.
    """
    z = np.asarray(clean_z, dtype=float)
    z = np.sort(z[np.isfinite(z)])[::-1]
    if z.size == 0:
        raise ValueError("mixture_warn_threshold: no finite pqtl_mixture_z on clean tubes")
    if not 0 <= max_rate < 1:
        raise ValueError(f"mixture_warn_threshold: max_rate must be in [0, 1); got {max_rate!r}")
    k = int(np.floor(max_rate * z.size))  # Flags allowed.
    # Just above the (k+1)-th largest value, so at most k values reach it.
    t = float(np.nextafter(z[k], np.inf))
    if step is not None:
        if not step > 0:
            raise ValueError(f"mixture_warn_threshold: step must be > 0; got {step!r}")
        t = float(np.ceil(t / step) * step)
    return t


def link_evidence(
    S: np.ndarray,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str],
) -> pl.DataFrame:
    """Returns participant-level pQTL concordance with the positive-identification rule.

    Per-sample scores are averaged per participant; then, for each linked genotype, each
    participant's proteome fit is robust-z scored against the other participants. A
    mismatch requires that another participant's proteome fits the genotype better than
    the linked participant's and at least as well as the median correct link
    (``typical_linked_z``). A low self score alone is not a mismatch.

    Args:
        S: ``pqtl_reference.reference_pqtl_scores`` output, shape (n_samples,
            n_genotyped).
        sample_usubjids: USUBJID per sample (rows of ``S``).
        geno_usubjids: USUBJID per genotyped participant (columns of ``S``).

    Returns:
        One row per genotyped participant with proteomics: ``USUBJID``, ``linked_z``,
        ``best_other_USUBJID``, ``best_other_z``, ``link_mismatch`` and
        ``typical_linked_z``, sorted by ``USUBJID``.
    """
    participants = sorted(set(sample_usubjids))
    # Participants x genotypes.
    M = np.vstack([S[sample_usubjids == p].mean(axis=0) for p in participants])
    p_index = {p: i for i, p in enumerate(participants)}

    rows: list[dict] = []
    for j, g in enumerate(geno_usubjids):
        if g not in p_index:
            continue
        own = p_index[g]
        others = np.arange(len(participants)) != own
        z = robust_z(M[:, j], others)
        best_idx = int(np.flatnonzero(others)[np.argmax(z[others])])
        rows.append(
            {
                "USUBJID": g,
                "linked_z": float(z[own]),
                "best_other_USUBJID": participants[best_idx],
                "best_other_z": float(z[best_idx]),
            }
        )

    _schema = {
        "USUBJID": pl.String,
        "linked_z": pl.Float64,
        "best_other_USUBJID": pl.String,
        "best_other_z": pl.Float64,
        "link_mismatch": pl.Boolean,
        "typical_linked_z": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=_schema)

    evidence = pl.DataFrame(rows)
    typical = evidence["linked_z"].median()
    return evidence.with_columns(
        ((pl.col("best_other_z") > pl.col("linked_z")) & (pl.col("best_other_z") >= typical)).alias(
            "link_mismatch"
        ),
        pl.lit(typical).alias("typical_linked_z"),
    ).sort("USUBJID")


# ── Genetic-sex cross-check ──────────────────────────────────────────────────


def verify_dna_link(
    pqtl_scores: pl.DataFrame,
    sex_predictions: pl.DataFrame | None,
    genetic_sex: pl.DataFrame | None,
    sex_prob: float = SEX_PROB_THRESHOLD,
) -> pl.DataFrame:
    """Cross-checks each sample's genetic sex against its proteomics-predicted sex.

    Each sample is judged on its own prediction.

    Args:
        pqtl_scores: Per-sample scores with ``SampleID`` and ``USUBJID`` (the labeled
            subject), e.g. ``extract_concordance`` with ``sample_ids``.
        sex_predictions: Proteomics-predicted sex per sample: ``SampleID``, ``p_male``.
        genetic_sex: ``USUBJID`` and ``genetic_sex`` (1 male, 2 female, null no call:
            unknown, XXY, X0 or gray), ``infer_genetic_sex`` with the IIDs taken as
            USUBJIDs. A null genetic sex is no evidence, never a mismatch.
        sex_prob: A prediction is confident when ``p_male`` is above it or below ``1 -
            sex_prob`` (``TrialConfig.identity_sex_prob``).

    Returns:
        ``pqtl_scores`` with ``dna_sex_concordant``: True or False for a confident
        prediction against a known genetic sex, null otherwise (all null when either
        input is None).
    """
    if sex_predictions is None or genetic_sex is None:
        return pqtl_scores.with_columns(pl.lit(None, dtype=pl.Boolean).alias("dna_sex_concordant"))

    merged = pqtl_scores.join(
        sex_predictions.select("SampleID", pl.col("p_male").alias("_p_male")),
        on="SampleID",
        how="left",
        validate="m:1",
        maintain_order="left",
    ).join(
        genetic_sex.select("USUBJID", pl.col("genetic_sex").alias("_gen")),
        on="USUBJID",
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    gen = pl.col("_gen")
    male = pl.col("_p_male") > sex_prob
    female = pl.col("_p_male") < (1 - sex_prob)
    return merged.with_columns(
        pl.when(((gen == 1) & male) | ((gen == 2) & female))
        .then(True)
        .when(((gen == 1) & female) | ((gen == 2) & male))
        .then(False)  # Confident mismatch.
        .otherwise(None)  # Unknown genetic sex or uncertain prediction.
        .alias("dna_sex_concordant"),
    ).drop("_gen", "_p_male")


# ── Duplicate genomes and the DNA-link table ─────────────────────────────────


def duplicate_genome_links(
    S: np.ndarray,
    sample_usubjids: np.ndarray,
    fit: np.ndarray,
    geno_usubjids: list[str],
    kinship: np.ndarray,
    *,
    duplicate_kinship: float,
    match_floor: float,
) -> pl.DataFrame:
    """Resolves participants whose genotypes are one genome.

    Genotype columns with KING kinship of at least ``duplicate_kinship`` form a group
    (connected through such pairs). A group member matches when the median of its fit
    tubes' z against its own link (``S[tube, own column]``) is at least
    ``match_floor``; a member with no fit tube does not match.

    - One member matches: that member is ``kept``, the others ``dropped``.
    - Several match: the matching members are ``hold`` (manual review) and the rest
      ``dropped``.
    - None matches: every member is ``dropped``; the reason is
      ``duplicate_genome_partner_without_proteomics`` when exactly one member has tubes
      (the others have none), else
      ``duplicate_genome_no_proteome_match``.

    The caller removes the dropped genotypes from the mapped list and treats ``hold``
    tubes as no evidence.

    Args:
        S: Robust z of each sample against each genotype.
        sample_usubjids: USUBJID per sample (rows of ``S``).
        fit: Boolean per sample: the tube passed technical QC.
        geno_usubjids: USUBJID per genotype (columns of ``S``).
        kinship: KING kinship between the genotypes, in ``geno_usubjids`` order.
        duplicate_kinship: Smallest kinship of two genotypes of one genome.
        match_floor: Smallest median own-link z of a matching member.

    Returns:
        One row per member of a group (genotypes with no duplicate are absent):
        ``USUBJID``, ``duplicate_USUBJID`` (the member of the group with the highest
        kinship to it), ``duplicate_kinship`` (that kinship), ``median_self_z``,
        ``link_status`` and ``link_reason``.
    """
    from scipy.sparse.csgraph import connected_components

    n = len(geno_usubjids)
    K = np.nan_to_num(np.asarray(kinship, dtype=float), nan=-1.0)
    np.fill_diagonal(K, -1.0)
    n_groups, group = connected_components(K >= duplicate_kinship, directed=False)
    usubjids = np.asarray(sample_usubjids)
    fit = np.asarray(fit, dtype=bool)

    rows: list[dict] = []
    for g in range(n_groups):
        members = np.flatnonzero(group == g)
        if members.size < 2:
            continue
        median_z: dict[int, float | None] = {}
        for j in members:
            tubes = (usubjids == geno_usubjids[j]) & fit
            z = S[tubes, j]
            z = z[np.isfinite(z)]
            median_z[j] = float(np.median(z)) if z.size else None
        matching = [j for j in members if median_z[j] is not None and median_z[j] >= match_floor]
        n_with_proteomics = sum(bool((usubjids == geno_usubjids[j]).any()) for j in members)
        for j in members:
            partner = int(members[np.argmax(K[j, members])])
            if len(matching) == 0 and n_with_proteomics == 1:
                # One proteomic member, below the floor, and partners with no proteome
                # to contest it: nothing can say whose genome it is.
                status, reason = (
                    "dropped",
                    "duplicate_genome_partner_without_proteomics",
                )
            elif len(matching) == 0:
                status, reason = "dropped", "duplicate_genome_no_proteome_match"
            elif j in matching and len(matching) == 1:
                status, reason = "kept", "duplicate_genome_other_dropped"
            elif j in matching:
                status, reason = "hold", "duplicate_genome_both_match"
            else:
                status, reason = "dropped", "duplicate_genome_proteome_mismatch"
            rows.append(
                {
                    "USUBJID": geno_usubjids[j],
                    "duplicate_USUBJID": geno_usubjids[partner],
                    "duplicate_kinship": float(K[j, partner]),
                    "median_self_z": median_z[j],
                    "link_status": status,
                    "link_reason": reason,
                }
            )
    assert len({r["USUBJID"] for r in rows}) == len(rows) <= n
    return pl.DataFrame(
        rows,
        schema={
            "USUBJID": pl.String,
            "duplicate_USUBJID": pl.String,
            "duplicate_kinship": pl.Float64,
            "median_self_z": pl.Float64,
            "link_status": pl.String,
            "link_reason": pl.String,
        },
    )


# Karyotype notes of the DNA-link table (annotation only).
NOTE_OPPOSITE_OWN_DNA = "protein sex differs from own-DNA genetic sex"
NOTE_OPPOSITE_NO_OWN_DNA = (
    "protein sex differs from genetic sex on tubes without a pQTL match to this genome"
)


def _karyotype_note(tubes: pl.DataFrame) -> pl.DataFrame:
    """Returns each subject's ``proteome_sex_vs_genome`` (see ``dna_link_table``).

    It is taken over the subject's fit tubes with a confident prediction against a
    known genetic sex (``dna_sex_concordant`` not null).
    """
    known = tubes.filter(pl.col("_fit") & pl.col("dna_sex_concordant").is_not_null())
    own = pl.col("pqtl_call") == "match"
    conc = pl.col("dna_sex_concordant")
    return known.group_by("USUBJID").agg(
        pl.when(own.any())
        .then(
            pl.when((~conc).filter(own).all())
            .then(pl.lit("opposite_own_dna"))
            .when(conc.filter(own).all())
            .then(pl.lit("concordant"))
        )
        .when((~conc).any())
        .then(pl.lit("opposite_no_own_dna"))
        .when(conc.any())
        .then(pl.lit("concordant"))
        .alias("proteome_sex_vs_genome")
    )


def dna_link_table(
    scores: pl.DataFrame,
    fit: np.ndarray,
    links: pl.DataFrame,
    duplicates: pl.DataFrame,
    genetic_sex: pl.DataFrame,
    *,
    trial: str,
    platform: str,
) -> pl.DataFrame:
    """Builds the DNA-link table: one row per genotype of a subject with proteomics.

    The table also has a row for each genotype of a participant with no proteomics that
    the evidence names. The genotype IID is the USUBJID, so ``IID`` repeats it.
    Columns are ``DNA_LINK_SCHEMA``:

    - the link-level evidence of ``link_evidence``;
    - tube counts by call over the subject's fit tubes, and ``n_dna_sex_discordant``
      over all its tubes;
    - ``duplicate_USUBJID`` and ``duplicate_kinship``;
    - ``relink_USUBJID``: an ungenotyped subject whose every fit tube names this
      genotype at ``pqtl_best_min`` or more (the lexicographically first when several
      do);
    - ``link_status``: ``dropped`` or ``hold`` from the duplicate resolution, else
      ``suspect`` when every fit tube is a mismatch, else ``ok``, with ``link_reason``;
    - the karyotype note (annotation only: it changes no flag and no ``link_status``),
      over the fit tubes with a confident prediction against a known genetic sex.
      ``proteome_sex_vs_genome``: when some are own-DNA tubes (``pqtl_call``
      ``match``), ``opposite_own_dna`` if every own-DNA one is opposite the genetic
      sex, ``concordant`` if every one agrees, else null; with none,
      ``opposite_no_own_dna`` if any is opposite, else ``concordant`` if any agrees;
      null when there is no such tube (a null genetic sex included). ``sex_note`` is
      ``NOTE_OPPOSITE_OWN_DNA`` or ``NOTE_OPPOSITE_NO_OWN_DNA`` for the two opposite
      values, else null; ``identity.fill_pqtl_only`` nulls the own-DNA text of a
      subject with an identity event.
      ``opposite_own_dna`` is never ``suspect``.

    A genotype of a participant with no proteomics (absent from ``links``, which
    ``link_evidence`` builds over proteomic subjects) gets a row only when the evidence
    names it: it is a member of a duplicate-genome group, or the target of a
    ``relink_USUBJID``. It has no tube, so its tube counts are 0 and its link-level
    evidence (``linked_z``, ``best_other_*``, ``typical_linked_z``, ``link_mismatch``)
    is null; its status is ``ok`` unless the duplicate resolution dropped it, and
    ``genetic_sex`` is its own.

    ``n_pqtl_only`` is null here; the identity step fills it.

    Args:
        scores: The classified per-tube frame (``classify_pqtl``) with the ``hold`` and
            ``dropped`` tubes already set to ``none`` and their best-other columns null.
        fit: Boolean per row of ``scores``: the tube passed technical QC.
        links: ``link_evidence`` output.
        duplicates: ``duplicate_genome_links`` output.
        genetic_sex: ``USUBJID`` and ``SEX_COLUMNS`` per genotype.
        trial: Trial name for the ``trial`` column.
        platform: Platform name for the ``platform`` column.
    """
    fit_flag = pl.Series("_fit", np.asarray(fit, dtype=bool))
    tubes = scores.with_columns(fit_flag)
    counts = tubes.group_by("USUBJID").agg(
        pl.col("_fit").sum().cast(pl.Int64).alias("n_fit_tubes"),
        ((pl.col("pqtl_call") == "match") & pl.col("_fit")).sum().cast(pl.Int64).alias("n_match"),
        ((pl.col("pqtl_call") == "mismatch") & pl.col("_fit"))
        .sum()
        .cast(pl.Int64)
        .alias("n_mismatch"),
        ((pl.col("pqtl_call") == "none") & pl.col("_fit")).sum().cast(pl.Int64).alias("n_none"),
        (~pl.col("dna_sex_concordant")).sum().cast(pl.Int64).alias("n_dna_sex_discordant"),
    )
    # Ungenotyped subjects whose every fit tube names one genotype: relink candidates.
    candidates = (
        tubes.filter(pl.col("pqtl_self_z").is_null() & pl.col("_fit"))
        .group_by("USUBJID")
        .agg(
            pl.col("pqtl_best_nonrel_USUBJID").null_count().alias("_nulls"),
            pl.col("pqtl_best_nonrel_USUBJID").n_unique().alias("_n_named"),
            pl.col("pqtl_best_nonrel_USUBJID").first().alias("_target"),
        )
        .filter((pl.col("_nulls") == 0) & (pl.col("_n_named") == 1))
        .group_by("_target")
        .agg(pl.col("USUBJID").sort().first().alias("relink_USUBJID"))
        .rename({"_target": "USUBJID"})
    )
    # Genotypes with no proteomics that the evidence names get a row of their own.
    named = (set(duplicates["USUBJID"]) | set(candidates["USUBJID"])) - set(links["USUBJID"])
    links = pl.concat(
        [
            links,
            pl.DataFrame({"USUBJID": sorted(named)}, schema={"USUBJID": pl.String}),
        ],
        how="diagonal",
    )
    out = (
        links.join(counts, on="USUBJID", how="left", maintain_order="left")
        .join(
            duplicates.select(
                "USUBJID",
                "duplicate_USUBJID",
                "duplicate_kinship",
                pl.col("link_status").alias("_dup_status"),
                pl.col("link_reason").alias("_dup_reason"),
            ),
            on="USUBJID",
            how="left",
            maintain_order="left",
        )
        .join(candidates, on="USUBJID", how="left", maintain_order="left")
        .join(
            genetic_sex.select("USUBJID", *SEX_COLUMNS),
            on="USUBJID",
            how="left",
            maintain_order="left",
        )
        .join(_karyotype_note(tubes), on="USUBJID", how="left", maintain_order="left")
        .with_columns(  # No genetic sex call (XXY, X0, gray, absent): no note.
            pl.when(pl.col("genetic_sex").is_not_null())
            .then(pl.col("proteome_sex_vs_genome"))
            .alias("proteome_sex_vs_genome")
        )
        .with_columns(
            pl.col("proteome_sex_vs_genome")
            .replace_strict(
                {
                    "opposite_own_dna": NOTE_OPPOSITE_OWN_DNA,
                    "opposite_no_own_dna": NOTE_OPPOSITE_NO_OWN_DNA,
                },
                default=None,
                return_dtype=pl.String,
            )
            .alias("sex_note")
        )
        .with_columns(
            pl.col("USUBJID").alias("IID"),
            pl.col(
                "n_fit_tubes", "n_match", "n_mismatch", "n_none", "n_dna_sex_discordant"
            ).fill_null(0),
            pl.lit(None, dtype=pl.Int64).alias("n_pqtl_only"),
            pl.lit(trial).alias("trial"),
            pl.lit(platform).alias("platform"),
        )
        .with_columns(
            pl.when(pl.col("_dup_status").is_in(["dropped", "hold"]))
            .then(pl.col("_dup_status"))
            .when((pl.col("n_fit_tubes") > 0) & (pl.col("n_mismatch") == pl.col("n_fit_tubes")))
            .then(pl.lit("suspect"))
            .otherwise(pl.lit("ok"))
            .alias("link_status"),
            pl.when(pl.col("_dup_status").is_in(["dropped", "hold"]))
            .then(pl.col("_dup_reason"))
            .when((pl.col("n_fit_tubes") > 0) & (pl.col("n_mismatch") == pl.col("n_fit_tubes")))
            .then(pl.lit("all_fit_tubes_mismatch"))
            .alias("link_reason"),
        )
        .sort("USUBJID")
    )
    return out.select([pl.col(c).cast(t) for c, t in DNA_LINK_SCHEMA.items()])
