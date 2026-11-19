"""Parses reference cis-pQTL catalogs and scores samples against genotypes with them.

Published cis-pQTL catalogs (UKB-PPP, deCODE and the others in ``CATALOG_NAMES``) are
parsed into ``CATALOG_SCHEMA`` frames, loaded with checks on the trial's platform and
genome build, aligned to the trial's PVAR alleles, and used to score every sample
against every genotype. Variants and effect sizes come from the catalog, so scoring
needs no cross-fitting. Catalog rows join to genotypes on ``(chrom, pos, a1, a2)``
through ``CatalogGenotypes.catalog_variant``, never on rsID, so rows without an rsID
still score.

Example::

    catalog = parse_ukbppp("41586_2023_6592_MOESM3_ESM.xlsx")
    genotypes = pqtl.extract_catalog_genotypes(catalog, pgen_dir)
    S = reference_pqtl_scores(
        Z, protein_ids, genotypes, catalog, sample_usubjids, geno_usubjids
    )

A QC run goes through ``pqtl_concordance_for_qc``: ``pqtl_inputs``, ``pqtl_catalog``,
``pqtl_matrix`` and ``pqtl_calls``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from analysis.qc.dna_relink import relabel, relink_map
from analysis.qc.genome_tables import (
    check_projected_pcs,
    kinship_from_table,
    pcs_for,
    read_ancestry_pcs,
    read_king_table,
)
from analysis.qc.identity import (
    LpParams,
    ResidualCache,
    _build_fit_mask,
    matrix_for,
    residualize_samples,
    robust_z,
)
from analysis.qc.platform import get_platform
from analysis.qc.pqtl import (
    KING_DUPLICATE,
    PQTL_BEST_MIN,
    SEX_COLUMNS,
    classify_pqtl,
    dna_link_table,
    duplicate_genome_links,
    extract_catalog_genotypes,
    extract_concordance,
    kinship_matrix,
    link_evidence,
    normalize_chrom,
    pgen_prefixes,
    read_psam,
    verify_dna_link,
)
from analysis.qc.schemas import PQTL_SCORES_SCHEMA
from analysis.qc_paths import PQTL_CATALOG_DIR

if TYPE_CHECKING:
    from analysis.qc.pqtl import CatalogGenotypes

logger = logging.getLogger(__name__)

WEIGHT_EXPONENT: float = 0.25
"""Per-side exponent on -log10(p). Both the observed and the predicted protein vectors
are multiplied by ``nlp ** WEIGHT_EXPONENT``, so each protein's term in the correlation
carries an effective weight of sqrt(nlp)."""
# Defaults of TrialConfig ``pqtl_maf_min`` and ``pqtl_contribution_cap``, both off: the
# smallest trial minor-allele frequency a protein's lead variant may have, and the bound
# on each protein's term in a tube-genotype correlation (0 is no bound).
MAF_MIN: float = 0.0
CONTRIBUTION_CAP: float = 0.0
# Default of TrialConfig ``pqtl_nlp_recompute_floor``: a catalog -log10(p) at or above
# it is recomputed from beta/se when the row has a usable se, and a row with no p and no
# se gets this value (fill_underflowed_nlp).
NLP_RECOMPUTE_FLOOR: float = 300.0

# The parsed catalogs, under their file names; paths.yaml ``pqtl_catalog_dir`` points at
# the directory.
_PARSED_DIR = PQTL_CATALOG_DIR

CATALOG_NAMES: dict[str, str] = {
    # Olink Explore HT (five catalogs).
    "ukbppp": "ukbppp_sun2023.parquet",
    "jhs": "jhs_tahir2024.parquet",
    "gh": "genes_health_olink_2026.parquet",
    "mesa": "mesa_olink_2026.parquet",
    "ckb": "ckb_olink_2025.parquet",
    # SomaScan (eight catalogs; `ckb_soma` is the 2026 CKB Atlas).
    "decode": "decode_ferkingstad2021.parquet",
    "aric": "aric_zhang2022.parquet",
    "fenland_hg38": "fenland_pietzner2021_hg38.parquet",
    "ckb_soma": "ckb_soma_2026.parquet",
    "mesa_soma": "mesa_soma_2026.parquet",
    "gh_soma": "genes_health_soma_2026.parquet",
    "ages": "ages_gudjonsson2022.parquet",
    "interval": "interval_sun2018_hg38.parquet",
}
"""Catalog names a config may use, and the parsed parquet each resolves to (in
``analysis.qc_paths.PQTL_CATALOG_DIR``)."""

MIXED_PLATFORM = "mixed"
CATALOG_REGISTRY: dict[str, tuple[str, str]] = {
    "ukbppp_sun2023.parquet": ("hg38", "olink"),
    "jhs_tahir2024.parquet": ("hg38", "olink"),
    "genes_health_olink_2026.parquet": ("hg38", "olink"),  # Olink Explore HT
    "mesa_olink_2026.parquet": ("hg38", "olink"),
    "ckb_olink_2025.parquet": ("hg38", "olink"),
    "decode_ferkingstad2021.parquet": ("hg38", "soma"),
    "aric_zhang2022.parquet": ("hg38", "soma"),
    "fenland_pietzner2021_hg38.parquet": ("hg38", "soma"),
    "ckb_soma_2026.parquet": ("hg38", "soma"),
    "ages_gudjonsson2022.parquet": ("hg38", "soma"),  # hg19 positions lifted to hg38
    "interval_sun2018_hg38.parquet": ("hg38", "soma"),
    "mesa_soma_2026.parquet": ("hg38", "soma"),
    "genes_health_soma_2026.parquet": ("hg38", "soma"),  # SomaScan 11K
}
"""Genome build and platform ("olink", "soma" or ``MIXED_PLATFORM``) of each parsed
catalog, by file name. ``load_reference_catalog`` takes both from here. A file that is not listed loads only when the config states its build
and platform."""

REQUIRED_BUILD = "hg38"
EUR_ANCESTRIES = ("EUR", "EA")
"""``ancestry`` values that count as European (``EA``: a catalog's European Americans)."""
POOLED_ANCESTRIES = ("MULTI",)
"""Pooled-ancestry catalogs, kept when ``include_non_eur`` is False."""
KEPT_ANCESTRIES = EUR_ANCESTRIES + POOLED_ANCESTRIES
"""``ancestry`` values kept when ``include_non_eur`` is False."""
CATALOG_MERGES = ("strongest", "list_order")
"""Values of ``TrialConfig.pqtl_catalog_merge``. ``strongest`` keeps the strongest row
per protein across every listed catalog (``select_catalog_rows``); ``list_order`` lets
the first listed catalog with a usable row for a protein supply all its rows
(``merge_catalog_list_order``)."""
CATALOG_RANK = "catalog_rank"
"""Column ``load_reference_catalog`` adds with ``with_rank``: each row's catalog
position in the ``source`` list, 0 first."""

SIGN_STRONG_NLP: float = 20.0
"""-log10(p) at or above which a catalog row votes in ``sign_discordant_rows``."""
SIGN_MIN_OPPOSING: int = 2
"""Fewest other catalogs whose strong row must oppose a row's sign before it is
flagged."""
CATALOG_COHORTS: dict[str, str] = {}
"""Catalogs (``CATALOG_NAMES``) that report the same cohort's rows, mapped to a shared
cohort label: they never vote on each other in ``sign_discordant_rows``."""

SIGN_EXCLUSIONS: dict[str, frozenset[str]] = {
    "ukbppp_sun2023.parquet": frozenset({"P31994"}),
    "ckb_olink_2025.parquet": frozenset({"P31994"}),
}
"""Proteins whose rows ``load_reference_catalog`` drops from a parsed catalog, by
file name: UniProt accessions for Olink catalogs (matched against every accession of a
row's ``uniprot``), SeqIds for SomaScan ones.

``derive_sign_exclusions`` returns the same list for the configured catalogs. When one row
of a protein is flagged, all of that protein's rows are dropped from that catalog."""

CATALOG_SCHEMA: dict[str, pl.DataType] = {
    "rsid": pl.String,
    "chrom": pl.String,
    "pos": pl.Int64,
    "protein_id": pl.String,
    "uniprot": pl.String,
    "platform_id": pl.String,
    "beta": pl.Float64,
    "se": pl.Float64,
    "neg_log10p": pl.Float64,
    "maf": pl.Float64,
    "a1": pl.String,  # Effect allele; beta is reported for it.
    "a2": pl.String,  # Other allele.
    "ancestry": pl.String,
    "source": pl.String,
    "eaf": pl.Float64,  # Frequency of a1 in the discovery sample, or null.
}
"""Columns of a parsed catalog. ``eaf`` keeps the direction ``maf`` folds away, so a
palindromic SNP can be oriented by frequency (``align_catalog_alleles``); a stored
catalog without it loads with a null ``eaf`` (``OPTIONAL_CATALOG_COLUMNS``)."""
OPTIONAL_CATALOG_COLUMNS: tuple[str, ...] = ("eaf",)
"""``CATALOG_SCHEMA`` columns a stored catalog may lack; ``load_reference_catalog`` adds
them as nulls."""

# ── Internal helpers ────────────────────────────────────────────────────────


def _chrom_str(raw) -> str | None:
    """Normalizes a chromosome value to a bare string ('1', '22', 'X')."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return str(int(raw))
    return str(raw).strip().replace("chr", "")


def _safe_str(val) -> str | None:
    """Converts a cell value to a stripped string, treating '-' as None."""
    if val is None:
        return None
    s = str(val).strip()
    return None if s == "-" else s


# ── Catalog parsers ─────────────────────────────────────────────────────────


def _parse_ukbppp_st9(ws) -> dict[str, list]:
    """Extracts cis-pQTL rows from UKB-PPP ST9 (European discovery).

    ST9 layout, header at row 5, data from row 6:
      [0] Variant ID, [1] CHROM, [2] GENPOS(hg38), [3] Region ID,
      [4] Region Start, [5] Region End, [6] MHC,
      [7] UKBPPP ProteinID, [8] Assay Target, [9] Target UniProt,
      [10] rsID, [11] A1FREQ, [12] BETA, [13] SE, [14] log10(p),
      ... [19] cis/trans, [20] cis gene
    """
    cols: dict[str, list] = {k: [] for k in CATALOG_SCHEMA}
    for row in ws.iter_rows(min_row=6, values_only=True):
        if row[19] != "cis":
            continue
        pid_raw = row[7]
        if pid_raw is None:
            continue
        parts = str(pid_raw).split(":")
        if len(parts) < 3:
            continue

        # Variant ID format: CHROM:GENPOS:A0:A1:imp:v1; BETA is for A1.
        vid = str(row[0]) if row[0] is not None else ""
        vid_parts = vid.split(":")
        a0 = vid_parts[2] if len(vid_parts) > 3 else None
        a1 = vid_parts[3] if len(vid_parts) > 3 else None

        cols["rsid"].append(_safe_str(row[10]))
        cols["chrom"].append(_chrom_str(row[1]))
        cols["pos"].append(int(row[2]) if row[2] is not None else None)
        cols["protein_id"].append(parts[0])
        cols["uniprot"].append(_safe_str(row[9]))
        cols["platform_id"].append(parts[2])
        cols["beta"].append(float(row[12]) if row[12] is not None else None)
        cols["se"].append(float(row[13]) if row[13] is not None else None)
        cols["neg_log10p"].append(float(row[14]) if row[14] is not None else None)
        a1freq = float(row[11]) if row[11] is not None else None
        cols["maf"].append(min(a1freq, 1.0 - a1freq) if a1freq is not None else None)
        cols["eaf"].append(a1freq)
        cols["a1"].append(a1)  # Tested allele.
        cols["a2"].append(a0)
        cols["ancestry"].append("EUR")
        cols["source"].append("ukbppp")
    return cols


def _parse_ukbppp_st11(ws) -> dict[str, list]:
    """Extracts cis-pQTL rows from UKB-PPP ST11 (non-European ancestries).

    ST11 layout, header at row 6, data from row 7:
      [0] Variant ID, [1] CHROM, [2] GENPOS(hg38),
      [3] Region Start, [4] Region End, [5] MHC,
      [6] UKBPPP ProteinID, [7] Assay Target, [8] Target UniProt,
      [9] rsID, [10] Ancestry, [11] A1FREQ, [12] BETA, [13] SE,
      [14] log10(p), [15] Sig in disc., [16] cis/trans, [17] cis gene
    """
    cols: dict[str, list] = {k: [] for k in CATALOG_SCHEMA}
    for row in ws.iter_rows(min_row=7, values_only=True):
        if row[16] != "cis":
            continue
        pid_raw = row[6]
        if pid_raw is None:
            continue
        parts = str(pid_raw).split(":")
        if len(parts) < 3:
            continue

        # Variant ID format: CHROM:GENPOS:A0:A1:imp:v1; BETA is for A1.
        vid = str(row[0]) if row[0] is not None else ""
        vid_parts = vid.split(":")
        a0 = vid_parts[2] if len(vid_parts) > 3 else None
        a1 = vid_parts[3] if len(vid_parts) > 3 else None

        cols["rsid"].append(_safe_str(row[9]))
        cols["chrom"].append(_chrom_str(row[1]))
        cols["pos"].append(int(row[2]) if row[2] is not None else None)
        cols["protein_id"].append(parts[0])
        cols["uniprot"].append(_safe_str(row[8]))
        cols["platform_id"].append(parts[2])
        cols["beta"].append(float(row[12]) if row[12] is not None else None)
        cols["se"].append(float(row[13]) if row[13] is not None else None)
        cols["neg_log10p"].append(float(row[14]) if row[14] is not None else None)
        a1freq = float(row[11]) if row[11] is not None else None
        cols["maf"].append(min(a1freq, 1.0 - a1freq) if a1freq is not None else None)
        cols["eaf"].append(a1freq)
        cols["a1"].append(a1)
        cols["a2"].append(a0)
        ancestry = _safe_str(row[10])
        cols["ancestry"].append(ancestry if ancestry else "NON_EUR")
        cols["source"].append("ukbppp")
    return cols


def parse_ukbppp(
    path: str | Path,
    *,
    include_non_eur: bool = True,
) -> pl.DataFrame:
    """Parses the UKB-PPP cis-pQTL catalog (Sun et al. 2023, Olink Explore 3072).

    Sheet ST9 holds the European discovery (p < 1.7e-11), tagged ``ancestry = "EUR"``;
    sheet ST11 holds the non-European ancestries (CSA, AFR, EAS, MID, AMR), tagged with
    their own code. ``platform_id`` is the OlinkID from the ``UKBPPP ProteinID`` field
    (``gene:UniProt:OlinkID:v1``). Positions are hg38.

    Args:
        path: Path to ``ukbppp_sun2023_supplementary.xlsx``.
        include_non_eur: Whether to append the ST11 rows to the ST9 rows.

    Returns:
        The cis-pQTL catalog with ``CATALOG_SCHEMA`` columns.
    """
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)

    st9_cols = _parse_ukbppp_st9(wb["ST9"])
    n_st9 = len(st9_cols["rsid"])
    logger.info("UKB-PPP ST9: parsed %d cis-pQTLs (EUR)", n_st9)

    if include_non_eur:
        st11_cols = _parse_ukbppp_st11(wb["ST11"])
        n_st11 = len(st11_cols["rsid"])
        logger.info("UKB-PPP ST11: parsed %d cis-pQTLs (non-EUR)", n_st11)
        for k in CATALOG_SCHEMA:
            st9_cols[k].extend(st11_cols[k])

    wb.close()

    if not st9_cols["rsid"]:
        return pl.DataFrame(schema=CATALOG_SCHEMA)

    return pl.DataFrame(st9_cols, schema=CATALOG_SCHEMA)


def parse_decode(path: str | Path) -> pl.DataFrame:
    """Parses the deCODE cis-pQTL catalog (Ferkingstad et al. 2021) from sheet ST02.

    The header is at row 3 (with embedded newlines); data start at row 4. Only cis
    rows are kept. ``platform_id`` is the SomaScan SeqId. Betas and p-values are the
    adjusted ones (conditional on other pQTLs at the locus); ``se`` is null,
    positions are hg38 and ``ancestry`` is ``"EUR"``.

    ``a1`` is Amin, the effect allele, and ``a2`` is Amaj. Rows where either allele is
    missing, ``!``-marked (deCODE's mark for an unresolved indel or multiallelic site)
    or longer than one base are dropped: they cannot be aligned reliably against the
    trial's PVAR REF/ALT.

    Args:
        path: Path to ``decode_ferkingstad2021_supplementary.xlsx``.

    Returns:
        The cis-pQTL catalog with ``CATALOG_SCHEMA`` columns.
    """
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb["ST02"]

    cols: dict[str, list] = {k: [] for k in CATALOG_SCHEMA}
    n_dropped_allele = 0

    for row in ws.iter_rows(min_row=4, values_only=True):
        cis_trans = str(row[21]).strip() if row[21] is not None else ""
        if cis_trans != "cis":
            continue

        # Amin (column 17) is the effect allele, Amaj (column 18) the other.
        amin = _safe_str(row[17])
        amaj = _safe_str(row[18])

        if (
            amin is None
            or amaj is None
            or "!" in amin
            or "!" in amaj
            or len(amin) != 1
            or len(amaj) != 1
        ):
            n_dropped_allele += 1
            continue

        cols["rsid"].append(_safe_str(row[10]))
        cols["chrom"].append(_chrom_str(row[13]))
        cols["pos"].append(int(row[14]) if row[14] is not None else None)
        cols["protein_id"].append(str(row[2]).strip() if row[2] is not None else None)
        cols["uniprot"].append(_safe_str(row[8]))
        cols["platform_id"].append(str(row[9]).strip() if row[9] is not None else None)
        cols["beta"].append(float(row[25]) if row[25] is not None else None)
        cols["se"].append(None)
        cols["neg_log10p"].append(float(row[27]) if row[27] is not None else None)
        maf_pct = float(row[20]) if row[20] is not None else None
        cols["maf"].append(maf_pct / 100.0 if maf_pct is not None else None)
        # Amin, the effect allele, is the minor allele, so its frequency is the MAF.
        cols["eaf"].append(maf_pct / 100.0 if maf_pct is not None else None)
        cols["a1"].append(amin.upper())
        cols["a2"].append(amaj.upper())
        cols["ancestry"].append("EUR")
        cols["source"].append("decode")

    wb.close()
    n = len(cols["rsid"])
    logger.info(
        "deCODE: parsed %d cis-pQTLs from %s (%d dropped: unresolvable alleles)",
        n,
        path,
        n_dropped_allele,
    )

    if n == 0:
        return pl.DataFrame(schema=CATALOG_SCHEMA)

    return pl.DataFrame(cols, schema=CATALOG_SCHEMA)


def _catalog_path(source: str | Path) -> Path:
    """Returns the parquet path of a catalog name (``CATALOG_NAMES``) or path.

    Raises:
        ValueError: If ``source`` is neither a catalog name nor a ``.parquet`` path.
        FileNotFoundError: If the path does not exist.
    """
    name = str(source).strip().lower()
    if name in CATALOG_NAMES:
        path = _PARSED_DIR / CATALOG_NAMES[name]
    elif Path(source).suffix == ".parquet":
        path = Path(source)
    else:
        raise ValueError(
            f"Unknown catalog source: {source!r}. Pass one of "
            f"{', '.join(repr(n) for n in CATALOG_NAMES)} or a path to a .parquet file."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Catalog {source!r} resolves to {path}, which does not exist")
    return path


def _check_catalog_provenance(
    path: Path, platform: str, build: str | None, catalog_platform: str | None
) -> None:
    """Checks that ``path`` is an hg38 catalog of the trial ``platform``.

    ``CATALOG_REGISTRY`` gives the build and platform of a parsed catalog; a file it
    does not list needs ``build`` and ``catalog_platform`` from the config. Values the
    config states for a registered file must agree with the registry.

    Args:
        path: The catalog parquet.
        platform: The trial's platform, ``"olink"`` or ``"soma"``.
        build: Genome build the config states, or None.
        catalog_platform: Catalog platform the config states, or None.

    Raises:
        ValueError: If the build or platform is unknown, disagrees with the registry,
            is not hg38, is mixed or is not the trial's.
    """
    stated_platform = get_platform(catalog_platform).name if catalog_platform else None
    if path.name in CATALOG_REGISTRY:
        cat_build, cat_platform = CATALOG_REGISTRY[path.name]
        for what, stated, known in (
            ("build", build, cat_build),
            ("platform", stated_platform, cat_platform),
        ):
            if stated is not None and stated != known:
                raise ValueError(
                    f"Config states {what} {stated!r} for {path.name}, "
                    f"but the catalog registry has {known!r}"
                )
    else:
        if build is None or stated_platform is None:
            raise ValueError(
                f"{path.name} is not in the catalog registry: state its genome build "
                "and platform in the config (pqtl_catalog_build, pqtl_catalog_platform)"
            )
        cat_build, cat_platform = build, stated_platform
    if cat_build != REQUIRED_BUILD:
        raise ValueError(
            f"Catalog {path.name} is {cat_build}; the genotypes are {REQUIRED_BUILD}, "
            "so its positions would not match.  Lift it over first."
        )
    if cat_platform == MIXED_PLATFORM:
        raise ValueError(
            f"Catalog {path.name} mixes Olink and SomaScan rows and cannot be used on one platform"
        )
    if cat_platform != platform:
        raise ValueError(
            f"Catalog {path.name} is a {cat_platform} catalog; the trial platform is {platform}"
        )


def _load_one_catalog(
    path: Path,
    *,
    platform: str,
    include_non_eur: bool,
    build: str | None,
    catalog_platform: str | None,
) -> pl.DataFrame:
    _check_catalog_provenance(path, platform, build, catalog_platform)
    df = pl.read_parquet(path)
    if not include_non_eur:
        if "ancestry" not in df.columns:
            raise ValueError(
                f"include_non_eur is False but {path.name} has no ancestry column to filter on"
            )
        df = df.filter(pl.col("ancestry").is_in(list(KEPT_ANCESTRIES)))
        if df.is_empty():
            raise ValueError(
                f"include_non_eur is False and {path.name} has no rows with ancestry "
                f"{' or '.join(KEPT_ANCESTRIES)}"
            )
    absent = [c for c in OPTIONAL_CATALOG_COLUMNS if c not in df.columns]
    if absent:
        logger.info("catalog %s has no %s column; loaded as null", path.name, ", ".join(absent))
        df = df.with_columns(pl.lit(None, dtype=CATALOG_SCHEMA[c]).alias(c) for c in absent)
    missing = [c for c in CATALOG_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog {path.name} lacks columns {', '.join(missing)}")
    wrong = [
        f"{c} is {df.schema[c]}, expected {t}"
        for c, t in CATALOG_SCHEMA.items()
        if df.schema[c] != t
    ]
    if wrong:
        raise ValueError(f"Catalog {path.name} does not follow CATALOG_SCHEMA: {'; '.join(wrong)}")
    n_null_pos = df["pos"].null_count()
    if n_null_pos:
        logger.warning(
            "catalog %s: dropped %d of %d rows with a null pos (failed liftover)",
            path.name,
            n_null_pos,
            df.height,
        )
        df = df.filter(pl.col("pos").is_not_null())
    return _drop_sign_exclusions(df, path.name)


def _uniprot_tokens(col: str) -> pl.Expr:
    """Returns every UniProt accession of a ``uniprot`` string column.

    ``,``, ``;`` and ``_`` all separate accessions.
    """
    return (
        pl.col(col)
        .str.replace_all(r"[;_]", ",")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
    )


def _protein_key(platform: str) -> pl.Expr:
    """Returns a row's protein keys.

    These are the UniProt accessions on Olink and the SeqId (``-`` as ``_``) on
    SomaScan.
    """
    if platform == "olink":
        return _uniprot_tokens("uniprot")
    return pl.concat_list(pl.col("platform_id").str.replace_all("-", "_"))


def _drop_sign_exclusions(df: pl.DataFrame, file_name: str) -> pl.DataFrame:
    """Drops the rows of ``file_name``'s ``SIGN_EXCLUSIONS`` proteins from ``df``."""
    excluded = SIGN_EXCLUSIONS.get(file_name)
    if not excluded:
        return df
    platform = CATALOG_REGISTRY[file_name][1]
    hit = (
        _protein_key(platform)
        .list.eval(pl.element().is_in(sorted(excluded)))
        .list.any()
        .fill_null(False)
    )
    out = df.filter(~hit)
    logger.info(
        "catalog %s sign exclusions: dropped %d rows of %d (proteins %s)",
        file_name,
        df.height - out.height,
        df.height,
        ", ".join(sorted(excluded)),
    )
    return out


def sign_discordant_rows(
    catalogs: dict[str, pl.DataFrame],
    *,
    platform: str,
    strong: float = SIGN_STRONG_NLP,
    min_opposing: int = SIGN_MIN_OPPOSING,
    cohorts: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Returns the rows whose sign most other catalogs' strong rows contradict.

    A row takes part when it has both alleles, is not an A/T or C/G SNP, has a beta and,
    after ``fill_underflowed_nlp``, ``neg_log10p >= strong``. Rows meet at a site: a
    protein key (``_protein_key``: any shared UniProt accession on Olink, the SeqId on
    SomaScan), the chromosome, the position and the unordered allele pair; betas are
    compared for one allele. At its site, each row is compared with the strongest row of
    every other catalog not of its cohort, and votes count cohorts. A row is returned
    when at least ``min_opposing`` cohorts disagree in sign and they outnumber the
    cohorts on its side, its own included, so an even split flags neither side.

    Args:
        catalogs: Catalog name to ``CATALOG_SCHEMA`` frame, all of one platform.
        platform: The catalogs' platform, ``"olink"`` or ``"soma"``.
        strong: Smallest ``neg_log10p`` of a row that takes part.
        min_opposing: Fewest opposing cohorts that flag a row.
        cohorts: Catalog name to cohort label (default ``CATALOG_COHORTS``); a name not
            in it is its own cohort.

    Returns:
        The flagged rows with ``catalog``, ``protein``, ``chrom``, ``pos``, ``alleles``,
        ``ancestry``, ``neg_log10p``, ``beta``, ``n_same``, ``n_opposite``, ``same`` and
        ``opposite`` (comma-joined catalog names).
    """
    cohorts = CATALOG_COHORTS if cohorts is None else cohorts
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    frames = []
    for name, df in catalogs.items():
        frames.append(
            fill_underflowed_nlp(
                df.select([c for c in CATALOG_SCHEMA if c not in OPTIONAL_CATALOG_COLUMNS])
            ).with_columns(
                pl.lit(name).alias("catalog"),
                pl.lit(cohorts.get(name, name)).alias("_cohort"),
            )
        )
    a1, a2 = pl.col("a1").str.to_uppercase(), pl.col("a2").str.to_uppercase()
    lo = pl.min_horizontal(a1, a2)
    rows = (
        pl.concat(frames, how="vertical")
        .filter(
            pl.col("a2").is_not_null()
            & pl.col("beta").is_not_null()
            & (pl.col("neg_log10p") >= strong)
        )
        .filter(
            ~((a1.str.len_chars() == 1) & (a2 == a1.replace_strict(comp, default=None))).fill_null(
                False
            )
        )
        .with_columns(
            pl.concat_str([lo, pl.max_horizontal(a1, a2)], separator="/").alias("alleles"),
            pl.when(a1 == lo).then(pl.col("beta")).otherwise(-pl.col("beta")).alias("_b"),
            _protein_key(platform).alias("protein"),
        )
        .explode("protein")
        .filter(pl.col("protein").is_not_null())
        .with_row_index("_row")
    )
    site = ["protein", "chrom", "pos", "alleles"]
    best = (
        rows.sort(["neg_log10p", "catalog"], descending=[True, False])
        .unique(subset=["catalog", *site], keep="first", maintain_order=True)
        .select(
            *site,
            pl.col("catalog").alias("_other"),
            pl.col("_cohort").alias("_ocohort"),
            pl.col("_b").alias("_ob"),
        )
    )
    pairs = (
        rows.join(best, on=site)
        .filter(pl.col("_ocohort") != pl.col("_cohort"))
        .with_columns((pl.col("_b").sign() == pl.col("_ob").sign()).alias("_agree"))
    )
    out = (
        pairs.group_by(
            "_row",
            "catalog",
            *site,
            "ancestry",
            "neg_log10p",
            "beta",
            maintain_order=True,
        )
        .agg(
            pl.col("_ocohort").filter(pl.col("_agree")).n_unique().cast(pl.Int64).alias("n_same"),
            pl.col("_ocohort")
            .filter(~pl.col("_agree"))
            .n_unique()
            .cast(pl.Int64)
            .alias("n_opposite"),
            pl.col("_other").filter(pl.col("_agree")).sort().str.join(",").alias("same"),
            pl.col("_other").filter(~pl.col("_agree")).sort().str.join(",").alias("opposite"),
        )
        .filter(
            (pl.col("n_opposite") >= min_opposing) & (pl.col("n_opposite") > pl.col("n_same") + 1)
        )
        .sort("catalog", "protein", "chrom", "pos", "alleles", "ancestry")
        .drop("_row")
    )
    return out


def derive_sign_exclusions(
    *, strong: float = SIGN_STRONG_NLP, min_opposing: int = SIGN_MIN_OPPOSING
) -> dict[str, frozenset[str]]:
    """Derives ``SIGN_EXCLUSIONS`` from the parsed catalogs.

    Runs ``sign_discordant_rows`` over the parsed catalogs (``CATALOG_NAMES``) of each
    platform, read as stored.

    Returns:
        Catalog file name to the proteins of that catalog that have a flagged row.
    """
    out: dict[str, frozenset[str]] = {}
    for platform in ("olink", "soma"):
        files = {n: f for n, f in CATALOG_NAMES.items() if CATALOG_REGISTRY[f][1] == platform}
        flagged = sign_discordant_rows(
            {n: pl.read_parquet(_PARSED_DIR / f) for n, f in files.items()},
            platform=platform,
            strong=strong,
            min_opposing=min_opposing,
        )
        for name, proteins in (
            flagged.group_by("catalog").agg(pl.col("protein").unique()).iter_rows()
        ):
            out[files[name]] = frozenset(proteins)
    return out


def load_reference_catalog(
    source: str | Path | Sequence[str | Path],
    *,
    platform: str,
    include_non_eur: bool = True,
    build: str | None = None,
    catalog_platform: str | None = None,
    with_rank: bool = False,
) -> pl.DataFrame:
    """Loads one or several parsed reference pQTL catalogs.

    Rows with a null ``pos`` (failed liftovers) are dropped and counted in the log.
    Several catalogs are concatenated; selection among their rows is left to
    ``select_catalog_rows``.

    Args:
        source: A catalog name (``CATALOG_NAMES``, resolved to the parsed parquet in
            ``PQTL_CATALOG_DIR``), a path to a ``.parquet``, or a list of them. Each
            catalog of a list must carry its own ``source`` value.
        platform: The trial's platform (``"olink"``, ``"soma"`` or ``"somascan"``).
        include_non_eur: When False, keeps only the ``KEPT_ANCESTRIES`` rows (EUR, ARIC
            ``EA`` and pooled ``MULTI``).
        build: Genome build of a catalog missing from ``CATALOG_REGISTRY``.
        catalog_platform: Platform of a catalog missing from ``CATALOG_REGISTRY``.
        with_rank: Whether to add ``CATALOG_RANK``, each row's catalog position in
            ``source`` (0 first), for ``merge_catalog_list_order``.

    Returns:
        One catalog with its stored columns, several reduced to the ``CATALOG_SCHEMA``
        columns; a missing ``eaf`` is added as null, and ``with_rank`` adds
        ``CATALOG_RANK``.

    Raises:
        ValueError: If no catalog is given; a name or path is unknown; a catalog is not
            hg38, is of another platform or mixes platforms,
            lacks a ``CATALOG_SCHEMA`` column or type, has no ``ancestry`` column (or no
            kept row) under ``include_non_eur=False``; or two catalogs share a
            ``source`` value.
        FileNotFoundError: If a name or path resolves to no file.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    if not sources:
        raise ValueError("No reference catalog given")
    trial_platform = get_platform(platform).name
    paths = [_catalog_path(src) for src in sources]
    frames = [
        _load_one_catalog(
            path,
            platform=trial_platform,
            include_non_eur=include_non_eur,
            build=build,
            catalog_platform=catalog_platform,
        )
        for path in paths
    ]
    if with_rank:
        frames = [
            f.with_columns(pl.lit(k, dtype=pl.Int64).alias(CATALOG_RANK))
            for k, f in enumerate(frames)
        ]
    if len(frames) == 1:
        return frames[0]
    seen: dict[str, str] = {}
    for src, frame in zip(sources, frames, strict=True):
        for name in frame["source"].unique().to_list():
            if name is None:
                raise ValueError(f"Catalog {src!r} has a null source value")
            if name in seen:
                raise ValueError(
                    f"Catalogs {seen[name]!r} and {src!r} share source {name!r}; "
                    "each catalog needs its own source value"
                )
            seen[name] = str(src)
    columns = list(CATALOG_SCHEMA) + ([CATALOG_RANK] if with_rank else [])
    return pl.concat([f.select(columns) for f in frames], how="vertical")


# ── Allele harmonization ────────────────────────────────────────────────────

_COMPLEMENT_MAP = {"A": "T", "C": "G", "G": "C", "T": "A"}

PALINDROMIC_MODES = ("rescue", "drop")
"""Values of ``TrialConfig.pqtl_palindromic`` and of ``align_catalog_alleles``'s
``palindromic``."""
PALINDROMIC: str = "drop"
"""Default of ``TrialConfig.pqtl_palindromic``: A/T and C/G SNPs are dropped.
``"rescue"`` orients them by allele frequency where both frequencies are clear of 0.5
(``PALINDROMIC_EAF_MARGIN``) and drops the rest."""
PALINDROMIC_EAF_MARGIN: float = 0.1
"""How far from 0.5 both the catalog's effect-allele frequency and the trial's frequency
of the same allele must be, on the same side, for an A/T or C/G SNP to be oriented: at
most 0.4 or at least 0.6. The trial frequency is calculated over all of the trial's
genotypes."""


def _alt_frequency(genotypes: CatalogGenotypes) -> pl.DataFrame:
    """Returns each variant's ALT frequency over the called dosages of every column."""
    d = genotypes.dosages
    with np.errstate(invalid="ignore", divide="ignore"):
        af = np.nansum(d, axis=1) / np.isfinite(d).sum(axis=1) / 2
    return pl.DataFrame(
        {"variant_id": list(genotypes.variant_ids), "_alt_af": af.astype(np.float64)},
        schema={"variant_id": pl.String, "_alt_af": pl.Float64},
    )


def align_catalog_alleles(
    catalog: pl.DataFrame,
    genotypes: CatalogGenotypes,
    *,
    palindromic: str = PALINDROMIC,
) -> pl.DataFrame:
    """Joins catalog rows to trial variants and expresses betas per ALT allele.

    Rows join to ``genotypes.catalog_variant`` on ``(chrom, pos, a1, a2)`` (chromosome
    normalized, alleles upper-cased, null ``a2`` matching null) and take its
    ``variant_id``. The PVAR ``(REF, ALT)`` of that variant, from ``genotypes.alleles``,
    then sets the beta direction:

    - ``a1`` (or its strand complement) is ALT: beta is kept.
    - ``a1`` (or its strand complement) is REF: beta is negated.
    - An A/T or C/G variant: the strand cannot be told from the alleles. With
      ``palindromic="drop"`` the row is dropped. With ``"rescue"`` the catalog's
      alleles are read on the forward strand, and the row is kept when the catalog
      gives both alleles and ``eaf``, and ``eaf`` and the trial's frequency of ``a1`` (from the called
      dosages of every genotype) are both at most ``0.5 - PALINDROMIC_EAF_MARGIN`` or
      both at least ``0.5 + PALINDROMIC_EAF_MARGIN``; ``a1`` == ALT keeps beta, ``a1``
      == REF negates it. Every other A/T or C/G row is dropped.

    Rows with no matching variant or a null beta are dropped. A catalog without an
    ``eaf`` column rescues nothing.

    Args:
        catalog: Catalog rows with ``chrom``, ``pos``, ``a1``, ``a2`` and ``beta``.
        genotypes: Trial genotypes at the catalog's variants.
        palindromic: ``"drop"`` or ``"rescue"`` (``PALINDROMIC_MODES``).

    Returns:
        The matched rows of ``catalog`` with ALT-oriented ``beta`` and a ``variant_id``
        column.

    Raises:
        ValueError: If ``catalog`` has no rows, has no ``a1`` column or an all-null
            one, or ``palindromic`` is not a known mode.
    """
    if catalog.height == 0:
        raise ValueError(
            "catalog is empty (0 rows): no catalog protein is measured in this "
            "trial; check the catalog's platform and the remap to trial IDs"
        )
    if "a1" not in catalog.columns or catalog["a1"].drop_nulls().is_empty():
        raise ValueError(
            "catalog has no effect alleles: column 'a1' is missing or all "
            f"null ({catalog.height} rows); re-parse it with alleles"
        )
    if palindromic not in PALINDROMIC_MODES:
        raise ValueError(
            f"palindromic must be one of {', '.join(PALINDROMIC_MODES)}; got {palindromic!r}"
        )

    chroms = catalog["chrom"].drop_nulls().unique().to_list()
    keys = ["_chrom", "pos", "_a1", "_a2"]
    variant = genotypes.catalog_variant.select(
        pl.col("chrom").alias("_chrom"),
        "pos",
        pl.col("a1").str.to_uppercase().alias("_a1"),
        pl.col("a2").str.to_uppercase().alias("_a2"),
        "variant_id",
    )
    pvar = pl.DataFrame(
        {
            "variant_id": list(genotypes.alleles),
            "_ref": [r.upper() for r, _ in genotypes.alleles.values()],
            "_alt": [a.upper() for _, a in genotypes.alleles.values()],
        },
        schema={"variant_id": pl.String, "_ref": pl.String, "_alt": pl.String},
    )
    comp_a1 = pl.col("_a1").replace_strict(_COMPLEMENT_MAP, default=None)
    joined = (
        catalog.with_columns(
            pl.col("chrom")
            .replace_strict({c: normalize_chrom(c) for c in chroms}, return_dtype=pl.String)
            .alias("_chrom"),
            pl.col("a1").str.to_uppercase().alias("_a1"),
            pl.col("a2").str.to_uppercase().alias("_a2"),
        )
        .join(variant, on=keys, how="left", nulls_equal=True, maintain_order="left")
        .join(pvar, on="variant_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("variant_id").is_null() | pl.col("beta").is_null())
            .then(pl.lit("unmatched"))
            .when(pl.col("_alt") == pl.col("_ref").replace_strict(_COMPLEMENT_MAP, default=None))
            .then(pl.lit("ambiguous"))
            .when((pl.col("_a1") == pl.col("_alt")) | (comp_a1 == pl.col("_alt")))
            .then(pl.lit("concordant"))
            .when((pl.col("_a1") == pl.col("_ref")) | (comp_a1 == pl.col("_ref")))
            .then(pl.lit("flipped"))
            .otherwise(pl.lit("unmatched"))
            .alias("_status")
        )
    )
    if palindromic == "rescue" and "eaf" in catalog.columns:
        lo, hi = 0.5 - PALINDROMIC_EAF_MARGIN, 0.5 + PALINDROMIC_EAF_MARGIN
        a1_is_alt = pl.col("_a1") == pl.col("_alt")
        f_a1 = pl.when(a1_is_alt).then(pl.col("_alt_af")).otherwise(1.0 - pl.col("_alt_af"))
        clear = ((pl.col("eaf") <= lo) & (f_a1 <= lo)) | ((pl.col("eaf") >= hi) & (f_a1 >= hi))
        literal = a1_is_alt | (pl.col("_a1") == pl.col("_ref"))
        rescued = (pl.col("_status") == "ambiguous") & pl.col("_a2").is_not_null() & literal & clear
        joined = joined.join(
            _alt_frequency(genotypes),
            on="variant_id",
            how="left",
            maintain_order="left",
        ).with_columns(
            pl.when(rescued.fill_null(False))
            .then(pl.when(a1_is_alt).then(pl.lit("rescued")).otherwise(pl.lit("rescued_flipped")))
            .otherwise(pl.col("_status"))
            .alias("_status")
        )
    counts = dict(joined["_status"].value_counts().iter_rows())
    logger.info(
        "allele alignment: %d concordant, %d flipped, %d palindromic rescued by frequency, "
        "%d ambiguous (dropped), %d without a matched variant (dropped)",
        counts.get("concordant", 0),
        counts.get("flipped", 0),
        counts.get("rescued", 0) + counts.get("rescued_flipped", 0),
        counts.get("ambiguous", 0),
        counts.get("unmatched", 0),
    )
    return (
        joined.filter(
            pl.col("_status").is_in(["concordant", "flipped", "rescued", "rescued_flipped"])
        )
        .with_columns(
            pl.when(pl.col("_status").is_in(["flipped", "rescued_flipped"]))
            .then(-pl.col("beta"))
            .otherwise(pl.col("beta"))
            .alias("beta")
        )
        .select(*catalog.columns, "variant_id")
    )


def fill_underflowed_nlp(catalog: pl.DataFrame, floor: float = NLP_RECOMPUTE_FLOOR) -> pl.DataFrame:
    """Recovers -log10(p) from beta/se where the catalog's p underflowed or is capped.

    Where ``beta`` and a
    positive ``se`` are present and ``neg_log10p`` is null or at least ``floor``, it
    becomes ``-log10(2 * norm.sf(|beta/se|))``, computed from ``norm.logsf`` so that it
    does not underflow (scipy's ``chi2.logsf`` returns -inf beyond z**2 of about 1400).
    Other rows with a usable ``se`` keep the catalog value.

    A row whose ``neg_log10p`` is null and that has no usable ``se`` (null or not
    positive) but has a ``beta`` gets ``neg_log10p = floor``, so it stays in the catalog
    instead of being dropped. A row without ``beta`` is left alone and is dropped later
    by ``select_catalog_rows``. The log gives, per ``source``, the rows recomputed, the
    rows set to the floor and the rows with no beta.
    """
    from scipy.stats import norm

    has_beta = pl.col("beta").is_not_null()
    usable_se = has_beta & (pl.col("se") > 0)
    recompute = usable_se & (pl.col("neg_log10p").is_null() | (pl.col("neg_log10p") >= floor))
    to_floor = pl.col("neg_log10p").is_null() & has_beta & ~usable_se.fill_null(False)

    z = (catalog["beta"] / catalog["se"]).abs().fill_null(np.nan).to_numpy()
    recovered = pl.Series(-(np.log(2) + norm.logsf(z)) / np.log(10), dtype=pl.Float64)
    out = catalog.with_columns(
        pl.when(recompute.fill_null(False))
        .then(recovered)
        .when(to_floor)
        .then(pl.lit(floor, dtype=pl.Float64))
        .otherwise(pl.col("neg_log10p"))
        .alias("neg_log10p")
    )
    group = pl.col("source") if "source" in catalog.columns else pl.lit("catalog").alias("source")
    counts = catalog.group_by(group, maintain_order=True).agg(
        recompute.fill_null(False).sum().alias("recomputed"),
        to_floor.sum().alias("floored"),
        (~has_beta).sum().alias("no_beta"),
    )
    for name, n_rec, n_floor, n_drop in counts.iter_rows():
        logger.info(
            "neg_log10p, catalog %s: %d rows recomputed from beta/se (null or >= %g), "
            "%d rows set to the floor %g (no p, no se), %d rows dropped (no beta)",
            name,
            n_rec,
            floor,
            n_floor,
            floor,
            n_drop,
        )
    return out


def select_catalog_rows(catalog: pl.DataFrame) -> pl.DataFrame:
    """Keeps the strongest pQTL per protein, then one protein per variant.

    Rows without ``beta`` or ``neg_log10p`` are dropped. The rest are sorted by
    ``neg_log10p`` (descending), then ``platform_id``, ``variant_id``, ``ancestry`` and
    ``source`` (ascending) as tiebreaks, so the result does not depend on input row
    order. ``ancestry`` and ``source`` separate rows that repeat a protein-variant pair
    with equal ``neg_log10p`` but a different beta (UKB-PPP ST9 + ST11, or a merged
    multi-catalog table); ``beta``, ``a1`` and ``a2`` break any remaining tie. The first
    row per ``platform_id`` is kept, then the first row per ``variant_id``: proteins
    sharing a causal variant would give collinear predictions.

    When ``catalog`` carries ``CATALOG_RANK`` (only ``catalog_merge = "list_order"``
    keeps it, ``merge_catalog_list_order``), the one-protein-per-variant step ranks by
    catalog rank first and then as above, so a later catalog's protein cannot evict the
    priority catalog's protein at a shared variant; the rank column is dropped and the
    rows come back in the order above.
    """
    tie_keys = [
        "neg_log10p",
        "platform_id",
        "variant_id",
        "ancestry",
        "source",
        "beta",
        *[c for c in ("a1", "a2") if c in catalog.columns],
    ]
    descending = [True] + [False] * (len(tie_keys) - 1)
    per_protein = (
        catalog.filter(pl.col("beta").is_not_null() & pl.col("neg_log10p").is_not_null())
        .sort(tie_keys, descending=descending, nulls_last=True)
        .unique(subset=["platform_id"], keep="first", maintain_order=True)
    )
    if CATALOG_RANK not in per_protein.columns:
        return per_protein.unique(subset=["variant_id"], keep="first", maintain_order=True)
    return (
        per_protein.sort(
            [CATALOG_RANK, *tie_keys], descending=[False, *descending], nulls_last=True
        )
        .unique(subset=["variant_id"], keep="first", maintain_order=True)
        .drop(CATALOG_RANK)
        .sort(tie_keys, descending=descending, nulls_last=True)
    )


def merge_catalog_list_order(
    catalog: pl.DataFrame,
    genotypes: CatalogGenotypes,
    protein_ids: Sequence[str],
    *,
    nlp_recompute_floor: float = NLP_RECOMPUTE_FLOOR,
    palindromic: str = PALINDROMIC,
) -> pl.DataFrame:
    """Returns the rows of ``catalog`` that ``catalog_merge = "list_order"`` scores.

    A row is usable for a protein when it would reach ``select_catalog_rows``: its
    ``platform_id`` is in ``protein_ids``, it aligns to a trial variant
    (``align_catalog_alleles`` after ``fill_underflowed_nlp``) and it has a beta and a
    ``neg_log10p``. Per protein, the lowest ``CATALOG_RANK`` with a usable row wins, and
    every row of that protein at that rank is kept, usable or not (the scorer drops the
    rest as it always does). Proteins with no usable row and every other rank's rows are
    dropped.

    Args:
        catalog: Catalog rows with ``CATALOG_RANK`` (``load_reference_catalog`` with
            ``with_rank=True``).
        genotypes: Trial genotypes at the catalog's variants.
        protein_ids: The trial's protein IDs that can be scored.
        nlp_recompute_floor: Passed to ``fill_underflowed_nlp``.
        palindromic: Passed to ``align_catalog_alleles``.

    Returns:
        The kept rows with ``CATALOG_RANK``, in ``catalog``'s order;
        ``select_catalog_rows`` reads the rank to keep the priority catalog's protein at
        a variant two proteins share.

    Raises:
        ValueError: If ``catalog`` has no ``CATALOG_RANK`` column.
    """
    if CATALOG_RANK not in catalog.columns:
        raise ValueError(
            f"catalog_merge = 'list_order' needs the {CATALOG_RANK} column "
            "(load_reference_catalog(..., with_rank=True))"
        )
    usable = align_catalog_alleles(
        fill_underflowed_nlp(catalog, nlp_recompute_floor),
        genotypes,
        palindromic=palindromic,
    ).filter(
        pl.col("platform_id").is_in(list(protein_ids))
        & pl.col("beta").is_not_null()
        & pl.col("neg_log10p").is_not_null()
    )
    first = usable.group_by("platform_id").agg(pl.col(CATALOG_RANK).min())
    kept = catalog.join(first, on=["platform_id", CATALOG_RANK], how="semi", maintain_order="left")
    logger.info(
        "catalog_merge list_order: %d of %d rows kept; proteins by catalog rank %s",
        kept.height,
        catalog.height,
        dict(sorted(first[CATALOG_RANK].value_counts().iter_rows())),
    )
    return kept


# ── Reference-mode scoring ──────────────────────────────────────────────────


def _drop_rare_lead_variants(
    matched: pl.DataFrame, genotypes: CatalogGenotypes, maf_min: float
) -> pl.DataFrame:
    """Keeps the rows of ``matched`` whose variant's trial MAF is at least ``maf_min``.

    MAF is the smaller of the ALT and REF allele frequencies over every genotype column
    (the PSAM IIDs, mapped or not), from the called dosages. A variant with no called
    genotype fails. The protein whose lead variant fails is dropped, not moved to a
    weaker variant.
    """
    row = {v: i for i, v in enumerate(genotypes.variant_ids)}
    d = genotypes.dosages[np.array([row[v] for v in matched["variant_id"]], dtype=np.int64)]
    called = np.isfinite(d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        af = np.nansum(d, axis=1) / called / 2
    keep = np.minimum(af, 1 - af) >= maf_min
    logger.info(
        "pQTL MAF floor %g: %d of %d lead variants kept",
        maf_min,
        int(keep.sum()),
        matched.height,
    )
    return matched.filter(pl.Series(keep))


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - x.mean(axis=1, keepdims=True)) / sd


def reference_pqtl_scores(
    Z: np.ndarray,
    protein_ids: list[str],
    genotypes: CatalogGenotypes,
    catalog: pl.DataFrame,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str],
    *,
    weight_exponent: float = WEIGHT_EXPONENT,
    dosage_scaling: Literal["standardized", "centered"] = "standardized",
    maf_min: float = MAF_MIN,
    contribution_cap: float = CONTRIBUTION_CAP,
    nlp_recompute_floor: float = NLP_RECOMPUTE_FLOOR,
    palindromic: str = PALINDROMIC,
) -> np.ndarray:
    """Scores every sample against every genotype with a reference pQTL catalog.

    Underflowed and ceiling-limited p-values are recovered (``fill_underflowed_nlp``),
    the catalog is joined to the trial variants and its betas ALT-oriented
    (``align_catalog_alleles``), and one row per protein and variant is kept
    (``select_catalog_rows``). Each genotype's predicted protein vector is dosage x
    beta. Observed (column-standardized) and predicted vectors are row-standardized,
    multiplied per protein by ``nlp ** weight_exponent`` and row-standardized again, so
    each protein enters the correlation with weight ``nlp ** (2 * weight_exponent)``.
    Each sample's correlations are robust z-scored against the null of non-self
    genotypes.

    Args:
        Z: Residualized protein matrix, shape (n_samples, n_proteins).
        protein_ids: Protein IDs (columns of ``Z``), matched against the catalog's
            ``platform_id``.
        genotypes: Trial genotypes at the catalog positions
            (``pqtl.extract_catalog_genotypes``); its columns are the genotypes in
            ``geno_usubjids``.
        catalog: Reference catalog with ``chrom``, ``pos``, ``a1``, ``a2``,
            ``platform_id``, ``beta``, ``se`` and ``neg_log10p``. The caller may
            pre-filter it by ancestry.
        sample_usubjids: USUBJID per sample (rows of ``Z``).
        geno_usubjids: USUBJID per genotype column of ``genotypes``. None marks a
            genotype with no known subject: it is in every sample's null and is no
            sample's own.
        weight_exponent: Per-side exponent on ``neg_log10p`` (``WEIGHT_EXPONENT``).
        dosage_scaling: ``"standardized"`` uses ``genotypes.standardized`` (per-variant
            centered and scaled); ``"centered"`` uses raw dosages minus the per-variant
            mean, missing set to 0, which does not up-weight rare variants.
        maf_min: Proteins whose lead variant has a trial MAF below this are left out (0
            keeps all; ``_drop_rare_lead_variants``).
        contribution_cap: Each protein's term in a correlation, ``observed *
            predicted`` after the final row-standardization, is clipped to
            ``+-contribution_cap`` before the sum over proteins; 0 is no bound.
        nlp_recompute_floor: ``neg_log10p`` at or above this is recomputed from beta/se
            where the row has a usable se; a row with no p and no se is set to it
            (``fill_underflowed_nlp``).
        palindromic: ``"rescue"`` orients A/T and C/G SNPs by allele frequency where it
            can, ``"drop"`` drops them all (``align_catalog_alleles``).

    Returns:
        Robust z of each sample against each genotype, shape (n_samples, n_genotyped).

    Raises:
        ValueError: If ``dosage_scaling``, ``maf_min`` (outside [0, 0.5]) or
            ``contribution_cap`` (negative) is invalid, or no catalog row matches a
            measured protein at a genotyped position.
    """
    return _reference_pqtl_scores(
        Z,
        protein_ids,
        genotypes,
        catalog,
        sample_usubjids,
        geno_usubjids,
        weight_exponent=weight_exponent,
        dosage_scaling=dosage_scaling,
        maf_min=maf_min,
        contribution_cap=contribution_cap,
        nlp_recompute_floor=nlp_recompute_floor,
        palindromic=palindromic,
    )[0]


# 1 - r**2 between the label's predicted vector and another genotype's at or below this:
# the other is the label's genome again, or nearly, with nothing to add as a second
# component.
# Duplicate genomes (KING_DUPLICATE) are left out by kinship in pqtl_matrix.
MIXTURE_MIN_RESIDUAL_VAR: float = 0.05


def mixture_scores(
    obs: np.ndarray,
    predicted: np.ndarray,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str | None],
    *,
    r: np.ndarray | None = None,
    same_genome: np.ndarray | None = None,
) -> np.ndarray:
    """Returns the two-genotype mixture score of each sample (``pqtl_mixture_z``).

    For a sample whose label is genotype ``g``, fitting the observed vector on ``g``'s
    predicted vector and another genotype ``j``'s jointly (least squares) improves the
    fit over ``g`` alone by ``u_ij**2``, where ``u_ij = (r_ij - r_ig r_gj) / sqrt(1 -
    r_gj**2)`` (the partial correlation numerator; ``r`` the plain correlations). The
    signed ``u_ij`` (positive when ``j`` enters with a positive weight) is robust-z scored against every other genotype as the second component
    (the null of ``reference_pqtl_scores``), and the score is the largest of those z.
    Left out of both the null and the maximum: a genotype whose predicted vector
    (nearly) repeats the label's (``1 - r_gj**2`` at or below
    ``MIXTURE_MIN_RESIDUAL_VAR``) and every genome that ``same_genome`` marks as the
    label's.

    Args:
        obs: Row-standardized, weighted observed vectors (samples x proteins) that
            ``reference_pqtl_scores`` correlates.
        predicted: The matching predicted vectors (genotypes x proteins).
        sample_usubjids: USUBJID per sample (rows of ``obs``).
        geno_usubjids: USUBJID per genotype (rows of ``predicted``), None for none.
        r: The plain correlation matrix ``obs @ predicted.T / n_proteins`` when the
            caller has it.
        same_genome: Genotypes x genotypes, True for two genotypes of one genome (KING
            kinship at or above ``KING_DUPLICATE``, as ``pqtl_matrix`` passes it).

    Returns:
        The score per sample; NaN when the label is ungenotyped or no second component
        is usable.
    """
    n_prot = obs.shape[1]
    if r is None:
        r = obs @ predicted.T / n_prot
    R = predicted @ predicted.T / n_prot
    g_index = {p: i for i, p in enumerate(geno_usubjids) if p is not None}
    n_geno = len(geno_usubjids)
    out = np.full(obs.shape[0], np.nan)
    for i in range(obs.shape[0]):
        own = g_index.get(sample_usubjids[i], -1)
        if own < 0:
            continue
        resid_var = 1.0 - R[own] ** 2
        usable = (np.arange(n_geno) != own) & (resid_var > MIXTURE_MIN_RESIDUAL_VAR)
        if same_genome is not None:
            usable &= ~same_genome[own]
        if not usable.any():
            continue
        u = np.full(n_geno, np.nan)
        u[usable] = (r[i, usable] - r[i, own] * R[own, usable]) / np.sqrt(resid_var[usable])
        out[i] = float(robust_z(u, usable)[usable].max())
    return out


def _reference_pqtl_scores(
    Z: np.ndarray,
    protein_ids: list[str],
    genotypes: CatalogGenotypes,
    catalog: pl.DataFrame,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str],
    *,
    weight_exponent: float,
    dosage_scaling: str,
    maf_min: float,
    contribution_cap: float,
    nlp_recompute_floor: float,
    palindromic: str,
    with_mixture: bool = False,
    same_genome: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns ``(S, mixture)``: ``reference_pqtl_scores`` and the mixture scores.

    ``mixture`` is ``mixture_scores`` of the same vectors (given ``same_genome``) with
    ``with_mixture``, else None.
    """
    if dosage_scaling == "standardized":
        G = genotypes.standardized
    elif dosage_scaling == "centered":
        d = genotypes.dosages
        G = np.nan_to_num(d - np.nanmean(d, axis=1, keepdims=True))
    else:
        raise ValueError(
            f"dosage_scaling must be 'standardized' or 'centered', got {dosage_scaling!r}"
        )

    if not 0 <= maf_min <= 0.5:
        raise ValueError(f"maf_min must be in [0, 0.5], got {maf_min!r}")
    if not contribution_cap >= 0:
        raise ValueError(f"contribution_cap must be >= 0, got {contribution_cap!r}")

    aligned = align_catalog_alleles(
        fill_underflowed_nlp(catalog, nlp_recompute_floor),
        genotypes,
        palindromic=palindromic,
    )
    matched = select_catalog_rows(aligned.filter(pl.col("platform_id").is_in(list(protein_ids))))
    if maf_min > 0:
        matched = _drop_rare_lead_variants(matched, genotypes, maf_min)
    n_matched = matched.height
    n_geno = len(geno_usubjids)
    logger.info(
        "reference pQTL scoring: %d aligned catalog variants; %d proteins "
        "matched (one variant each); %d genotypes (%d mapped to subjects)",
        aligned["variant_id"].n_unique(),
        n_matched,
        n_geno,
        sum(u is not None for u in geno_usubjids),
    )
    if n_matched == 0:
        raise ValueError(
            "reference pQTL scoring: no catalog variant matches a measured protein at a "
            "genotyped position, so no tube can be scored"
        )

    pid_to_col = {p: i for i, p in enumerate(protein_ids)}
    vid_to_row = {v: i for i, v in enumerate(genotypes.variant_ids)}
    prot_cols = np.array([pid_to_col[p] for p in matched["platform_id"]])
    var_rows = np.array([vid_to_row[v] for v in matched["variant_id"]])

    # Predicted protein vector per genotype: dosage * ALT beta.
    predicted = _standardize_rows(G[var_rows].T * matched["beta"].to_numpy())

    obs = Z[:, prot_cols].copy()
    col_sd = obs.std(axis=0, keepdims=True)
    col_sd[col_sd == 0] = 1.0
    obs = _standardize_rows((obs - obs.mean(axis=0, keepdims=True)) / col_sd)

    w = matched["neg_log10p"].to_numpy() ** weight_exponent
    predicted = _standardize_rows(predicted * w)
    obs = _standardize_rows(obs * w)
    if contribution_cap > 0:
        corr = np.empty((obs.shape[0], predicted.shape[0]))
        for i in range(obs.shape[0]):
            corr[i] = (
                np.clip(predicted * obs[i], -contribution_cap, contribution_cap).sum(axis=1)
                / n_matched
            )
    else:
        corr = obs @ predicted.T / n_matched

    g_index = {p: i for i, p in enumerate(geno_usubjids) if p is not None}
    S = np.full((Z.shape[0], n_geno), np.nan)
    for i in range(Z.shape[0]):
        own = g_index.get(sample_usubjids[i], -1)
        S[i] = robust_z(corr[i], np.arange(n_geno) != own)
    if not with_mixture:
        return S, None
    plain = corr if contribution_cap == 0 else None
    return S, mixture_scores(
        obs, predicted, sample_usubjids, geno_usubjids, r=plain, same_genome=same_genome
    )


def reference_pqtl_concordance(
    Z: np.ndarray,
    protein_ids: list[str],
    genotypes: CatalogGenotypes,
    catalog: pl.DataFrame,
    sample_usubjids: np.ndarray,
    geno_usubjids: list[str],
    *,
    best_min: float = PQTL_BEST_MIN,
    weight_exponent: float = WEIGHT_EXPONENT,
    dosage_scaling: Literal["standardized", "centered"] = "standardized",
    maf_min: float = MAF_MIN,
    contribution_cap: float = CONTRIBUTION_CAP,
    nlp_recompute_floor: float = NLP_RECOMPUTE_FLOOR,
    palindromic: str = PALINDROMIC,
    relative: np.ndarray | None = None,
) -> pl.DataFrame:
    """Returns per-sample reference pQTL concordance scores.

    Runs ``reference_pqtl_scores`` and extracts the self and best-other z of each sample
    with a finite score (``pqtl.extract_concordance``). The calls are made by
    ``pqtl.classify_pqtl``.

    Args:
        Z: Residualized protein matrix, shape (n_samples, n_proteins).
        protein_ids: Protein IDs (columns of ``Z``).
        genotypes: Trial genotypes at the catalog positions.
        catalog: Reference catalog.
        sample_usubjids: USUBJID per sample (rows of ``Z``).
        geno_usubjids: USUBJID per genotype column of ``genotypes``.
        best_min: Passed to ``pqtl.extract_concordance``.
        weight_exponent: Passed to ``reference_pqtl_scores``.
        dosage_scaling: Passed to ``reference_pqtl_scores``.
        maf_min: Passed to ``reference_pqtl_scores``.
        contribution_cap: Passed to ``reference_pqtl_scores``.
        nlp_recompute_floor: Passed to ``reference_pqtl_scores``.
        palindromic: Passed to ``reference_pqtl_scores``.
        relative: Passed to ``pqtl.extract_concordance``.

    Returns:
        One row per sample with a finite score: ``USUBJID``, ``pqtl_self_z``,
        ``pqtl_best_z``, ``pqtl_best_USUBJID``, ``pqtl_best_nonrel_z`` and
        ``pqtl_best_nonrel_USUBJID``. A sample whose labeled subject is not genotyped
        gets a row with a null ``pqtl_self_z``.
    """
    S = reference_pqtl_scores(
        Z,
        protein_ids,
        genotypes,
        catalog,
        sample_usubjids,
        geno_usubjids,
        weight_exponent=weight_exponent,
        dosage_scaling=dosage_scaling,
        maf_min=maf_min,
        contribution_cap=contribution_cap,
        nlp_recompute_floor=nlp_recompute_floor,
        palindromic=palindromic,
    )
    return extract_concordance(
        S, sample_usubjids, geno_usubjids, best_min=best_min, relative=relative
    )


# ── QC-instance orchestration ───────────────────────────────────────────────


@dataclass(frozen=True)
class PqtlEvidence:
    """A score matrix and the genotypes kept for identity-event enrichment.

    Attributes:
        S: ``S[i, j]`` is the robust z of sample ``sample_ids[i]`` against the genotype
            of ``geno_usubjids[j]``.
        sample_ids: Sample ID per row of ``S``.
        geno_usubjids: Every genotype of the PSAM except the ones whose link was dropped
            (duplicate genomes), including participants with no proteomics, so a named
            donor
            need not be a sample's subject, nor a row of ADSL or the manifest.
        dosages: Raw catalog-variant dosages, columns in ``geno_usubjids`` order.
        kinship: KING kinship between every pair of those genotypes, same order.
        genetic_sex: ``USUBJID`` and ``pqtl.SEX_COLUMNS`` per genotype.
        dna_link: The DNA-link table (``schemas.DNA_LINK_SCHEMA``), which also lists the
            genotypes whose link was dropped.
    """

    S: np.ndarray
    sample_ids: list[str]
    geno_usubjids: list[str]
    dosages: np.ndarray
    kinship: np.ndarray
    genetic_sex: pl.DataFrame
    dna_link: pl.DataFrame


def remap_catalog_platform_ids(
    catalog: pl.DataFrame,
    analyte_metadata: pl.DataFrame,
    *,
    platform: str,
    protein_id_col: str,
    uniprot_col: str,
) -> pl.DataFrame:
    """Points ``platform_id`` at the trial's own protein IDs.

    Olink: rows are joined to the trial's OlinkIDs through UniProt; a UniProt measured by several trial assays yields
    one row per assay. Both sides' UniProt strings are split on ``,``, ``;`` and ``_``
    first (``_uniprot_tokens``), and a row reaches every trial assay sharing any
    accession with it, once: a catalog row for a set of UniProts (``A,B``) and a trial
    assay for a set (``A_B``) both match. A row keeps its own ``uniprot`` string. SomaScan: deCODE SeqIds
    use ``_`` where ADAT SeqIds use ``-``.

    Args:
        catalog: Catalog rows (``CATALOG_SCHEMA``).
        analyte_metadata: The trial's analytes with ``protein_id_col`` and
            ``uniprot_col``.
        platform: The trial's platform, ``"olink"`` or ``"soma"``.
        protein_id_col: The trial's protein ID column.
        uniprot_col: The trial's UniProt column.

    Returns:
        The rows that map to a trial protein, in catalog order and then trial protein
        ID, so ties left for ``select_catalog_rows`` resolve the same way every run.
    """
    trial = (
        analyte_metadata.select(protein_id_col, uniprot_col)
        .unique()
        .sort(protein_id_col, uniprot_col, nulls_last=True)
    )
    if platform == "soma":
        # SeqId alone identifies the aptamer; its UniProt may be null.
        seqids = trial[protein_id_col].drop_nulls()
        out = catalog.with_columns(pl.col("platform_id").str.replace_all("_", "-")).filter(
            pl.col("platform_id").is_in(seqids.implode())
        )
    else:
        tokens = (
            trial.drop_nulls()
            .select(
                pl.col(protein_id_col).alias("platform_id"),
                _uniprot_tokens(uniprot_col).alias("_tok"),
            )
            .explode("_tok")
            .drop_nulls("_tok")
            .unique()
        )
        out = (
            catalog.with_row_index("_row")
            .with_columns(_uniprot_tokens("uniprot").alias("_tok"))
            .explode("_tok")
            .drop("platform_id")
            .join(tokens, on="_tok", how="inner")
            .unique(subset=["_row", "platform_id"], keep="first")
            .sort("_row", "platform_id")
            .select(catalog.columns)
        )
    logger.info(
        "catalog remapped to trial %s IDs: %d / %d rows",
        protein_id_col,
        out.height,
        catalog.height,
    )
    return out


MIN_GENOTYPED_FRACTION: float = 0.10
"""Fewest proteomic study subjects, as a fraction, that must be in the genotype PSAM;
below it the PGEN IIDs are taken not to be USUBJIDs and ``pqtl_concordance_for_qc``
raises."""


def _genotype_labels(iids: list[str], subjects: set[str]) -> list[str]:
    """Returns the USUBJID of each genotype IID: the IID itself, for every PSAM row.

    Every genotype can be a sample's own genotype or the named donor of a tube. An
    identifier outside ``subjects`` is only logged here; the minimum-overlap check
    counts proteomic subjects alone.

    Raises:
        ValueError: If an IID repeats (a USUBJID has one genotype).
    """
    repeated = sorted(i for i, n in Counter(iids).items() if n > 1)
    if repeated:
        raise ValueError(
            f"the genotype PSAM repeats {len(repeated)} IID(s); IIDs are USUBJIDs and must be unique"
        )
    n_only = sum(i not in subjects for i in iids)
    if n_only:
        logger.info(
            "%d of %d genotype IIDs are not proteomic subjects; they stay nameable donors "
            "and null-distribution columns",
            n_only,
            len(iids),
        )
    return list(iids)


def _check_genotype_overlap(
    labeled: set[str],
    labels: list[str],
    iids: list[str],
) -> None:
    """Checks that ``MIN_GENOTYPED_FRACTION`` or more of ``labeled`` is in the PSAM."""
    genotyped = set(labels)
    n_ok = len(labeled & genotyped)
    if labeled and n_ok / len(labeled) >= MIN_GENOTYPED_FRACTION:
        return
    raise ValueError(
        f"pQTL concordance: {n_ok} of {len(labeled)} labeled subjects are "
        f"genotyped (their USUBJID is a PSAM IID; {len(genotyped)} genotype IIDs in "
        f"the PSAM); at least "
        f"{MIN_GENOTYPED_FRACTION:.0%} are required.  The PGEN IIDs must be "
        f"USUBJIDs ({len(iids)} IIDs read)"
    )


def require_pqtl_inputs(config) -> None:
    """Checks that the genotype directory and summary files of a pQTL run exist.

    A run with ``pqtl_enabled`` never skips pQTL: a missing input stops it. The KING
    table and the PC file are checked when the config names them; whether a source
    needs them is the config's check.

    Raises:
        FileNotFoundError: If ``config.pqtl_genotype_dir`` is not a directory, or
            ``pqtl_kinship_table`` or ``pqtl_ancestry_pcs`` is set and not a file.
    """
    genotype_dir = Path(config.pqtl_genotype_dir).expanduser()
    if not genotype_dir.is_dir():
        raise FileNotFoundError(
            f"pqtl_enabled is true but pqtl_genotype_dir {config.pqtl_genotype_dir!r} "
            "is not a directory"
        )
    for key in ("pqtl_kinship_table", "pqtl_ancestry_pcs"):
        path = getattr(config, key)
        if path is not None and not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"{key} {path!r} is not a file")


@dataclass(frozen=True, eq=False)
class PqtlInputs:
    """Everything pQTL concordance reads from a QC instance, before any scoring key.

    Attributes:
        X: The identity matrix (``build_identity_matrix``).
        samples: The rows of ``X``, with the technical ``fit`` mask.
        protein_ids: The columns of ``X``.
        adsl: The collected ADSL.
        plate_col: The plate column the residualization reads.
        sample_id_col: The sample ID column of ``samples``.
        analyte_metadata: The trial's ``protein_id_col`` and ``uniprot_col``, for the
            catalog remap.
        protein_id_col: The trial's protein ID column.
        uniprot_col: The trial's UniProt column.
        psam_iids: The genotype IIDs (USUBJIDs) in PSAM order.
        labels: The USUBJID of each genotype: the IID itself, since every genotype is
            labeled, proteomic subject or not, except a genome a ``dna_relink``
            correction files under its owner, which carries the owner's USUBJID.
        sex_preds: ``SampleID`` and ``p_male``; None without a sex model.
        platform_name: The trial's platform.
        residuals: The ``identity.ResidualCache`` shared with the identity stage when
            the driver built one matrix for both, else None; it is not saved with captured
            inputs.
        kinship: KING kinship between the genotypes, in ``labels`` order, from
            ``pqtl_kinship_table`` (``genome_tables.kinship_from_table``); None when
            the config names no table.
        pcs: Every projected ancestry PC of the genotypes, rows in ``labels`` order,
            from ``pqtl_ancestry_pcs`` (``genome_tables.pcs_for``); None when the
            config names no file.
    """

    X: np.ndarray
    samples: pl.DataFrame
    protein_ids: list[str]
    adsl: pl.DataFrame
    plate_col: str
    sample_id_col: str
    analyte_metadata: pl.DataFrame
    protein_id_col: str
    uniprot_col: str
    psam_iids: list[str]
    labels: list[str]
    sex_preds: pl.DataFrame | None
    platform_name: str
    residuals: ResidualCache | None = None
    kinship: np.ndarray | None = None
    pcs: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class PqtlMatrix:
    """Scores and genotype summaries before the duplicate-genome step (``pqtl_matrix``).

    Nothing here depends on a pQTL call key (``pqtl_best_min``, the floors, the kinship
    cuts, ``pqtl_relative_margin``): ``pqtl_calls`` applies those, and the
    duplicate-genome step.

    Attributes:
        S_mapped: ``S_mapped[i, j]`` is the robust z of sample ``sample_ids[i]`` against
            the genotype of ``mapped_usubjids[j]``.
        mapped_usubjids: Every genotype of the PSAM, duplicates included, whether or not
            its participant has proteomics.
        kinship_mapped: KING kinship between those genotypes: the genome-wide KING
            table under ``pqtl_kinship_source = "table"`` (NaN for a pair it does not
            list), else computed from ``dosages_mapped``.
        dosages_mapped: Their catalog-variant dosages.
        pcs_mapped: Their projected ancestry PCs (``PqtlInputs.pcs``), or None when
            the config names no PC file.
        genetic_sex_mapped: Their ``USUBJID`` and ``pqtl.SEX_COLUMNS``.
        links: ``pqtl.link_evidence`` output.
        usubjids: USUBJID per sample, following ``sample_ids``.
        sample_ids: Sample ID per row of ``S_mapped``.
        fit: The technical ``fit`` mask, following ``sample_ids``.
        platform_name: The trial's platform.
        mixture_z: ``mixture_scores`` against each sample's labeled genotype (NaN for
            an ungenotyped label), following ``sample_ids``.
    """

    S_mapped: np.ndarray
    mapped_usubjids: list[str]
    kinship_mapped: np.ndarray
    dosages_mapped: np.ndarray
    genetic_sex_mapped: pl.DataFrame
    links: pl.DataFrame
    usubjids: np.ndarray
    sample_ids: list[str]
    fit: np.ndarray
    platform_name: str
    # None in a matrix saved without the mixture score; its calls carry a null
    # pqtl_mixture_z.
    mixture_z: np.ndarray | None = None
    pcs_mapped: np.ndarray | None = None


PQTL_CALL_KEYS: tuple[str, ...] = (
    "study_name",
    "identity_sex_prob",
    "pqtl_best_min",
    "pqtl_margin_min",
    "pqtl_match_floor",
    "pqtl_mismatch_floor",
    "pqtl_relative_kinship",
    "pqtl_duplicate_kinship",
    "pqtl_ungenotyped_all_tubes",
    "pqtl_relative_margin",
    "pqtl_ungenotyped_naming",
    "pqtl_relative_name_margin",
    "pqtl_null_by_ancestry",
    "pqtl_null_pcs",
    "pqtl_null_scope",
    "pqtl_pcs_source",
)
"""The ``TrialConfig`` fields ``pqtl_calls`` reads."""


# ── The ancestry-aware null ──

# The null removes each tube's ancestry location, a robust leave-one-out fit on the top
# genotype PCs, from the candidates' z: location only, candidates only (the own genotype
# keeps its z). The PCs are the projected ones (pqtl_pcs_source = "projected") or the
# catalog-variant dosages' own (pqtl_pcs_source = "catalog").

PCS_MIN_GENOTYPES: int = 50
"""A panel with fewer genotypes raises: its principal components and the fit under them
are noise."""
PCS_MIN_GAP: float = 0.05
"""The smallest relative singular-value gap ``(s_k - s_{k+1}) / s_k`` at the cut of the
in-data PCs (``pqtl_pcs_source = "catalog"``). Below it the k-th and (k+1)th vectors
are near-tied, and which one enters the top-k span depends on the LAPACK path. Projected
PCs are read from a file, so it does not apply to them
(``genome_tables.check_projected_pcs``)."""
HUBER_C: float = 1.345
"""Huber's tuning constant (95% efficiency at a normal row, bounded influence beyond
it)."""
HUBER_ITERS: int = 2
"""Reweighting steps of the location fit (the first fit is plain least squares)."""
LEVERAGE_MIN: float = 1e-8
"""``1 - h`` at or below this is a leverage-one genotype: its leave-one-out location is
undefined, and the fit raises."""


def check_singular_gap(s: np.ndarray, k: int, *, study: str) -> float:
    """Returns the relative gap ``(s_k - s_{k+1}) / s_k`` of the singular values ``s``.

    Args:
        s: Singular values, descending.
        k: Number of principal components kept.
        study: Study name, for the error message.

    Raises:
        ValueError: If ``s`` has no (k+1)th value or the gap is below ``PCS_MIN_GAP``.
    """
    if len(s) <= k:
        raise ValueError(
            f"{study}: the genotype matrix has {len(s)} singular values; {k} principal components need at least {k + 1}"
        )
    gap = float((s[k - 1] - s[k]) / s[k - 1])
    if not gap >= PCS_MIN_GAP:
        raise ValueError(
            f"{study}: the relative singular-value gap of the genotype matrix at k = {k} is {gap:.4f}, below {PCS_MIN_GAP}: "
            f"the {k}th and {k + 1}th principal components are near-tied, so the PCs would depend on the LAPACK path "
            "(pqtl.null_pcs must cut at a gap)"
        )
    return gap


_PCS_CACHE: dict[tuple, tuple[np.ndarray, float]] = {}
PCS_CACHE_SIZE: int = 32
"""Matrices whose PCs are kept, oldest evicted first. ``pqtl_calls`` may run several
times on one saved matrix (keys differ by k), and one SVD of a real panel takes
seconds, so the cache holds several panels. Each entry is n_genotypes x k float64 scores,
so the cache stays small."""


def clear_pcs_cache() -> None:
    """Empties the cache of ``genotype_pcs``."""
    _PCS_CACHE.clear()


def genotype_pcs(dosages: np.ndarray, k: int, *, study: str) -> tuple[np.ndarray, float]:
    """Returns ``_genotype_pcs``, computed once per content of ``dosages`` and ``k``.

    The cache holds ``PCS_CACHE_SIZE`` matrices keyed by a digest of the dosage bytes,
    so a changed matrix or k recomputes. A cached result is the call's own output, so
    determinism holds; a copy is returned, so a caller cannot change what is kept. A
    panel that raises (``check_singular_gap``) is not kept.
    """
    d = np.ascontiguousarray(dosages)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(d.data)
    key = (d.shape, str(d.dtype), digest.hexdigest(), k)
    if key not in _PCS_CACHE:
        _PCS_CACHE[key] = _genotype_pcs(d, k, study=study)
        while len(_PCS_CACHE) > PCS_CACHE_SIZE:
            del _PCS_CACHE[next(iter(_PCS_CACHE))]
    scores, gap = _PCS_CACHE[key]
    return scores.copy(), gap


def _genotype_pcs(dosages: np.ndarray, k: int, *, study: str) -> tuple[np.ndarray, float]:
    """Returns the top ``k`` principal-component scores of the genotypes and the gap.

    The genotypes are the columns of ``dosages`` (catalog variants in rows). Variants
    are centered and scaled over the panel, missing dosages set to 0 and monomorphic
    variants left out; the SVD is ``numpy.linalg.svd`` of the centered
    genotype-by-variant matrix, always the same call. Each component's sign is fixed so
    that its largest absolute score is positive.

    Returns:
        ``(scores, gap)``: genotypes x ``k`` scores and the relative singular-value gap
        at the cut.

    Raises:
        ValueError: If the gap is below ``PCS_MIN_GAP`` (``check_singular_gap``).
    """
    d = np.asarray(dosages, dtype=np.float64)
    mu = np.nanmean(d, axis=1, keepdims=True)
    sd = np.nanstd(d, axis=1, keepdims=True)
    keep = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
    G = np.nan_to_num((d[keep] - mu[keep]) / sd[keep]).T  # genotypes x variants
    G = G - G.mean(axis=0)
    U, s, _ = np.linalg.svd(G, full_matrices=False)
    gap = check_singular_gap(s, k, study=study)
    scores = U[:, :k] * s[:k]
    for j in range(k):
        if scores[np.argmax(np.abs(scores[:, j])), j] < 0:
            scores[:, j] = -scores[:, j]
    return scores, gap


def _row_location(
    y: np.ndarray, X: np.ndarray, *, robust: bool, study: str, row: int
) -> np.ndarray:
    """Returns the leave-one-out fitted value of every observation of ``y`` on ``X``.

    The fit is least squares or, with ``robust``, ``HUBER_ITERS`` Huber IRLS steps
    (weights ``min(1, c s / |e|)``, ``s`` the MAD of the current residuals), so a large
    value in the row, the true donor or a second donor of the same ancestry, cannot bend
    the surface at its own group's position. The deleted residual is ``e_i / (1 -
    h_ii)`` with the weighted leverage, so an observation's location never uses its own
    z (exactly for fixed weights; the weights come from the full fit).

    Raises:
        ValueError: If a genotype has leverage 1 (``1 - h`` at or below
            ``LEVERAGE_MIN``).
    """
    w = np.ones(y.size)
    for step in range(HUBER_ITERS + 1 if robust else 1):
        sw = np.sqrt(w)
        Q, R = np.linalg.qr(X * sw[:, None])
        coef = np.linalg.solve(R, Q.T @ (y * sw))
        e = y - X @ coef
        if not robust or step == HUBER_ITERS:
            break
        scale = 1.4826 * np.median(np.abs(e - np.median(e)))
        w = (
            np.ones(y.size)
            if scale <= 0
            else np.minimum(1.0, HUBER_C * scale / np.maximum(np.abs(e), 1e-12))
        )
    one_minus_h = 1.0 - np.einsum("ij,ij->i", Q, Q)
    if one_minus_h.min() <= LEVERAGE_MIN:
        raise ValueError(
            f"{study}: tube row {row} has a genotype of leverage 1 in the ancestry location fit (1 - h = {one_minus_h.min():.2e}): "
            "its leave-one-out location is undefined"
        )
    return y - e / one_minus_h


def _usable(S_row: np.ndarray, exclude: int) -> np.ndarray:
    """Returns the columns of a z row the location fit reads.

    These are the finite ones other than the tube's own (``exclude``) genotype.
    """
    use = np.isfinite(S_row)
    if exclude >= 0:
        use[exclude] = False
    return use


def ancestry_locations(
    S: np.ndarray,
    pcs: np.ndarray,
    exclude: np.ndarray,
    *,
    robust: bool = True,
    study: str = "",
) -> np.ndarray:
    """Returns each candidate's ancestry location in each tube.

    The location is the leave-one-out fitted value of the tube's z row on ``[1, pcs]``
    over the finite columns other than ``exclude[i]`` (the tube's own genotype column,
    -1 for none); NaN where the column is excluded or its z is not finite.
    """
    X = np.column_stack([np.ones(S.shape[1]), pcs])
    loc = np.full(S.shape, np.nan)
    for i in range(S.shape[0]):
        use = _usable(S[i], int(exclude[i]))
        if use.any():
            loc[i, use] = _row_location(S[i, use], X[use], robust=robust, study=study, row=i)
    return loc


def ancestry_null_from_pcs(
    S: np.ndarray,
    pcs: np.ndarray,
    exclude: np.ndarray,
    rows: np.ndarray,
    *,
    robust: bool = True,
    study: str = "",
) -> np.ndarray:
    """Returns ``S`` with every candidate's location removed in the tubes of ``rows``.

    A candidate's z becomes ``z - location - median``, the median taken over the
    candidates so that the null stays centered at 0 in the pooled null's units (no
    rescaling). The excluded (own) column, a non-finite z and every tube outside
    ``rows`` are returned as they are, bit for bit.
    """
    out = np.array(S, dtype=np.float64, copy=True)
    X = np.column_stack([np.ones(S.shape[1]), pcs])
    for i in np.flatnonzero(rows):
        use = _usable(S[i], int(exclude[i]))
        if not use.any():
            continue
        res = S[i, use] - _row_location(S[i, use], X[use], robust=robust, study=study, row=i)
        out[i, use] = res - np.median(res)
    return out


def ancestry_null(
    S: np.ndarray,
    dosages: np.ndarray,
    *,
    exclude: np.ndarray,
    rows: np.ndarray,
    k: int,
    study: str,
    robust: bool = True,
) -> np.ndarray:
    """Applies ``ancestry_null_from_pcs`` on the top ``k`` PCs of ``dosages``.

    The columns of ``dosages`` are the columns of ``S``, in order. The singular-value
    gap is logged.
    """
    pcs, gap = genotype_pcs(dosages, k, study=study)
    logger.info(
        "pQTL ancestry null (pcs): %s, %d of %d tubes, %d genotypes, k = %d, relative singular-value gap %.4f",
        study,
        int(np.sum(rows)),
        S.shape[0],
        S.shape[1],
        k,
        gap,
    )
    return ancestry_null_from_pcs(S, pcs, exclude, rows, robust=robust, study=study)


def ancestry_null_rows(m: PqtlMatrix, config, dropped: set[str]) -> np.ndarray:
    """Returns the matrix's z with the ancestry null at ``config``'s PCs and scope.

    The null uses ``pqtl_null_pcs`` components and ``pqtl_null_scope``. Every genotype
    column is returned, in the matrix's order; ``pqtl_calls`` drops the ``dropped``
    (duplicate-genome) columns after this. The principal components cover every genotype of the
    PSAM (dropped links stay in the null, as they do in the robust z): under
    ``pqtl_pcs_source = "projected"`` the first ``pqtl_null_pcs`` columns of
    ``m.pcs_mapped`` (checked by ``genome_tables.check_projected_pcs``), under
    ``"catalog"`` the top PCs of ``m.dosages_mapped`` (``genotype_pcs``, with the
    singular-value gap guard). A tube's own column is its label's genotype, dropped or
    not; the ``"ungenotyped"`` scope acts on the tubes whose label has no kept genotype
    (none, or one in ``dropped``, which ``pqtl_calls`` treats as ungenotyped),
    ``"all"`` on every tube.

    Raises:
        ValueError: If the panel has fewer than ``PCS_MIN_GENOTYPES`` genotypes, or the
            projected PCs are absent from the matrix or cannot carry the fit.
    """
    n_geno = len(m.mapped_usubjids)
    if n_geno < PCS_MIN_GENOTYPES:
        raise ValueError(
            f"{config.study_name}: pqtl.null_by_ancestry = 'pcs' has {n_geno} genotypes in the panel; it needs at least {PCS_MIN_GENOTYPES}"
        )
    column = {u: j for j, u in enumerate(m.mapped_usubjids)}
    exclude = np.array(
        [column.get(u, -1) if u is not None else -1 for u in m.usubjids], dtype=np.int64
    )
    if config.pqtl_null_scope == "all":
        rows = np.ones(len(exclude), dtype=bool)
    else:
        rows = np.array([j < 0 or m.mapped_usubjids[j] in dropped for j in exclude], dtype=bool)
    if config.pqtl_pcs_source == "catalog":
        return ancestry_null(
            m.S_mapped,
            m.dosages_mapped,
            exclude=exclude,
            rows=rows,
            k=config.pqtl_null_pcs,
            study=config.study_name,
        )
    if m.pcs_mapped is None:
        raise ValueError(
            f"{config.study_name}: pqtl_pcs_source = 'projected' needs pqtl_ancestry_pcs; "
            "the matrix has no projected PCs (no in-process fallback)"
        )
    pcs = check_projected_pcs(m.pcs_mapped, config.pqtl_null_pcs, study=config.study_name)
    logger.info(
        "pQTL ancestry null (projected pcs): %s, %d of %d tubes, %d genotypes, k = %d",
        config.study_name,
        int(np.sum(rows)),
        m.S_mapped.shape[0],
        n_geno,
        config.pqtl_null_pcs,
    )
    return ancestry_null_from_pcs(m.S_mapped, pcs, exclude, rows, study=config.study_name)


def pqtl_inputs(qc_instance, config) -> PqtlInputs:
    """Reads the pQTL stage's inputs (``PqtlInputs``) from ``qc_instance``.

    Reads the genotype directory's first PSAM (every IID is a USUBJID), relabels the
    genomes the config's ``dna_relink`` corrections move, and checks the overlap before
    any genotype is read. The KING table and the projected PCs the
    config names are read and matched to the PSAM here too, so a mismatched file stops
    the run before the catalog variants are extracted.

    Raises:
        FileNotFoundError: If the genotype directory or a named summary file does not
            exist.
        ValueError: If fewer than ``MIN_GENOTYPED_FRACTION`` of the labeled study
            subjects are in the PSAM, a PSAM IID repeats, the KING table names a
            genotype outside the PSAM, or the PC file misses a PSAM genotype or holds
            one outside it; or a ``dna_relink`` correction does not resolve
            (``dna_relink.relink_map``).
    """
    require_pqtl_inputs(config)
    sid = qc_instance.sample_id_col
    X, samples_df, protein_ids, residuals = matrix_for(qc_instance)
    sample_ids = samples_df[sid].cast(pl.String).to_list()

    # PSAM overlap, before the expensive extraction.
    first_prefix = next(
        iter(pgen_prefixes(config.pqtl_genotype_dir, config.pqtl_pgen_pattern).values())
    )[0]
    psam_iids = read_psam(first_prefix)["IID"].to_list()
    labeled = {u for u in samples_df["USUBJID"].to_list() if u is not None}
    adsl = qc_instance._ensure_lazy(qc_instance.adsl).collect()
    # A dna_relink correction files a genome under its owner: the label changes, the
    # genotype column does not (dna_relink.py).
    relinks = relink_map(
        config.corrections,
        psam_iids,
        labeled | set(adsl[qc_instance.usubjid_col].cast(pl.String).to_list()),
    )
    labels = relabel(_genotype_labels(psam_iids, labeled), relinks)
    if relinks:
        logger.info("pQTL concordance: %d genome(s) relinked by dna_relink corrections", len(relinks))
    _check_genotype_overlap(labeled, labels, psam_iids)
    # The KING table and the PC file are keyed by PSAM IID; their rows follow the PSAM
    # order, which is the order of ``labels``.
    kinship = (
        None
        if config.pqtl_kinship_table is None
        else kinship_from_table(
            read_king_table(config.pqtl_kinship_table),
            psam_iids,
            source=config.pqtl_kinship_table,
        )
    )
    pcs = (
        None
        if config.pqtl_ancestry_pcs is None
        else pcs_for(
            read_ancestry_pcs(config.pqtl_ancestry_pcs),
            psam_iids,
            source=config.pqtl_ancestry_pcs,
        )
    )

    samples_df = samples_df.with_columns(pl.Series("fit", _build_fit_mask(qc_instance, sample_ids)))
    sex_preds = qc_instance.sex_oof_predictions
    if sex_preds is not None:
        sex_preds = sex_preds.select(
            pl.col(sid).cast(pl.String).alias("SampleID"),
            pl.col("Predicted_Male_Probability").alias("p_male"),
        )
    return PqtlInputs(
        X=X,
        samples=samples_df,
        protein_ids=protein_ids,
        adsl=adsl,
        plate_col=qc_instance.plate_id_col,
        sample_id_col=sid,
        analyte_metadata=qc_instance._ensure_lazy(qc_instance.analyte_metadata)
        .select(qc_instance.protein_id_col, qc_instance.uniprot_col)
        .collect(),
        protein_id_col=qc_instance.protein_id_col,
        uniprot_col=qc_instance.uniprot_col,
        psam_iids=psam_iids,
        labels=labels,
        sex_preds=sex_preds,
        platform_name=qc_instance.platform.name,
        residuals=residuals,
        kinship=kinship,
        pcs=pcs,
    )


def pqtl_catalog(inp: PqtlInputs, config) -> pl.DataFrame:
    """Loads the reference catalog of ``config``, remapped to the trial's protein IDs.

    Under ``pqtl_catalog_merge = "list_order"`` it carries ``CATALOG_RANK``.

    Raises:
        ValueError: If no catalog row maps to a trial protein.
    """
    catalog = remap_catalog_platform_ids(
        load_reference_catalog(
            config.pqtl_reference_catalog,
            platform=inp.platform_name,
            include_non_eur=config.pqtl_include_non_eur,
            build=config.pqtl_catalog_build,
            catalog_platform=config.pqtl_catalog_platform,
            with_rank=config.pqtl_catalog_merge == "list_order",
        ),
        inp.analyte_metadata,
        platform=inp.platform_name,
        protein_id_col=inp.protein_id_col,
        uniprot_col=inp.uniprot_col,
    )
    if catalog.height == 0:
        raise ValueError(
            f"pQTL concordance: no reference-catalog row maps to a trial "
            f"{inp.protein_id_col} ({config.pqtl_reference_catalog}); "
            "check the catalog's platform IDs and UniProt column"
        )
    return catalog


def pqtl_matrix(
    inp: PqtlInputs,
    config,
    genotypes: CatalogGenotypes,
    catalog: pl.DataFrame | None = None,
) -> PqtlMatrix:
    """Scores every sample against every genotype and summarizes the genotypes.

    Residualizes the identity matrix with the Lp keys (``LpParams.from_config``), then
    scores with the pQTL scoring keys of ``config``. The kinship is the KING table's
    (``inp.kinship``) under ``pqtl_kinship_source = "table"``, else computed from the
    catalog-variant dosages; the projected PCs (``inp.pcs``) ride along for the calls.

    Args:
        inp: The stage's inputs.
        config: The trial config.
        genotypes: Trial genotypes at the catalog's variants
            (``extract_catalog_genotypes``).
        catalog: The remapped catalog; ``pqtl_catalog`` of ``config`` when None.

    Raises:
        ValueError: If the genotypes' sample order differs from the first PSAM's, or
            ``pqtl_kinship_source = "table"`` and the inputs carry no KING table.
    """
    if config.pqtl_kinship_source == "table" and inp.kinship is None:
        raise ValueError(
            "pqtl_kinship_source = 'table' needs pqtl_kinship_table (no in-process fallback)"
        )
    if catalog is None:
        catalog = pqtl_catalog(inp, config)
    if genotypes.iids != inp.psam_iids:
        raise ValueError(
            "the first PSAM's sample order differs from the PGENs read for catalog variants; "
            "USUBJID labels would misalign"
        )
    labels = inp.labels
    samples_df = inp.samples
    sample_ids = samples_df[inp.sample_id_col].cast(pl.String).to_list()
    usubjids = samples_df["USUBJID"].to_numpy()
    labeled = {u for u in samples_df["USUBJID"].to_list() if u is not None}
    Z, keep = residualize_samples(
        inp.X,
        samples_df,
        inp.adsl,
        plate_col=inp.plate_col,
        params=LpParams.from_config(config),
        cache=inp.residuals,
    )
    protein_ids = [p for p, k in zip(inp.protein_ids, keep, strict=True) if k]
    # Genotypes and kinship come from every catalog row; list_order only narrows the
    # rows scored.
    if config.pqtl_catalog_merge == "list_order":
        catalog = merge_catalog_list_order(
            catalog,
            genotypes,
            protein_ids,
            nlp_recompute_floor=config.pqtl_nlp_recompute_floor,
            palindromic=config.pqtl_palindromic,
        )

    # Every genotype is labeled with its IID: it is in every sample's null, can be a
    # sample's own genotype and can be a tube's named donor.
    mapped_usubjids = list(labels)
    logger.info(
        "pQTL concordance: %d samples, %d of %d labeled subjects genotyped, "
        "%d genotyped participants with no proteomics, %d proteins, %d variants",
        len(sample_ids),
        len(labeled & set(mapped_usubjids)),
        len(labeled),
        len(set(mapped_usubjids) - labeled),
        len(protein_ids),
        len(genotypes.variant_ids),
    )
    kinship_mapped = (
        inp.kinship if config.pqtl_kinship_source == "table" else kinship_matrix(genotypes.dosages)
    )
    logger.info(
        "pQTL genotype summaries: kinship from %s, ancestry PCs %s",
        config.pqtl_kinship_table
        if config.pqtl_kinship_source == "table"
        else "the catalog variants",
        config.pqtl_ancestry_pcs if inp.pcs is not None else "not read",
    )
    S_all, mixture_z = _reference_pqtl_scores(
        Z,
        protein_ids,
        genotypes,
        catalog,
        usubjids,
        labels,
        weight_exponent=config.pqtl_weight_exponent,
        dosage_scaling=config.pqtl_dosage_scaling,
        maf_min=config.pqtl_maf_min,
        contribution_cap=config.pqtl_contribution_cap,
        nlp_recompute_floor=config.pqtl_nlp_recompute_floor,
        palindromic=config.pqtl_palindromic,
        with_mixture=True,
        same_genome=np.nan_to_num(kinship_mapped, nan=-1.0) >= KING_DUPLICATE,
    )
    labelled_rows = np.array([u is not None for u in usubjids])
    links = link_evidence(
        S_all[labelled_rows], usubjids[labelled_rows].astype(str), mapped_usubjids
    )
    return PqtlMatrix(
        S_mapped=S_all,
        mapped_usubjids=mapped_usubjids,
        kinship_mapped=kinship_mapped,
        dosages_mapped=genotypes.dosages,
        genetic_sex_mapped=genotypes.genetic_sex.select(
            pl.Series("USUBJID", mapped_usubjids, dtype=pl.String), *SEX_COLUMNS
        ),
        links=links,
        usubjids=usubjids,
        sample_ids=sample_ids,
        fit=samples_df["fit"].to_numpy(),
        platform_name=inp.platform_name,
        mixture_z=mixture_z,
        pcs_mapped=inp.pcs,
    )


def pqtl_calls(
    m: PqtlMatrix, config, sex_preds: pl.DataFrame | None
) -> tuple[pl.DataFrame, PqtlEvidence]:
    """Makes the per-tube calls and the evidence identity reads from a ``PqtlMatrix``.

    The steps are the duplicate-genome step (``pqtl_duplicate_kinship``,
    ``pqtl_match_floor``), the relative mask (``pqtl_relative_kinship``),
    ``pqtl.extract_concordance`` (``pqtl_best_min``), ``pqtl.classify_pqtl`` (with
    ``pqtl_relative_margin`` and ``pqtl_relative_name_margin``, whose helper columns
    ``pqtl_best_rel_z`` and ``pqtl_best_rel_USUBJID`` are dropped from the output), the
    matrix's ``pqtl_mixture_z`` (null for a tube whose link the duplicate step dropped
    or held), the DNA-sex check (``identity_sex_prob``) and the DNA-link table. Only the
    config fields in ``PQTL_CALL_KEYS`` are read, so the calls of other key values can
    be made on a saved matrix.

    With ``pqtl_null_by_ancestry = "pcs"`` the candidates' z of the scoped
    tubes (``ancestry_null_rows``) is replaced by the ancestry-null z before the calls,
    evidence included; the duplicate step, the own z and the mixture score read the
    matrix as saved.

    Returns:
        ``(scores, evidence)``: per-tube calls (``schemas.PQTL_SCORES_SCHEMA``) sorted
        by ``SampleID``, and the ``PqtlEvidence``.
    """
    usubjids = m.usubjids
    sample_ids = m.sample_ids
    mapped_usubjids = m.mapped_usubjids
    kinship_mapped = m.kinship_mapped
    fit_mask = m.fit
    # Duplicate genomes: keep the link whose proteome matches, drop the others.
    # A dropped genotype leaves the mapped list, so its subject's tubes are scored as
    # ungenotyped and it is never a named donor; it stays in the null. A member with no
    # proteomics has no tube, so it never matches and is dropped.
    duplicates = duplicate_genome_links(
        m.S_mapped,
        usubjids,
        fit_mask,
        mapped_usubjids,
        kinship_mapped,
        duplicate_kinship=config.pqtl_duplicate_kinship,
        match_floor=config.pqtl_match_floor,
    )
    dropped = set(duplicates.filter(pl.col("link_status") == "dropped")["USUBJID"])
    held = set(duplicates.filter(pl.col("link_status") == "hold")["USUBJID"])
    if dropped or held:
        logger.info(
            "pQTL duplicate genomes: %d participants in groups, %d links dropped, %d held",
            duplicates.height,
            len(dropped),
            len(held),
        )
    keep = [k for k, u in enumerate(mapped_usubjids) if u not in dropped]
    mapped_usubjids = [mapped_usubjids[k] for k in keep]
    # The ancestry null is applied to the full matrix, after the duplicate step
    # read the untransformed own z and before the dropped columns go, so the PCs and the
    # columns stay in the matrix's order.
    S = (
        ancestry_null_rows(m, config, dropped)
        if config.pqtl_null_by_ancestry == "pcs"
        else m.S_mapped
    )[:, keep]
    kinship = kinship_mapped[np.ix_(keep, keep)]
    dosages = m.dosages_mapped[:, keep]

    # Relatives of each sample's labeled genotype; an ungenotyped label has none.
    column = {u: j for j, u in enumerate(mapped_usubjids)}
    own = np.array([column.get(u, -1) for u in usubjids], dtype=np.int64)
    relative = np.zeros(S.shape, dtype=bool)
    has_own = own >= 0
    relative[has_own] = (
        np.nan_to_num(kinship[own[has_own]], nan=-1.0) >= config.pqtl_relative_kinship
    )
    scores = extract_concordance(
        S,
        usubjids,
        mapped_usubjids,
        best_min=config.pqtl_best_min,
        sample_ids=sample_ids,
        relative=relative,
        best_relative=config.pqtl_relative_margin is not None
        or config.pqtl_relative_name_margin is not None,
    )
    fit_of = dict(zip(sample_ids, fit_mask.tolist(), strict=True))

    def fit_of_scores(frame: pl.DataFrame) -> np.ndarray:
        return np.array([fit_of[s] for s in frame["SampleID"].to_list()], dtype=bool)

    scores = classify_pqtl(scores, kinship, mapped_usubjids, fit_of_scores(scores), config).join(
        pl.DataFrame(
            {
                "SampleID": sample_ids,
                "pqtl_mixture_z": np.full(len(sample_ids), np.nan)
                if m.mixture_z is None
                else m.mixture_z,
            },
            schema={"SampleID": pl.String, "pqtl_mixture_z": pl.Float64},
        ).with_columns(pl.col("pqtl_mixture_z").fill_nan(None)),
        on="SampleID",
        how="left",
        maintain_order="left",
    )
    # A held link (both duplicates match) is no evidence either way; a dropped link has
    # no genotype, and its tubes must not name the duplicate that was kept.
    no_evidence = pl.col("USUBJID").is_in(list(held | dropped))
    scores = scores.with_columns(
        pl.when(no_evidence).then(pl.lit("none")).otherwise(pl.col("pqtl_call")).alias("pqtl_call"),
        *(
            pl.when(no_evidence).then(None).otherwise(pl.col(c)).alias(c)
            for c in (
                "pqtl_best_z",
                "pqtl_best_USUBJID",
                "pqtl_best_nonrel_z",
                "pqtl_best_nonrel_USUBJID",
                "pqtl_best_kinship",
                "pqtl_relative_USUBJID",
                "pqtl_mixture_z",
            )
        ),
    )

    genetic_sex = m.genetic_sex_mapped.filter(~pl.col("USUBJID").is_in(list(dropped)))
    scores = (
        verify_dna_link(scores, sex_preds, genetic_sex, config.identity_sex_prob)
        .sort("SampleID")
        .with_columns(
            pl.lit(config.study_name).alias("trial"),
            pl.lit(m.platform_name).alias("platform"),
        )
        .select([pl.col(c).cast(t) for c, t in PQTL_SCORES_SCHEMA.items()])
    )
    dna_link = dna_link_table(
        scores,
        fit_of_scores(scores),
        m.links,
        duplicates,
        m.genetic_sex_mapped,
        trial=config.study_name,
        platform=m.platform_name,
    )

    evidence = PqtlEvidence(
        S=S,
        sample_ids=sample_ids,
        geno_usubjids=mapped_usubjids,
        dosages=dosages,
        kinship=kinship,
        genetic_sex=genetic_sex,
        dna_link=dna_link,
    )
    return scores, evidence


def pqtl_concordance_for_qc(qc_instance, config) -> tuple[pl.DataFrame, PqtlEvidence]:
    """Computes reference pQTL concordance for every study sample of a QC instance.

    Builds the identity matrix, residualizes it (arm x visit and plate, as for Lp),
    remaps the catalog to the trial's protein IDs, reads catalog variants from
    ``config.pqtl_genotype_dir``, scores every sample against every genotype and
    cross-checks genetic against predicted sex per sample: ``pqtl_inputs``,
    ``pqtl_catalog``, ``extract_catalog_genotypes``, ``pqtl_matrix``, ``pqtl_calls``.

    Genotype IIDs are USUBJIDs, and every PSAM IID is labeled with itself: a
    genotyped participant with no proteomics is in every sample's null distribution, can
    be the best-other donor of a tube and is in the returned evidence. A sample's own
    genotype is never its own other, and relatives of its labeled genotype (KING
    kinship of at least ``pqtl_relative_kinship``) are never its named non-relative
    donor.

    Returns:
        ``(scores, evidence)``: per-sample concordance sorted by ``SampleID``
        (``schemas.PQTL_SCORES_SCHEMA``), and the evidence behind it, over every
        genotype of the PSAM but the ones the duplicate-genome step dropped.

    Raises:
        ValueError: If fewer than ``MIN_GENOTYPED_FRACTION`` of the study samples'
            labeled subjects are in the genotype PSAM.
    """
    inp = pqtl_inputs(qc_instance, config)
    catalog = pqtl_catalog(inp, config)
    geno = extract_catalog_genotypes(
        catalog,
        config.pqtl_genotype_dir,
        pgen_pattern=config.pqtl_pgen_pattern,
        sex_thresholds=config.sex_call_thresholds(),
    )
    m = pqtl_matrix(inp, config, geno, catalog)
    return pqtl_calls(m, config, inp.sex_preds)
