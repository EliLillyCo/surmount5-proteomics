"""Parsers that turn the published cis-pQTL tables into ``CATALOG_SCHEMA`` frames.

Each parser writes the frame whose content digest ``pqtl_catalogs.CATALOG_FRAME_SHA256``
pins. UKB-PPP and deCODE are parsed by ``analysis.qc.pqtl_reference.parse_ukbppp`` and ``parse_decode``. Sources, digests and the
build order are in ``analysis.external.pqtl_catalogs``.

Every parser keeps the cis rows only. ``a1`` is the allele ``beta`` refers to and ``a2``
the other one (null where the table names none). Each publication writes very small
p-values its own way (a float, a scientific-notation string past float64's range, or
-log10(p) directly), so each parser keeps its own conversion to ``neg_log10p``.
"""

from __future__ import annotations

import logging
import math
import re
from decimal import Decimal
from pathlib import Path

import openpyxl
import polars as pl

from analysis.external.liftover import Liftover, RsidLocator, lift
from analysis.qc.pqtl import normalize_chrom
from analysis.qc.pqtl_reference import CATALOG_SCHEMA, _chrom_str, _safe_str

logger = logging.getLogger(__name__)


def _columns() -> dict[str, list]:
    return {k: [] for k in CATALOG_SCHEMA}


def _frame(cols: dict[str, list]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema=CATALOG_SCHEMA)


def _nlp_split_exponent(pval) -> float | None:
    """-log10(p) of a float, or of a scientific-notation string split at its exponent."""
    if pval is None:
        return None
    if isinstance(pval, (int, float)):
        p = float(pval)
        return -math.log10(p) if p > 0 else None
    return _nlp_text(str(pval))


def _nlp_text(text: str) -> float | None:
    s = text.strip()
    if not s:
        return None
    if "e" in s or "E" in s:
        mantissa_str, exp_str = s.lower().split("e")
        mantissa = float(mantissa_str)
        if mantissa <= 0:
            return None
        return -(math.log10(mantissa) + int(exp_str))
    p = float(s)
    return -math.log10(p) if p > 0 else None


# ── Jackson Heart Study (Olink) ─────────────────────────────────────────────


def parse_jhs_mesa(path: str | Path) -> pl.DataFrame:
    """Jackson Heart Study cis-pQTLs (Tahir et al. 2024, JCI, sheets ST2 and ST3).

    ST3 holds the sentinel pQTLs discovered in JHS (African American, hg38) with MESA
    replication columns that are not used; ST2 maps each OlinkID to its UniProt. Rows with
    ``CisTrans == "CIS"`` are kept, with the JHS discovery beta, SE and p; ``a1``/``a2`` are
    JHS A1/A2 and ``ancestry`` is ``"AFR"``. ST3 columns used: 0 OlinkID, 1 chr, 3 pos,
    4 A1, 5 A2, 6 AF1, 7 BETA, 8 SE, 9 P, 11 Assay, 16 CisTrans, 27 rsID.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    oid_to_uniprot: dict[str, str] = {}
    for row in wb["ST2"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None and row[2] is not None:
            oid_to_uniprot[str(row[0]).strip()] = str(row[2]).strip()
    cols = _columns()
    for row in wb["ST3"].iter_rows(min_row=2, values_only=True):
        if _safe_str(row[16]) != "CIS":
            continue
        oid = _safe_str(row[0])
        af1 = float(row[6]) if row[6] is not None else None
        cols["rsid"].append(_safe_str(row[27]))
        cols["chrom"].append(_chrom_str(row[1]))
        cols["pos"].append(int(row[3]) if row[3] is not None else None)
        cols["protein_id"].append(_safe_str(row[11]))
        cols["uniprot"].append(oid_to_uniprot.get(oid) if oid else None)
        cols["platform_id"].append(oid)
        cols["beta"].append(float(row[7]) if row[7] is not None else None)
        cols["se"].append(float(row[8]) if row[8] is not None else None)
        cols["neg_log10p"].append(_nlp_split_exponent(row[9]))
        cols["maf"].append(min(af1, 1.0 - af1) if af1 is not None else None)
        cols["eaf"].append(af1)
        cols["a1"].append(_safe_str(row[4]))
        cols["a2"].append(_safe_str(row[5]))
        cols["ancestry"].append("AFR")
        cols["source"].append("jhs_mesa")
    wb.close()
    return _frame(cols)


# ── Genes & Health (Olink Explore HT and SomaScan 11K) ──────────────────────

GH_PLATFORMS = {"Olink": ("ST2", "genes_and_health_olink"), "SomaLogic": ("ST3", "genes_and_health_soma")}
"""ST14 ``Platform`` value -> (lookup sheet, ``source``)."""


def _gh_lookup(ws, platform: str) -> dict[str, tuple[str | None, str | None, str | None]]:
    """``Affinity ID`` -> (platform_id, gene, UniProt) from ST2 (Olink) or ST3 (SomaLogic).

    Both sheets have their header on row 4. ST2: 0 Olink ID, 1 OlinkID, 2 Assay, 3 UniProt.
    ST3: 0 SomaLogic ID, 1 SeqId, 5 UniProt, 7 EntrezGeneSymbol.
    """
    id_col, gene_col, uniprot_col = (1, 2, 3) if platform == "Olink" else (1, 7, 5)
    lookup: dict[str, tuple[str | None, str | None, str | None]] = {}
    for row in ws.iter_rows(min_row=5, values_only=True):
        if row[0] is None:
            continue
        lookup[str(row[0])] = tuple(
            None if row[i] is None else str(row[i]) for i in (id_col, gene_col, uniprot_col)
        )
    return lookup


def _gh_check_variant_id(row) -> None:
    """Raises unless ``ID`` (``chrom_pos_OA_EA``) matches chrom, pos, OA and EA."""
    parts = str(row[4]).split("_")
    if len(parts) != 4:
        raise ValueError(f"ID {row[4]!r} is not chrom_pos_OA_EA")
    chrom, pos, oa, ea = parts
    if (
        normalize_chrom(chrom) != normalize_chrom(str(row[5]))
        or int(pos) != int(row[6])
        or oa != row[7]
        or ea != row[8]
    ):
        raise ValueError(
            f"ID {row[4]!r} disagrees with chrom {row[5]!r}, pos {row[6]!r}, OA {row[7]!r}, EA {row[8]!r}"
        )


def parse_genes_health(path: str | Path, platform: str) -> pl.DataFrame:
    """Genes & Health cis-pQTLs of one affinity platform (Pietzner 2026, Nat Genet, ST14).

    ST14 holds the variant-level statistics of the Olink and SomaLogic pQTLs whose targets
    Seer also measured (hg38, British South Asian, ``ancestry`` ``"SAS"``). Column 0 is the
    platform, 1 the ``Affinity ID``, 3 cis/trans, 4 ``ID`` (``chrom_pos_OA_EA``, checked
    against 5 chrom, 6 position, 7 OA, 8 EA), 9 EAF, 11 BETA, 12 SE, 13 LOG10P. ``a1`` is EA
    and ``a2`` OA. ``platform_id`` is the Explore HT OlinkID or the ``xxxxx-y`` SeqId from
    the lookup sheet; a row it cannot resolve is dropped and counted in the log. A pQTL
    matched to several Seer protein groups repeats with identical statistics, so only
    distinct rows are kept. ``rsid`` is null.

    Args:
        path: The supplement workbook.
        platform: ``"Olink"`` or ``"SomaLogic"``.

    Raises:
        ValueError: For another platform, or a row whose ``ID`` disagrees with its columns.
    """
    if platform not in GH_PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}")
    sheet, source = GH_PLATFORMS[platform]
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    lookup = _gh_lookup(wb[sheet], platform)
    cols = _columns()
    n_unresolved = 0
    for row in wb["ST14"].iter_rows(min_row=5, values_only=True):
        if row[0] != platform or row[3] != "cis" or row[1] is None:
            continue
        _gh_check_variant_id(row)
        resolved = lookup.get(str(row[1]))
        if resolved is None or resolved[0] is None:
            n_unresolved += 1
            continue
        platform_id, gene, uniprot = resolved
        eaf = float(row[9]) if row[9] is not None else None
        cols["rsid"].append(None)
        cols["chrom"].append(_chrom_str(row[5]))
        cols["pos"].append(int(row[6]) if row[6] is not None else None)
        cols["protein_id"].append(gene)
        cols["uniprot"].append(uniprot)
        cols["platform_id"].append(platform_id)
        cols["beta"].append(float(row[11]) if row[11] is not None else None)
        cols["se"].append(float(row[12]) if row[12] is not None else None)
        cols["neg_log10p"].append(float(row[13]) if row[13] is not None else None)
        cols["maf"].append(min(eaf, 1.0 - eaf) if eaf is not None else None)
        cols["eaf"].append(eaf)
        cols["a1"].append(_safe_str(row[8]))
        cols["a2"].append(_safe_str(row[7]))
        cols["ancestry"].append("SAS")
        cols["source"].append(source)
    wb.close()
    df = _frame(cols).unique(maintain_order=True)
    logger.info(
        "Genes & Health %s: %d cis-pQTLs (%d unresolved against %s)", platform, df.height, n_unresolved, sheet
    )
    return df


# ── MESA (Olink Explore 3072 and SomaScan 7k, Zenodo 17642645) ──────────────

MESA_MAX_P = 5e-8
"""MESA keeps one lead cis variant per protein below this p, the other catalogs' threshold."""

_MESA_COLUMNS = {
    "chromosome": pl.String,
    "base_pair_location": pl.Int64,
    "effect_allele": pl.String,
    "other_allele": pl.String,
    "beta": pl.Float64,
    "standard_error": pl.Float64,
    "effect_allele_frequency": pl.Float64,
    "p_value": pl.Float64,
    "variant_id": pl.String,
    "phenotype_id": pl.String,
    "cis_trans": pl.String,
}
_MESA_OLINK_ID = re.compile(r"(OID\d+)_(.+)")
_MESA_SOMA_ID = re.compile(r"seq\.(\d+)\.(\d+)_(.+)")


def uniprot_by_platform_id(catalogs: list[pl.DataFrame]) -> pl.DataFrame:
    """``platform_id`` -> ``uniprot`` from parsed catalogs; the first catalog naming an ID wins."""
    frames = [c.select("platform_id", "uniprot").drop_nulls() for c in catalogs]
    return pl.concat(frames).unique(subset="platform_id", keep="first", maintain_order=True)


def _mesa_split_phenotype(platform: str, phenotype_ids: pl.Series) -> pl.DataFrame:
    """Splits ``OID<n>_<gene>`` or ``seq.<a>.<b>_<gene>`` into ``platform_id`` and gene.

    Raises:
        ValueError: If a ``phenotype_id`` does not have the platform's form.
    """
    pattern = _MESA_OLINK_ID if platform == "olink" else _MESA_SOMA_ID
    ids = phenotype_ids.to_list()
    parts = []
    for pid in ids:
        m = pattern.fullmatch(pid)
        if m is None:
            raise ValueError(f"unexpected {platform} phenotype_id {pid!r}")
        g = m.groups()
        parts.append((g[0], g[1]) if platform == "olink" else (f"{g[0]}_{g[1]}", g[2]))
    return pl.DataFrame(
        {"phenotype_id": ids, "platform_id": [p[0] for p in parts], "protein_id": [p[1] for p in parts]},
        schema={"phenotype_id": pl.String, "platform_id": pl.String, "protein_id": pl.String},
    )


def parse_mesa(
    path: str | Path, *, platform: str, source: str, uniprot: pl.DataFrame, max_p: float = MESA_MAX_P
) -> pl.DataFrame:
    """The lead cis variant of each protein in a MESA ``*_cis_trans_pQTL_combined.tsv.gz``.

    Nicholas et al. 2026 (Nat Commun) mapped every cis variant within 1 Mb of the TSS
    (EAF >= 0.01), ancestry-pooled (hg38). Among the cis rows of
    a ``phenotype_id`` with p below ``max_p``, the lead is the smallest p (ties: larger
    |beta/SE|, then ``variant_id``); a protein with none gets no row. ``platform_id`` is the
    Explore 3072 OlinkID (``olink``) or the SeqId as ``10000_28`` (``soma``); ``protein_id``
    the gene in ``phenotype_id``; ``uniprot`` comes by ``platform_id`` from ``uniprot``
    (null when absent). ``ancestry`` is ``"MULTI"`` (pooled).

    Args:
        path: The ``.tsv.gz``.
        platform: ``"olink"`` or ``"soma"``.
        source: Value of the ``source`` column.
        uniprot: ``platform_id`` and ``uniprot`` (``uniprot_by_platform_id``).
        max_p: Keep cis rows with p-value below this.
    """
    cis = pl.scan_csv(str(path), separator="\t", schema_overrides=_MESA_COLUMNS).filter(
        (pl.col("cis_trans") == "cis") & (pl.col("p_value") < max_p)
    )
    leads = (
        cis.sort(
            "phenotype_id",
            "p_value",
            (pl.col("beta") / pl.col("standard_error")).abs(),
            "variant_id",
            descending=[False, False, True, False],
        )
        .unique(subset="phenotype_id", keep="first", maintain_order=True)
        .collect(engine="streaming")
        .sort("phenotype_id")
    )
    out = (
        leads.join(
            _mesa_split_phenotype(platform, leads["phenotype_id"]), on="phenotype_id", how="left", maintain_order="left"
        )
        .join(uniprot, on="platform_id", how="left", maintain_order="left")
        .select(
            pl.lit(None, dtype=pl.String).alias("rsid"),
            pl.col("chromosome").alias("chrom"),
            pl.col("base_pair_location").alias("pos"),
            "protein_id",
            "uniprot",
            "platform_id",
            "beta",
            pl.col("standard_error").alias("se"),
            pl.when(pl.col("p_value") > 0).then(-pl.col("p_value").log10()).otherwise(None).alias("neg_log10p"),
            pl.min_horizontal("effect_allele_frequency", 1.0 - pl.col("effect_allele_frequency")).alias("maf"),
            pl.col("effect_allele").alias("a1"),
            pl.col("other_allele").alias("a2"),
            pl.lit("MULTI").alias("ancestry"),
            pl.lit(source).alias("source"),
            pl.col("effect_allele_frequency").alias("eaf"),
        )
    )
    return out.cast(CATALOG_SCHEMA)


# ── China Kadoorie Biobank, Nat Commun 2025 (Olink and SomaScan sheets) ─────


def _ckb_rows(path: str | Path, sheet: str):
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    rows = [r for r in wb[sheet].iter_rows(min_row=2, values_only=True) if r[0] is not None and r[13] == "cis"]
    wb.close()
    return rows


def _ckb_append(cols: dict[str, list], row, chrom: str, gene, uniprot, platform_id, source: str) -> None:
    """One CKB sheet row: 3 POS, 6 effect, 7 non-effect allele, 8 EAF, 10 BETA, 11 SE, 12 P."""
    eaf = float(row[8])
    p = float(row[12])
    cols["rsid"].append(None)
    cols["chrom"].append(chrom)
    cols["pos"].append(int(row[3]))
    cols["protein_id"].append(gene)
    cols["uniprot"].append(uniprot)
    cols["platform_id"].append(platform_id)
    cols["beta"].append(float(row[10]))
    cols["se"].append(float(row[11]))
    cols["neg_log10p"].append(-math.log10(p) if p > 0 else None)
    cols["maf"].append(min(eaf, 1.0 - eaf))
    cols["eaf"].append(eaf)
    cols["a1"].append(str(row[6]).strip())
    cols["a2"].append(str(row[7]).strip())
    cols["ancestry"].append("EAS")
    cols["source"].append(source)


def ckb_gene_annotation(catalogs: list[pl.DataFrame]) -> dict[str, tuple[str, str]]:
    """Gene symbol -> (UniProt, OlinkID): first catalog naming it, then the smallest OlinkID."""
    out: dict[str, tuple[str, str]] = {}
    for cat in catalogs:
        firsts = (
            cat.select("protein_id", "uniprot", "platform_id")
            .drop_nulls()
            .sort("protein_id", "platform_id", "uniprot")
            .unique(subset="protein_id", keep="first", maintain_order=True)
        )
        for gene, uniprot, olink_id in firsts.iter_rows():
            out.setdefault(gene, (uniprot, olink_id))
    return out


def parse_ckb_olink(path: str | Path, annotation: dict[str, tuple[str, str]]) -> pl.DataFrame:
    """CKB Olink Explore 3072 sentinel cis-pQTLs (Nat Commun 2025, Supplementary Data 4, sheet ``Olink``).

    GRCh38 coordinates; ``ancestry`` is ``"EAS"``. The sheet names each assay by gene
    symbol only (column 0); ``uniprot`` and ``platform_id`` come by symbol from
    ``annotation`` (``ckb_gene_annotation``), and a row whose symbol it lacks is dropped and
    counted in the log. Every cis sentinel is kept; chromosome 23 is X. A non-positive ``P`` gives
    a null ``neg_log10p``; the loader derives it from beta and SE.
    """
    cols = _columns()
    rows = _ckb_rows(path, "Olink")
    n_unresolved = 0
    for row in rows:
        gene = str(row[0]).strip()
        if gene not in annotation:
            n_unresolved += 1
            continue
        uniprot, olink_id = annotation[gene]
        chrom = str(row[2]).strip().removeprefix("chr")
        _ckb_append(cols, row, "X" if chrom == "23" else chrom, gene, uniprot, olink_id, "ckb_olink")
    logger.info("CKB Olink: %d cis rows, %d dropped for unannotated gene symbols", len(rows), n_unresolved)
    return _frame(cols)


_CKB_SOMASCAN_ID = re.compile(r"seq\.(\d+)\.(\d+)")


def _ckb_seqid(somascan_id: str) -> str:
    """``seq.10046.55`` -> ``10046_55``.

    Raises:
        ValueError: For an ID not of the form ``seq.<analyte>.<revision>``.
    """
    m = _CKB_SOMASCAN_ID.fullmatch(somascan_id)
    if m is None:
        raise ValueError(f"unexpected SomaScan ID {somascan_id!r}; expected seq.<analyte>.<revision>")
    return f"{m[1]}_{m[2]}"


def ckb_seqid_annotation(catalogs: list[pl.DataFrame]) -> dict[str, tuple[str, str | None]]:
    """SeqId -> (gene symbol, UniProt); the first catalog naming a SeqId wins."""
    out: dict[str, tuple[str, str | None]] = {}
    for cat in catalogs:
        firsts = cat.select("platform_id", "protein_id", "uniprot").unique(
            subset="platform_id", keep="first", maintain_order=True
        )
        for seqid, gene, uniprot in firsts.iter_rows():
            out.setdefault(seqid, (gene, uniprot))
    return out


def parse_ckb_soma(path: str | Path, annotation: dict[str, tuple[str, str | None]]) -> pl.DataFrame:
    """CKB SomaScan v4.1 sentinel cis-pQTLs (Nat Commun 2025, Supplementary Data 4, sheet ``SomaScan``).

    Same cohort and layout as ``parse_ckb_olink``. ``platform_id`` is the SeqId as
    ``10046_55`` (from ``seq.10046.55``); ``protein_id`` and ``uniprot`` come by SeqId from
    ``annotation`` (``ckb_seqid_annotation``), null when absent.

    Raises:
        ValueError: For an ID not of the form ``seq.<analyte>.<revision>``.
    """
    cols = _columns()
    for row in _ckb_rows(path, "SomaScan"):
        seqid = _ckb_seqid(str(row[0]).strip())
        gene, uniprot = annotation.get(seqid, (None, None))
        chrom = str(int(row[2]))
        _ckb_append(cols, row, "X" if chrom == "23" else chrom, gene, uniprot, seqid, "ckb_soma")
    return _frame(cols)


# ── Olink Explore 3072 Fixed LOD file (a UniProt source for MESA Olink only) ──


def parse_olink_3072_lod(path: str | Path) -> pl.DataFrame:
    """``platform_id`` (OlinkID) -> ``uniprot`` from Olink's public Explore 3072 Fixed LOD file.

    Not a QC catalog: ``build`` uses only its (OlinkID, UniProt) pairs, as one of the UniProt
    sources of the MESA Olink catalog. The file is semicolon-separated with one row per assay
    per reagent lot (columns ``OlinkID``, ``AssayType``, ``UniProt``, ``Assay``, ``Panel``, ...);
    every lot of an assay must name the same UniProt.

    Raises:
        ValueError: If an OlinkID has more than one UniProt across its rows.
    """
    pairs = (
        pl.read_csv(path, separator=";", infer_schema=False, columns=["OlinkID", "UniProt"])
        .select(pl.col("OlinkID").alias("platform_id"), pl.col("UniProt").alias("uniprot"))
        .unique(maintain_order=True)
    )
    multi = pairs.filter(pl.col("platform_id").is_duplicated())["platform_id"].unique(maintain_order=True)
    if multi.len():
        raise ValueError(f"OlinkIDs with more than one UniProt in {Path(path).name}: {multi.to_list()}")
    return pairs


# ── ARIC (SomaScan) ─────────────────────────────────────────────────────────

ARIC_SHEETS = (("ST3.1--significant SOMAmer (EA)", "EA"), ("ST3.2--significant SOMAmer (AA)", "AA"))


def parse_aric(path: str | Path) -> pl.DataFrame:
    """ARIC sentinel cis-pQTLs at FDR < 5% (Zhang et al. 2022, Nat Genet, ST3.1 EA and ST3.2 AA).

    Each sheet has a title row, a header row and a subheader row; data from row 4, then
    blank and footnote rows (skipped: the first cell is not a ``SeqId_`` ID). Columns:
    0 SOMAmer ID (``SeqId_14151_4`` -> ``platform_id`` ``14151_4``), 3 UniProt, 4 Entrez
    gene symbol, 5 Chr, 8 sentinel rsID, 9 Pos (hg38), 10 tested (effect) allele, 11 its
    frequency, 12 Beta, 13 p-value. SE and the other allele are written as null. Every
    p-value is parsed as text, split at its exponent. Rows with neither a p-value nor an SE are dropped.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    cols = _columns()
    for sheet, ancestry in ARIC_SHEETS:
        for row in wb[sheet].iter_rows(min_row=4, values_only=True):
            if row[0] is None or not str(row[0]).startswith("SeqId_"):
                continue
            af = float(row[11]) if row[11] is not None else None
            cols["rsid"].append(_safe_str(row[8]))
            cols["chrom"].append(_chrom_str(row[5]))
            cols["pos"].append(int(row[9]) if row[9] is not None else None)
            cols["protein_id"].append(_safe_str(row[4]))
            cols["uniprot"].append(_safe_str(row[3]))
            cols["platform_id"].append(str(row[0]).strip()[len("SeqId_") :])
            cols["beta"].append(float(row[12]) if row[12] is not None else None)
            cols["se"].append(None)
            cols["neg_log10p"].append(None if row[13] is None else _nlp_text(str(row[13])))
            cols["maf"].append(min(af, 1.0 - af) if af is not None else None)
            cols["eaf"].append(af)
            cols["a1"].append(_safe_str(row[10]))
            cols["a2"].append(None)
            cols["ancestry"].append(ancestry)
            cols["source"].append("aric")
    wb.close()
    return _frame(cols).filter(~(pl.col("neg_log10p").is_null() & pl.col("se").is_null()))


# ── Fenland (SomaScan v4, hg19) ─────────────────────────────────────────────


def _strip_quote(val) -> str | None:
    """Strips Excel's leading apostrophe text escape."""
    if val is None:
        return None
    s = str(val).strip()
    return s.lstrip("'") if s else None


def _nlp_fenland(raw) -> float | None:
    """-log10(p) through ``Decimal`` for floats and text alike (numeric or text input)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return -float(Decimal(raw).log10()) if raw > 0 else None
    s = str(raw).strip()
    if not s:
        return None
    d = Decimal(s)
    return -float(d.log10()) if d > 0 else None


def parse_fenland(path: str | Path) -> pl.DataFrame:
    """Fenland lead cis-pQTLs (Pietzner et al. 2021, Science, sheet ST2), hg19 positions.

    Rows with ``cis.trans == "cis"`` and
    ``signal.locus == 1`` (the lead signal) are kept. Columns: 1 SomaScan.id
    (``SeqId_14151_4`` -> ``platform_id`` ``14151_4``), 4 signal.locus, 6 rsID, 7 cis.trans,
    8 Chr, 9 Position, 10 EA, 11 NEA, 12 EAF, 13 Effect, 14 SE, 15 P.value, 18 HGNC symbol;
    a leading ``'`` (Excel text escape) is stripped. ``uniprot`` is null (not in the sheet).
    ``fenland_liftover`` lifts the result to hg38.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    cols = _columns()
    n_cis = 0
    for row in wb["ST2"].iter_rows(min_row=2, values_only=True):
        if row[7] != "cis":
            continue
        n_cis += 1
        if row[4] != 1:
            continue
        eaf = float(row[12]) if row[12] is not None else None
        cols["rsid"].append(_strip_quote(row[6]))
        cols["chrom"].append(_chrom_str(row[8]))
        cols["pos"].append(int(row[9]) if row[9] is not None else None)
        cols["protein_id"].append(_strip_quote(row[18]))
        cols["uniprot"].append(None)
        cols["platform_id"].append(str(row[1]).replace("SeqId_", "") if row[1] is not None else None)
        cols["beta"].append(float(row[13]) if row[13] is not None else None)
        cols["se"].append(float(row[14]) if row[14] is not None else None)
        cols["neg_log10p"].append(_nlp_fenland(row[15]))
        cols["maf"].append(min(eaf, 1.0 - eaf) if eaf is not None else None)
        cols["eaf"].append(eaf)
        cols["a1"].append(_strip_quote(row[10]))
        cols["a2"].append(_strip_quote(row[11]))
        cols["ancestry"].append("EUR")
        cols["source"].append("fenland")
    wb.close()
    logger.info("Fenland: %d lead cis-pQTLs of %d cis rows", len(cols["rsid"]), n_cis)
    return _frame(cols)


# ── CKB Atlas (SomaScan v4.1, Zenodo) ───────────────────────────────────────


def parse_ckb_atlas(path: str | Path) -> pl.DataFrame:
    """CKB Atlas sentinel cis-pQTLs (SomaScan v4.1; Supplementary Tables ST2 and ST3).

    ST3 gives each association's SomaScan ID (column 0), cis or trans call (2), variant as
    ``chr:pos:EA:REF`` (3), chromosome (4), GRCh38 position (5), EA (8), REF (9), EAF (10),
    beta (12), SE (13) and P (14); rows with ``cis`` in column 2 are kept, ``a1`` is EA and
    ``a2`` is REF. The betas are in residual SD units of ANML-normalized, rank-inverse-normalized
    protein levels. ST2 (from row 3) gives each SeqId's gene symbol (column 6) and UniProt (10).
    ``ancestry`` is ``"EAS"`` and ``source`` is ``ckb_atlas``. A non-positive ``P`` gives a null
    ``neg_log10p``; the loader derives it from beta and SE.

    Raises:
        ValueError: For a SomaScan ID not of the form ``seq.<analyte>.<revision>``; an ST2 ID
            that disagrees with its hyphenated SeqId; or an ST3 variant string that disagrees
            with its chromosome, position or alleles.
        KeyError: For an ST3 SeqId that ST2 does not list.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    annotation: dict[str, tuple[str | None, str | None]] = {}
    for row in wb["ST2"].iter_rows(min_row=3, values_only=True):
        if row[0] is None or not str(row[0]).startswith("seq."):
            continue
        seqid = _ckb_seqid(str(row[0]))
        if seqid.replace("_", "-") != row[1]:
            raise ValueError(f"ST2 ID {row[0]!r} disagrees with SeqId {row[1]!r}")
        annotation[seqid] = (row[6], row[10])
    cols = _columns()
    for row in wb["ST3"].iter_rows(min_row=3, values_only=True):
        if row[0] is None or row[2] != "cis":
            continue
        seqid = _ckb_seqid(str(row[0]))
        gene, uniprot = annotation[seqid]
        chrom = str(int(row[4]))
        chrom = "X" if chrom == "23" else chrom
        variant = str(row[3]).split(":")
        if (variant[0].removeprefix("chr"), int(variant[1]), variant[2], variant[3]) != (chrom, int(row[5]), row[8], row[9]):
            raise ValueError(f"ST3 variant {row[3]!r} disagrees with its chromosome, position or alleles")
        eaf = float(row[10])
        cols["rsid"].append(None)
        cols["chrom"].append(chrom)
        cols["pos"].append(int(row[5]))
        cols["protein_id"].append(gene)
        cols["uniprot"].append(uniprot)
        cols["platform_id"].append(seqid)
        cols["beta"].append(float(row[12]))
        cols["se"].append(float(row[13]))
        cols["neg_log10p"].append(_nlp_split_exponent(float(row[14])))
        cols["maf"].append(min(eaf, 1.0 - eaf))
        cols["eaf"].append(eaf)
        cols["a1"].append(row[8])
        cols["a2"].append(row[9])
        cols["ancestry"].append("EAS")
        cols["source"].append("ckb_atlas")
    wb.close()
    return _frame(cols)


# ── INTERVAL (SomaScan, hg19) ───────────────────────────────────────────────

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def parse_interval(path: str | Path, liftover: Liftover, rsids: RsidLocator) -> pl.DataFrame:
    """INTERVAL sentinel cis-pQTLs (Sun et al. 2018, Nature, sheet ``ST4 - pQTL summary``), lifted to hg38.

    Rows from row 7 whose column 14 is ``cis`` are kept, with the meta-analysis estimates (22
    beta, 23 SE, 24 P). Other columns: 1 SOMAmer ID (``<name>.<analyte>.<revision>.<version>``;
    ``platform_id`` is ``<analyte>_<revision>``), 2 protein name, 4 UniProt, 5 rsID, 6 chromosome,
    7 hg19 position, 10 effect allele, 11 other allele, 12 EAF. Positions are lifted by
    ``analysis.external.liftover.lift``, and a row that does not lift is dropped. On a minus-strand
    lift both alleles are reverse-complemented, which keeps ``beta`` on the effect allele.
    ``ancestry`` is ``"EUR"``.

    Raises:
        ValueError: For a SOMAmer ID whose last three fields are not numeric.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    cols = _columns()
    for row in wb["ST4 - pQTL summary"].iter_rows(min_row=7, values_only=True):
        if row[0] is None or str(row[14]).strip() != "cis":
            continue
        chrom = str(int(row[6]))
        hit = lift(liftover, rsids, chrom, int(row[7]), str(row[5]))
        if hit is None:
            continue
        parts = str(row[1]).split(".")
        if not all(part.isdigit() for part in parts[-3:]):
            raise ValueError(f"unexpected SOMAmer ID {row[1]!r}; expected <name>.<analyte>.<revision>.<version>")
        effect, other = str(row[10]), str(row[11])
        if hit.strand == "-":
            effect, other = effect.translate(_COMPLEMENT)[::-1], other.translate(_COMPLEMENT)[::-1]
        eaf = float(row[12])
        cols["rsid"].append(str(row[5]))
        cols["chrom"].append(chrom)
        cols["pos"].append(hit.pos)
        cols["protein_id"].append(str(row[2]))
        cols["uniprot"].append(str(row[4]))
        cols["platform_id"].append("_".join(parts[-3:-1]))
        cols["beta"].append(float(row[22]))
        cols["se"].append(float(row[23]))
        cols["neg_log10p"].append(_nlp_split_exponent(float(row[24])))
        cols["maf"].append(min(eaf, 1.0 - eaf))
        cols["eaf"].append(eaf)
        cols["a1"].append(effect)
        cols["a2"].append(other)
        cols["ancestry"].append("EUR")
        cols["source"].append("interval")
    wb.close()
    return _frame(cols)


# ── AGES-Reykjavik (SomaScan, serum, hg19) ──────────────────────────────────

AGES_SCHEMA: dict[str, pl.DataType] = {
    **CATALOG_SCHEMA,
    "pos_hg19": pl.Int64,
    "liftover_strand": pl.String,
}


def decode_seqid_uniprot(path: str | Path) -> dict[str, str]:
    """SeqId (``14151_4``) -> UniProt from every row of deCODE's ST02 (Ferkingstad et al. 2021): SeqId in column 9, UniProt in 8."""
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    out = {
        str(row[9]).strip(): str(row[8]).strip()
        for row in wb["ST02"].iter_rows(min_row=4, values_only=True)
        if row[9] is not None and row[8] is not None
    }
    wb.close()
    return out


def parse_ages(
    path: str | Path, seqid_uniprot: dict[str, str], liftover: Liftover, rsids: RsidLocator
) -> pl.DataFrame:
    """AGES-Reykjavik cis-pQTLs (Gudjonsson et al. 2022, Nat Commun, Supplementary Data 3, sheet ``Data 3``), lifted to hg38.

    The discovery cohort is Icelandic serum on a SomaScan panel. Rows from row 5 whose column 9
    (``cis/trans``) is ``cis`` are kept, with the single-variant estimates (12 beta, 13 SE, 14 P).
    Other columns: 2 SOMAmer ID (``14151_4_3``; ``platform_id`` is its first two fields, ``14151_4``),
    3 gene symbol, 5 rsID, 6 chromosome (23 is X), 7 hg19 position, 10 effect allele, 11 EAF. The sheet has no
    other allele, so ``a2`` is null, and no UniProt: ``uniprot`` comes by SeqId from
    ``seqid_uniprot`` (``decode_seqid_uniprot``), null when absent. Positions are lifted by
    ``analysis.external.liftover.lift``; ``pos`` is null where the variant does not lift, the effect
    allele is reverse-complemented on a minus-strand lift (``eaf``, the frequency of that allele, is
    kept as published), and ``pos_hg19`` and ``liftover_strand`` keep the original position and the
    strand of the lift.
    A P of 0 gives a null ``neg_log10p``. ``maf`` is ``min(eaf, 1 - eaf)``. ``ancestry`` is ``"EUR"``.
    """
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    cols: dict[str, list] = {k: [] for k in AGES_SCHEMA}
    for row in wb["Data 3"].iter_rows(min_row=5, values_only=True):
        if row[0] is None or row[9] != "cis":
            continue
        chrom = _chrom_str(row[6])
        chrom = "X" if chrom == "23" else chrom
        pos_hg19 = int(row[7])
        seqid = "_".join(str(row[2]).strip().split("_")[:2])
        hit = lift(liftover, rsids, chrom, pos_hg19, _safe_str(row[5]))
        a1 = _safe_str(row[10])
        if hit is not None and hit.strand == "-" and a1 is not None:
            a1 = a1.translate(_COMPLEMENT)[::-1]
        eaf = float(row[11])
        cols["rsid"].append(_safe_str(row[5]))
        cols["chrom"].append(chrom)
        cols["pos"].append(hit.pos if hit else None)
        cols["protein_id"].append(_safe_str(row[3]))
        cols["uniprot"].append(seqid_uniprot.get(seqid))
        cols["platform_id"].append(seqid)
        cols["beta"].append(float(row[12]))
        cols["se"].append(float(row[13]))
        cols["neg_log10p"].append(_nlp_split_exponent(row[14]))
        cols["maf"].append(min(eaf, 1.0 - eaf))
        cols["eaf"].append(eaf)
        cols["a1"].append(a1)
        cols["a2"].append(None)
        cols["ancestry"].append("EUR")
        cols["source"].append("ages")
        cols["pos_hg19"].append(pos_hg19)
        cols["liftover_strand"].append(hit.strand if hit else None)
    wb.close()
    return pl.DataFrame(cols, schema=AGES_SCHEMA)
