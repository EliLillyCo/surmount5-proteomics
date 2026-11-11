"""Write the UniProt gene-update layer over the vendor assay lists (the build reads it).

    pixi run python analysis/external/assay_gene_updates.py [--cache-dir <dir>]

The base layer is the two vendor files as published (``vendor_assay_tables.py``). This script
queries UniProt REST (``rest.uniprot.org``) once for every accession they name and writes
``analysis/reference/assay_gene_updates.tsv``, the table the build applies on top. The build never
calls UniProt, so the table is the pin: it records the UniProt release (the ``X-UniProt-Release``
response header) on every row. UniProt data are CC BY 4.0.

One row per (assay, vendor accession) the build must change or must know about:

* every target of a multi-target assay, since the build cannot pair genes to accessions without
  UniProt. The unit is (assay, accession). Equal counts pair by position only when UniProt
  confirms every pair (the gene is the entry's gene symbol, a synonym or a former name);
  otherwise each vendor gene goes to the accessions UniProt names it on, a gene-symbol match
  beats a synonym, and on SomaScan the menu's HGNC and Entrez IDs break a remaining tie. A
  target no vendor gene names takes its gene symbol; one with none is ``unresolved``.
* a single-target assay whose gene changes, by the rules below, or that is kept for review.

An entry's gene symbols (``Entry.primary``) are the HGNC symbols its HGNC cross-references record in
UniProt release 2026_03. A
read-through locus is dropped when a plain gene is linked too (AARSD1, not PTGES3L-AARSD1; the locus
type comes from the pinned HGNC complete set, ``HGNC_URL``). An accession that links several genes
(identical proteins from paralogous loci) takes every one of them, except pseudogene loci (TEX28, not
TEX28P1/TEX28P2), whatever the vendor lists (``PARALOG``: DEFA1|DEFA1B, CGB3|CGB5|CGB8). An entry without an HGNC cross-reference keeps UniProt's
gene name. UniProt's own gene name, where it differs, counts as a synonym.
Pseudogene and non-coding symbols are symbols like any other (ADGRF2P, SLIRP-OT1).

Rules for one vendor gene of one accession (``update_gene``):

* the entry's gene symbol: kept, no row; of a paralog accession, every linked gene (``PARALOG``),
  with no row when the vendor already lists them all;
* a synonym or a former name (UniSave history) of the entry, or the same name spelled with ``-``
  for ``_``: renamed to the symbol (``rename``); the vendor name stays the assay label;
* blank: the symbol (``blank filled``);
* a vendor code (an Olink code such as YP010 or F231L, a LOC or FLJ placeholder, or the
  entry's UniProt mnemonic, as MR1L2 for MRFAP1L2): the symbol (``vendor code replaced``); a code
  whose entry has no gene stays as the vendor gives it, and a SomaScan aptamer of the same
  accession, blank in the menu, takes it (``vendor code from the Olink list, same accession``);
* anything else is kept and listed for review (different gene, no UniProt gene name, accession not
  in UniProt); the vendor genes of an accession not in UniProt are resolved through HGNC, a previous
  symbol taking its approved symbol (``resolve_not_in_uniprot``: GOLGA6L3|GOLGA6L3P -> GOLGA6L3P).

``CURATED`` holds the curated accession and gene rows. For an assay absent from the vendor list,
the accession is read from the supplied assay summary and the genes from UniProt (``assay absent
from vendor list``).

The annotation build (stage ``vendor_assay_tables``) takes each gene symbol's HGNC ID from the pinned
HGNC release alone (``hgnc_aliases.symbol_ids_from_hgnc``). This script checks that those IDs equal the
ones UniProt's cross-references record (``check_hgnc_ids``), so the build needs no UniProt call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests

from analysis.external import hgnc_aliases as ha
from analysis.external import vendor_assay_tables as vat
from analysis.external.pinned import fetch_pinned

UNIPROT_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNISAVE = "https://rest.uniprot.org/unisave/{accession}"
# HGNC complete set (pinned in analysis.external.hgnc_aliases): the locus type of every HGNC ID an entry
# cross-references (read-through or not). The gene symbols themselves are the ones UniProt records.
HGNC_RELEASE, HGNC_URL, HGNC_SHA256 = ha.HGNC_RELEASE, ha.HGNC_URL, ha.HGNC_SHA256
HGNC_READ_THROUGH = "readthrough"
# A paralog expansion never adds a pseudogene locus (HGNC locus group). An entry whose only locus is a
# pseudogene keeps it.
HGNC_PSEUDOGENE_GROUP = "pseudogene"
BATCH = 400
TIMEOUT = 300
HISTORY_CHUNK = 20
ENTRY_FIELDS = "accession,id,gene_names,xref_hgnc,xref_geneid"
CACHE_DIR = Path.home() / ".cache" / "surmount5-proteomics" / "uniprot"

# UniProt accession format (uniprot.org/help/accession_numbers).
ACCESSION = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")
# Vendor labels that are replaced by the gene symbol: Olink codes (YP010, F231L, VP113) and LOC/FLJ names.
# Tested only on a gene that is not a UniProt name of its accession, so CD101 or S100G never match.
NOT_A_SYMBOL_PATTERN = re.compile(r"[A-Z]{1,2}[0-9]{3}[A-Z]?|LOC[0-9]+|FLJ[0-9]+")

# Reasons. A row's reason joins the resolution step and every update applied, with "; ".
RENAME = "rename"
BLANK_FILLED = "blank filled"
NOT_A_SYMBOL = "vendor code replaced"
ANALYTE_LABEL = "analyte label"
ACCESSION_FILLED = "accession filled"
ABSENT = vat.ABSENT_REASON
MULTI_POSITION = "multi-target: paired by position, UniProt confirms"
MULTI_UNIPROT = "multi-target: matched through UniProt"
MULTI_TIE_BREAK = "multi-target: tie broken by the menu's HGNC/Entrez IDs"
MULTI_NO_VENDOR_GENE = "multi-target: no vendor gene names this target, its gene symbol"
UNRESOLVED = "unresolved"
REVIEW_DIFFERENT_GENE = "review: vendor and UniProt name different genes, vendor gene kept"
REVIEW_NO_UNIPROT_GENE = "review: UniProt has no gene name, vendor gene kept"
REVIEW_NOT_IN_UNIPROT = "review: accession not in this UniProt release, vendor gene kept"
HGNC_PREVIOUS = "previous HGNC symbol, the approved symbol"
OLINK_CODE_SHARED = "vendor code from the Olink list, same accession"
REVIEW_SHARED_GENE = vat.SHARED_GENE_REASON
REVIEW_MERGED_LOCUS = "review: HGNC complex locus constituent symbol kept, UniProt merged the entry into PALM2AKAP2"
PARALOG = "paralog accession: every HGNC-linked gene"

UPDATES_PATH = vat.UPDATES_PATH
COLUMNS = (
    "platform",
    "assay_id",
    "vendor_accession",
    "accession",
    "vendor_gene",
    "updated_gene",
    "reason",
    "uniprot_release",
)


@dataclass(frozen=True)
class GeneRecord:
    """One gene of a UniProt entry: its primary name (None when UniProt gives only ORF names) and synonyms."""

    name: str | None
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    """What UniProt says about one vendor accession.

    Attributes:
        accession: The vendor's accession.
        status: ``current``; ``secondary`` (merged or demerged into ``targets``); ``deleted``;
            ``not an accession`` (the vendor text is not an accession at all).
        targets: The current entries the accession resolves to.
        genes: Gene records of those entries.
        former: Names an earlier version of the entry carried and the current one does not.
        hgnc: ``(HGNC ID, symbol)`` cross-references, the symbol as UniProt records it.
        geneid: NCBI Gene IDs.
        mnemonic: The entry name without its species suffix (``MR1L2`` of ``MR1L2_HUMAN``); None
            when the accession resolves to several entries or none.
        hgnc_symbols: The symbols the entry's HGNC cross-references record, read-through loci
            dropped when a plain gene is linked (``hgnc_symbols``); empty when the entry has none, and
            UniProt's gene names stand.
    """

    accession: str
    status: str
    targets: tuple[str, ...]
    genes: tuple[GeneRecord, ...]
    former: tuple[str, ...] = ()
    hgnc: tuple[tuple[str, str], ...] = ()
    geneid: tuple[str, ...] = ()
    mnemonic: str | None = None
    hgnc_symbols: tuple[str, ...] = ()

    @property
    def uniprot_names(self) -> tuple[str, ...]:
        """UniProt's primary gene names, one per gene record that has one, in entry order."""
        return tuple(dict.fromkeys(g.name for g in self.genes if g.name))

    @property
    def primary(self) -> tuple[str, ...]:
        """The entry's gene symbols: its HGNC cross-references' symbols, else UniProt's gene names."""
        return self.hgnc_symbols or self.uniprot_names

    @property
    def synonyms(self) -> tuple[str, ...]:
        """UniProt's synonyms, and its gene names where HGNC's symbol differs."""
        names = (n for n in self.uniprot_names if n not in self.primary)
        return tuple(dict.fromkeys([*(s for g in self.genes for s in g.synonyms), *names]))

    def symbols_of(self, name: str) -> tuple[str, ...]:
        """The gene symbols for one of UniProt's gene names: every symbol of a paralog accession, else
        the name itself (or the entry's only HGNC symbol)."""
        if len(self.primary) > 1:
            return self.primary
        if not self.hgnc_symbols or name in self.hgnc_symbols:
            return (name,)
        return self.hgnc_symbols

    @property
    def in_uniprot(self) -> bool:
        return self.status in ("current", "secondary")


@dataclass(frozen=True)
class Curated:
    """A row no rule derives: the accession and gene to use, and why."""

    accession: str
    gene: str
    reason: str


@dataclass(frozen=True)
class UpdateRow:
    platform: str
    assay_id: str
    vendor_accession: str
    accession: str
    vendor_gene: str | None
    updated_gene: str | None
    reason: str


# (platform, assay, vendor accession) -> the row to write. Each must match the vendor file.
CURATED: dict[tuple[str, str, str], Curated] = {
    # Olink's list names the assay INS-CPEPTIDE (the analyte); the gene is INS and the assay label
    # stays INS-CPEPTIDE.
    ("olink", "OID44722", "P01308"): Curated(
        "P01308", "INS", f"{ANALYTE_LABEL}: INS-CPEPTIDE is the assay label, the gene is UniProt's"
    ),
    # Curated constituent-symbol rows: the vendor's genes are kept.
    ("olink", "OID43351", "Q9Y2D5"): Curated("Q9Y2D5", "AKAP2", REVIEW_MERGED_LOCUS),
    ("olink", "OID44072", "Q8IXS6"): Curated("Q8IXS6", "PALM2", REVIEW_MERGED_LOCUS),
    ("soma", "34463-30", "Q9Y2D5"): Curated("Q9Y2D5", "AKAP2", REVIEW_MERGED_LOCUS),
    # The list's UniProt cell holds the analyte name NT-proBNP; the accession is P16860 (NPPB).
    ("olink", "OID44822", "NT-proBNP"): Curated(
        "P16860",
        "NPPB",
        f"{ANALYTE_LABEL}: NT-proBNP is the assay label, the gene is UniProt's; "
        f"{ACCESSION_FILLED}: the list gives the analyte name NT-proBNP; the accession is P16860 (NPPB)",
    ),
}


# ---------------------------------------------------------------------------
# UniProt entries
# ---------------------------------------------------------------------------


def entry_from_json(
    accession: str, status: str, targets: tuple[str, ...], raws: list[dict], former: tuple[str, ...]
) -> Entry:
    """An ``Entry`` from the UniProtKB JSON of the accession's current entries."""
    genes: list[GeneRecord] = []
    hgnc: list[tuple[str, str]] = []
    geneid: list[str] = []
    for raw in raws:
        for gene in raw.get("genes", []):
            name = gene.get("geneName", {}).get("value")
            synonyms = tuple(s["value"] for s in gene.get("synonyms", []))
            if name or synonyms:
                genes.append(GeneRecord(name, synonyms))
        for xref in raw.get("uniProtKBCrossReferences", []):
            if xref["database"] == "HGNC":
                symbol = next((p["value"] for p in xref.get("properties", []) if p["key"] == "GeneName"), "")
                hgnc.append((xref["id"], symbol))
            elif xref["database"] == "GeneID":
                geneid.append(xref["id"])
    mnemonic = raws[0]["uniProtkbId"].rsplit("_", 1)[0] if len(raws) == 1 else None
    return Entry(accession, status, targets, tuple(genes), former, tuple(hgnc), tuple(geneid), mnemonic)


def unisave_names(text: str) -> set[str]:
    """Every gene name and synonym on the ``GN`` lines of UniSave flat-file entry versions."""
    names: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("GN   "):
            continue
        body = re.sub(r"\{[^}]*\}", "", line[5:])
        fields = re.findall(r"(Name|Synonyms)=([^;]+);", body)
        if fields:
            for _, value in fields:
                names.update(v.strip() for v in value.split(",") if v.strip())
        else:  # pre-2000 entries: "GN   PP32R1." or "GN   A OR B."
            names.update(v.strip(" .") for v in re.split(r"\s+(?:OR|AND)\s+", body) if v.strip(" ."))
    return names


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return name.upper().replace("-", "_")


def _is_mnemonic(gene: str, entry: Entry) -> bool:
    return entry.mnemonic is not None and _norm(gene) == _norm(entry.mnemonic)


def is_name(gene: str, entry: Entry) -> bool:
    """Whether ``gene`` is a primary name, synonym, former name or the mnemonic of the entry (or a spelling of one)."""
    names = {*entry.primary, *entry.synonyms, *entry.former}
    return gene in names or _norm(gene) in {_norm(n) for n in names} or _is_mnemonic(gene, entry)


def update_gene(gene: str | None, entry: Entry) -> tuple[tuple[str, ...], str | None]:
    """The gene(s) one vendor gene of one accession becomes, and the reason (None: unchanged, nothing to list)."""
    primary = entry.primary
    if gene is None:
        if not entry.in_uniprot:
            return (), REVIEW_NOT_IN_UNIPROT
        return (primary, BLANK_FILLED) if primary else ((), REVIEW_NO_UNIPROT_GENE)
    if gene in primary:
        return (primary, PARALOG) if len(primary) > 1 else ((gene,), None)
    if not entry.in_uniprot:
        return (gene,), REVIEW_NOT_IN_UNIPROT
    spelled = [p for p in primary if _norm(p) == _norm(gene)]
    if spelled:
        return (primary if len(primary) > 1 else (spelled[0],)), RENAME
    if gene in entry.uniprot_names:
        return entry.symbols_of(gene), RENAME
    for record in entry.genes:
        if gene in record.synonyms and record.name:
            return entry.symbols_of(record.name), RENAME
    if gene in entry.former:
        return primary, RENAME
    if (NOT_A_SYMBOL_PATTERN.fullmatch(gene) or _is_mnemonic(gene, entry)) and primary:
        return primary, NOT_A_SYMBOL
    if not primary:
        return (gene,), REVIEW_NO_UNIPROT_GENE
    return (gene,), REVIEW_DIFFERENT_GENE


def _apply(genes: list[str], entry: Entry) -> tuple[str | None, list[str]]:
    """Updated genes (pipe-joined, order kept, duplicates dropped) and the reasons that applied.

    A paralog accession whose vendor genes are already every linked gene (in any order) is unchanged.
    """
    new: list[str] = []
    reasons: list[str] = []
    for gene in genes or [None]:
        out, reason = update_gene(gene, entry)
        new.extend(out)
        if reason:
            reasons.append(reason)
    reasons = list(dict.fromkeys(reasons))
    if reasons == [PARALOG] and set(new) == set(genes):
        return "|".join(genes), []
    return ("|".join(dict.fromkeys(new)) or None), reasons


def _join(genes: list[str]) -> str | None:
    return "|".join(genes) or None


def resolve_assay(
    platform: str,
    assay_id: str,
    accessions: list[str],
    genes: list[str],
    entries: dict[str, Entry],
    *,
    hgnc_ids: list[str] = (),
    entrez_ids: list[str] = (),
    curated: dict[tuple[str, str, str], Curated] | None = None,
) -> list[UpdateRow]:
    """The update-layer rows of one vendor assay.

    Args:
        platform: ``olink`` or ``soma``.
        assay_id: OlinkID or SeqId.
        accessions: The vendor's UniProt accessions, in vendor order.
        genes: The vendor's gene names, in vendor order (empty when blank).
        entries: Vendor accession -> UniProt ``Entry``.
        hgnc_ids, entrez_ids: The menu's IDs for the row (SomaScan), used only to break ties.
        curated: ``CURATED``-shaped overrides.

    Returns:
        Rows in accession order; none for a single-target assay whose gene stands.
    """
    curated = curated or {}

    def row(acc: str, vendor: list[str], updated: str | None, reasons: list[str]) -> UpdateRow:
        fixed = curated.get((platform, assay_id, acc))
        if fixed:
            return UpdateRow(platform, assay_id, acc, fixed.accession, _join(vendor), fixed.gene, fixed.reason)
        return UpdateRow(platform, assay_id, acc, acc, _join(vendor), updated, "; ".join(reasons))

    if len(accessions) == 1:
        (acc,) = accessions
        if (platform, assay_id, acc) in curated:
            return [row(acc, genes, None, [])]
        updated, reasons = _apply(genes, entries[acc])
        if not reasons:
            return []
        return [row(acc, genes, updated, reasons)]

    assigned: dict[str, list[str]] = {a: [] for a in accessions}
    how: dict[str, str] = {}
    unmatched: list[str] = []
    if len(accessions) == len(genes) and all(is_name(g, entries[a]) for a, g in zip(accessions, genes)):
        for a, g in zip(accessions, genes):
            assigned[a].append(g)
            how[a] = MULTI_POSITION
    else:
        for g in genes:
            candidates = [a for a in accessions if g in entries[a].primary]
            step = MULTI_UNIPROT
            if not candidates:
                candidates = [a for a in accessions if is_name(g, entries[a])]
                ids = set(hgnc_ids) | set(entrez_ids)
                if len(candidates) > 1 and ids:
                    narrowed = [
                        a for a in candidates
                        if ids & ({h for h, _ in entries[a].hgnc} | set(entries[a].geneid))
                    ]
                    if narrowed and len(narrowed) < len(candidates):
                        candidates, step = narrowed, MULTI_TIE_BREAK
            if not candidates:
                unmatched.append(g)
            for a in candidates:
                assigned[a].append(g)
                how[a] = step if how.get(a) in (None, MULTI_UNIPROT) else how[a]
    suffix = f" (unmatched vendor genes: {'|'.join(unmatched)})" if unmatched else ""
    rows = []
    for a in accessions:
        if assigned[a]:
            updated, reasons = _apply(assigned[a], entries[a])
            rows.append(row(a, assigned[a], updated, [how[a], *reasons]))
        elif entries[a].primary:
            rows.append(row(a, [], "|".join(entries[a].primary), [MULTI_NO_VENDOR_GENE + suffix]))
        else:
            rows.append(row(a, [], None, [UNRESOLVED + suffix]))
    return rows


def absent_assay_rows(
    platform: str, vendor: pl.DataFrame, analysed: pl.DataFrame, entries: dict[str, Entry]
) -> list[UpdateRow]:
    """Rows for the analyzed assays the vendor list lacks: the study's UniProt, UniProt's genes.

    Args:
        platform: ``olink`` or ``soma``.
        vendor: The vendor table (``AssayID`` column).
        analysed: ``AssayID`` and ``UniProt`` (the study's, ``_``/``|``/``,``/``;`` separated).
        entries: Accession -> ``Entry`` for the study accessions.
    """
    listed = set(vendor["AssayID"])
    rows = []
    for assay_id, uniprot in analysed.select("AssayID", "UniProt").iter_rows():
        if assay_id in listed:
            continue
        for acc in [a for a in re.split(r"[_|,;\s]+", uniprot or "") if a]:
            gene = "|".join(entries[acc].primary) or None
            rows.append(UpdateRow(platform, assay_id, acc, acc, None, gene, ABSENT))
    return rows


def share_olink_codes(rows: list[UpdateRow]) -> list[UpdateRow]:
    """SomaScan rows left without a gene take the Olink vendor code of the same accession.

    An accession without a UniProt gene keeps Olink's code (YU005, YO027) on the Olink assay; a
    SomaScan aptamer of that accession, blank in the menu, shows the same code
    (``OLINK_CODE_SHARED``) so both platforms name the protein alike.
    """
    codes = {
        r.accession: r.updated_gene
        for r in rows
        if r.platform == "olink" and r.updated_gene and REVIEW_NO_UNIPROT_GENE in r.reason
    }
    return [
        UpdateRow(r.platform, r.assay_id, r.vendor_accession, r.accession, r.vendor_gene, codes[r.accession], OLINK_CODE_SHARED)
        if r.platform == "soma" and r.updated_gene is None and r.accession in codes
        else r
        for r in rows
    ]


def keep_vendor_genes_of_distinct_proteins(
    rows: list[UpdateRow], vendor: dict[str, list[tuple[str, list[str], list[str], list[str], list[str]]]]
) -> list[UpdateRow]:
    """Two assays of different accessions that the update would give one single gene keep the vendor genes.

    Per platform, every assay's final genes are its update rows' (or the vendor's when it has none). A
    single gene that two or more assays of different accessions share, where the vendor genes differ,
    reverts the updated rows of those assays to their vendor genes without Olink codes (``REVIEW_SHARED_GENE``).
    An analyte assay shares its gene by design.

    Args:
        rows: The update rows.
        vendor: Platform -> ``vendor_rows`` (assay, accessions, genes, ...).
    """
    by_assay: dict[tuple[str, str], list[UpdateRow]] = {}
    for r in rows:
        by_assay.setdefault((r.platform, r.assay_id), []).append(r)
    final: dict[tuple[str, str], tuple[str, str, frozenset[str]]] = {}
    for platform, vrows in vendor.items():
        for assay, accs, genes, *_ in vrows:
            own = by_assay.get((platform, assay))
            if own and any(ANALYTE_LABEL in r.reason for r in own):
                continue
            new = {g for r in own for g in _split(r.updated_gene)} if own else set(genes)
            if len(new) == 1:
                final[(platform, assay)] = (next(iter(new)), "|".join(genes), frozenset(r.accession for r in own) if own else frozenset(accs))
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (platform, assay), (gene, _, _) in final.items():
        groups.setdefault((platform, gene), []).append((platform, assay))
    revert: set[tuple[str, str]] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        vendor_genes = {final[m][1] for m in members}
        accessions = {final[m][2] for m in members}
        if len(vendor_genes) > 1 and len(accessions) > 1:
            revert |= {m for m in members if m in by_assay}
    return [
        UpdateRow(r.platform, r.assay_id, r.vendor_accession, r.accession, r.vendor_gene, _vendor_genes_without_codes(r.vendor_gene), REVIEW_SHARED_GENE)
        if (r.platform, r.assay_id) in revert
        else r
        for r in rows
    ]


def _vendor_genes_without_codes(vendor_gene: str | None) -> str | None:
    """The vendor genes without Olink codes (``NOT_A_SYMBOL_PATTERN``)."""
    return "|".join(g for g in _split(vendor_gene) if not NOT_A_SYMBOL_PATTERN.fullmatch(g)) or None


def resolve_not_in_uniprot(rows: list[UpdateRow], names: pl.DataFrame) -> list[UpdateRow]:
    """Vendor genes of accessions not in UniProt, resolved through HGNC: a previous symbol of exactly one HGNC ID
    takes that ID's approved symbol; genes then appear once.

    SomaScan 34959-8 (A6NEY3): GOLGA6L3|GOLGA6L3P -> GOLGA6L3P, GOLGA6L3 being a previous symbol of the same
    pseudogene (HGNC:37441). An approved symbol, or a name HGNC does not hold or holds for several IDs, stays.
    A changed row's reason gains ``HGNC_PREVIOUS``.

    Args:
        rows: The update rows.
        names: ``hgnc_aliases.read_hgnc_names`` of the pinned release.
    """
    approved = set(names["symbol"])
    previous: dict[str, set[str]] = {}
    for symbol, prev in names.select("symbol", "prev_symbol").iter_rows():
        for name in prev:
            previous.setdefault(name, set()).add(symbol)
    out = []
    for r in rows:
        if REVIEW_NOT_IN_UNIPROT not in r.reason or not r.updated_gene:
            out.append(r)
            continue
        genes = _split(r.updated_gene)
        resolved = list(dict.fromkeys(
            g if g in approved or len(previous.get(g, ())) != 1 else next(iter(previous[g])) for g in genes
        ))
        if resolved == genes:
            out.append(r)
        else:
            out.append(UpdateRow(r.platform, r.assay_id, r.vendor_accession, r.accession, r.vendor_gene,
                                 "|".join(resolved), f"{r.reason}; {HGNC_PREVIOUS}"))
    return out


def check_curated_used(curated: dict[tuple[str, str, str], Curated], used: set[tuple[str, str, str]]) -> None:
    """Raise when a curated row names an (assay, accession) that is not in the vendor files."""
    stale = sorted(set(curated) - used)
    if stale:
        raise ValueError(f"curated update rows match no vendor assay: {stale}")


def write_table(rows: list[UpdateRow], release: str, path: Path = UPDATES_PATH) -> pl.DataFrame:
    """Write the rows (in the given order) with the UniProt release on each."""
    table = pl.DataFrame(
        [(*(getattr(r, c) for c in COLUMNS[:-1]), release) for r in rows],
        schema=dict.fromkeys(COLUMNS, pl.String),
        orient="row",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path, separator="\t")
    return table


# ---------------------------------------------------------------------------
# UniProt REST (the only network part)
# ---------------------------------------------------------------------------


class UniProt:
    """UniProt REST calls, each response cached as JSON or text under ``cache_dir``."""

    def __init__(self, cache_dir: Path):
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.releases: set[str] = set()

    def _get(self, key: str, url: str, params: dict, as_json: bool = True):
        path = self.cache / key
        if not path.exists():
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            path.write_text(response.text)
            (self.cache / f"{key}.release").write_text(response.headers["X-UniProt-Release"])
            time.sleep(0.2)
        self.releases.add((self.cache / f"{key}.release").read_text())
        text = path.read_text()
        return json.loads(text) if as_json else text

    def entries(self, accessions: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(accessions), BATCH):
            batch = accessions[i : i + BATCH]
            payload = self._get(
                f"accessions_{hashlib.sha256((ENTRY_FIELDS + ':' + ','.join(batch)).encode()).hexdigest()[:16]}.json",
                UNIPROT_ACCESSIONS,
                {"accessions": ",".join(batch), "format": "json", "fields": ENTRY_FIELDS, "size": BATCH},
            )
            out.update({e["primaryAccession"]: e for e in payload["results"]})
        return out

    def secondary(self, accession: str) -> list[dict]:
        payload = self._get(
            f"sec_{accession}_{hashlib.sha256(ENTRY_FIELDS.encode()).hexdigest()[:8]}.json",
            UNIPROT_SEARCH,
            {"query": f"sec_acc:{accession}", "format": "json", "fields": ENTRY_FIELDS, "size": 25},
        )
        return payload["results"]

    def former_names(self, accession: str, wanted: set[str]) -> set[str]:
        """The names in ``wanted`` that an earlier version of the entry carried (UniSave).

        Versions are read newest first, ``HISTORY_CHUNK`` at a time, stopping once every wanted
        name is found.
        """
        versions = self._get(f"unisave_{accession}.json", UNISAVE.format(accession=accession), {"format": "json"})
        numbers = [str(v["entryVersion"]) for v in versions["results"]]  # newest first
        found: set[str] = set()
        for i in range(0, len(numbers), HISTORY_CHUNK):
            text = self._get(
                f"unisave_{accession}_{numbers[i]}-{numbers[min(i + HISTORY_CHUNK, len(numbers)) - 1]}.txt",
                UNISAVE.format(accession=accession),
                {"format": "txt", "versions": ",".join(numbers[i : i + HISTORY_CHUNK])},
                as_json=False,
            )
            found |= unisave_names(text) & wanted
            if found == wanted:
                break
        return found


def hgnc_symbols(entry: Entry, hgnc: dict[str, tuple[str, str]]) -> tuple[str, ...]:
    """The symbols the entry's HGNC cross-references record (HGNC as of UniProt's build).

    A read-through locus (locus type from the pinned HGNC release) is dropped when a plain gene is
    linked too (AARSD1, not PTGES3L-AARSD1), and so is a pseudogene locus (locus group) when a gene
    that is not one is linked (TEX28, not TEX28P1/TEX28P2). Every other linked gene is kept.

    Raises:
        KeyError: When an ID is missing from the pinned HGNC release.
        ValueError: When a cross-reference records no symbol.
    """
    recorded = dict(entry.hgnc)
    blank = [i for i, s in recorded.items() if not s]
    if blank:
        raise ValueError(f"{entry.accession}: HGNC cross-references without a recorded symbol: {blank}")
    linked = [(recorded[i], hgnc[i][1], hgnc[i][2]) for i in recorded]
    if any(t == HGNC_READ_THROUGH for _, t, _ in linked) and any(t != HGNC_READ_THROUGH for _, t, _ in linked):
        linked = [x for x in linked if x[1] != HGNC_READ_THROUGH]
    if any(g == HGNC_PSEUDOGENE_GROUP for *_, g in linked) and any(g != HGNC_PSEUDOGENE_GROUP for *_, g in linked):
        linked = [x for x in linked if x[2] != HGNC_PSEUDOGENE_GROUP]
    return tuple(dict.fromkeys(s for s, *_ in linked))


def final_genes(
    vendor: dict[str, list[tuple[str, list[str], list[str], list[str], list[str]]]], rows: list[UpdateRow]
) -> list[tuple[str, list[str]]]:
    """``(accession, genes)`` of every (assay, accession) the annotation will hold.

    The update row's genes where there is one (its corrected accession), else the vendor's; then the
    absent-assay rows.
    """
    by_key = {(r.platform, r.assay_id, r.vendor_accession): r for r in rows}
    out = []
    for platform, vrows in vendor.items():
        for assay, accs, genes, *_ in vrows:
            for acc in accs:
                row = by_key.get((platform, assay, acc))
                out.append((row.accession, _split(row.updated_gene)) if row else (acc, genes))
    out += [(r.accession, _split(r.updated_gene)) for r in rows if r.reason == ABSENT]
    return out


def symbol_hgnc_ids(
    final: list[tuple[str, list[str]]], entries: dict[str, Entry], approved: dict[str, str]
) -> dict[str, str]:
    """Our gene symbol -> its HGNC ID.

    The ID the accession's cross-reference records for that symbol; else (a kept vendor gene such as
    AKAP2) the ID of HGNC's approved symbol; a name that is neither (HERVK-113, Olink codes) has none.

    Args:
        final: ``final_genes``.
        entries: Accession -> ``Entry``.
        approved: HGNC approved symbol -> HGNC ID (the pinned release).

    Raises:
        ValueError: When one symbol would take two IDs.
    """
    ids: dict[str, set[str]] = {}
    for accession, genes in final:
        recorded = {s: i for i, s in entries[accession].hgnc} if accession in entries else {}
        for gene in genes:
            hgnc_id = recorded.get(gene) or approved.get(gene)
            if hgnc_id:
                ids.setdefault(gene, set()).add(hgnc_id)
    clash = {g: sorted(i) for g, i in ids.items() if len(i) > 1}
    if clash:
        raise ValueError(f"gene symbols with several HGNC IDs: {clash}")
    return {g: next(iter(i)) for g, i in sorted(ids.items())}


def check_hgnc_ids(final: list[tuple[str, list[str]]], entries: dict[str, Entry], names: pl.DataFrame) -> dict[str, str]:
    """Our symbol -> HGNC ID, checked: the IDs UniProt records (``symbol_hgnc_ids``) must equal the IDs the
    annotation build derives from the HGNC release alone (``hgnc_aliases.symbol_ids_from_hgnc``).

    Raises:
        ValueError: Naming the symbols whose IDs differ.
    """
    approved = dict(names.select("symbol", "hgnc_id").iter_rows())
    recorded = symbol_hgnc_ids(final, entries, approved)
    derived = ha.symbol_ids_from_hgnc({g for _, genes in final for g in genes}, names)
    differ = {s: (recorded.get(s), derived.get(s)) for s in set(recorded) | set(derived) if recorded.get(s) != derived.get(s)}
    if differ:
        raise ValueError(f"HGNC IDs from UniProt and from the HGNC release differ (recorded, derived): {differ}")
    return recorded


def read_hgnc(path: Path) -> dict[str, tuple[str, str, str]]:
    """HGNC ID -> (approved symbol, locus type, locus group) of the HGNC complete set."""
    table = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    return {i: (s, t, g) for i, s, t, g in table.select("hgnc_id", "symbol", "locus_type", "locus_group").iter_rows()}


def load_entries(
    accessions: list[str],
    uniprot: UniProt,
    history: dict[str, set[str]] | None = None,
    hgnc: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Entry]:
    """Accession -> ``Entry``: current entries, secondary accessions through search.

    Args:
        accessions: Vendor or study accessions (text that is not an accession is kept as such).
        uniprot: The REST client.
        history: Accession -> vendor genes to look for in its earlier versions (``former``).
        hgnc: HGNC ID -> (symbol, locus type, locus group) (``read_hgnc``).
    """
    history = history or {}
    hgnc = hgnc or {}
    valid = sorted(a for a in set(accessions) if ACCESSION.fullmatch(a))
    raw = uniprot.entries(valid)
    out: dict[str, Entry] = {}
    for acc in sorted(set(accessions)):
        if not ACCESSION.fullmatch(acc):
            out[acc] = Entry(acc, "not an accession", (), ())
            continue
        if acc in raw:
            raws, status = [raw[acc]], "current"
        else:
            raws = uniprot.secondary(acc)
            status = "secondary" if raws else "deleted"
        current = entry_from_json(acc, status, tuple(r["primaryAccession"] for r in raws), raws, ())
        former: tuple[str, ...] = ()
        if acc in history and status == "current":
            former = tuple(sorted(uniprot.former_names(acc, history[acc])))
        out[acc] = Entry(
            acc,
            status,
            current.targets,
            current.genes,
            former,
            current.hgnc,
            current.geneid,
            current.mnemonic,
            hgnc_symbols(current, hgnc),
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _split(value: str | None) -> list[str]:
    return [v for v in (value or "").split("|") if v]


def vendor_rows(platform: str, table: pl.DataFrame) -> list[tuple[str, list[str], list[str], list[str], list[str]]]:
    """``(assay, accessions, genes, hgnc ids, entrez ids)`` of a vendor table, in file order."""
    hgnc = table["HGNC_ID"] if "HGNC_ID" in table.columns else [None] * table.height
    entrez = table["Entrez_Gene_ID"] if "Entrez_Gene_ID" in table.columns else [None] * table.height
    return [
        (assay, _split(uniprot), _split(genes), _split(h), _split(e))
        for (assay, uniprot, genes), h, e in zip(table.select(table.columns[0], "UniProt_ID", "Gene_Symbol").iter_rows(), hgnc, entrez)
    ]


def build_rows(
    vendor: dict[str, pl.DataFrame],
    analysed: dict[str, pl.DataFrame],
    uniprot: UniProt,
    hgnc: dict[str, tuple[str, str]],
) -> tuple[list[UpdateRow], str, dict[str, Entry]]:
    """Every update row of both platforms, the UniProt release they come from and the entries read
    (locus types from ``hgnc``)."""
    per_platform = {p: vendor_rows(p, t) for p, t in vendor.items()}
    accessions = sorted(
        {a for rows in per_platform.values() for _, accs, *_ in rows for a in accs}
        | {
            a
            for p, frame in analysed.items()
            for a in re.split(r"[_|,;\s]+", "|".join(frame["UniProt"].drop_nulls()))
            if a
        }
    )
    entries = load_entries(accessions, uniprot, hgnc=hgnc)
    # UniSave only for the vendor genes that look like symbols and are not a current name of their accession.
    # In a multi-target assay, only a gene that no target names is looked for, in every target.
    history: dict[str, set[str]] = {}
    for rows in per_platform.values():
        for _, accs, genes, *_ in rows:
            current = [a for a in accs if entries[a].status == "current"]
            for g in genes:
                if NOT_A_SYMBOL_PATTERN.fullmatch(g) or any(is_name(g, entries[a]) for a in accs):
                    continue
                for a in current:
                    history.setdefault(a, set()).add(g)
    entries = load_entries(accessions, uniprot, history, hgnc)
    out: list[UpdateRow] = []
    used: set[tuple[str, str, str]] = set()
    for platform, rows in per_platform.items():
        for assay, accs, genes, hgnc_ids, entrez_ids in rows:
            used |= {(platform, assay, a) for a in accs if (platform, assay, a) in CURATED}
            out += resolve_assay(
                platform, assay, accs, genes, entries, hgnc_ids=hgnc_ids, entrez_ids=entrez_ids, curated=CURATED
            )
        vendor_ids = pl.DataFrame({"AssayID": [r[0] for r in rows]})
        out += absent_assay_rows(platform, vendor_ids, analysed[platform], entries)
    check_curated_used(CURATED, used)
    out = share_olink_codes(keep_vendor_genes_of_distinct_proteins(out, per_platform))
    if len(uniprot.releases) != 1:
        raise RuntimeError(f"UniProt responses span releases {sorted(uniprot.releases)}; clear the cache and rerun")
    return out, uniprot.releases.pop(), entries


def analysed_assays(platform: str) -> pl.DataFrame:
    """``AssayID``, ``UniProt`` of the study's analyzed assays (the QC assay summary)."""
    from analysis.qc_contract import PLATFORM_SPECS, read_assay_summary

    return read_assay_summary(platform).select(
        pl.col(PLATFORM_SPECS[platform].protein_id).alias("AssayID"), pl.col("UniProt").cast(pl.String)
    )


def main(argv: list[str] | None = None) -> int:
    """Query UniProt, write the update table and check the HGNC IDs the build will derive."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--menu", type=Path, default=vat.MENU_PATH, help="the SomaLogic annotated menu")
    ap.add_argument("--olink-list", type=Path, default=vat.OLINK_LIST_PATH, help="the Olink assay list")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="UniProt response cache (clear it to re-query)")
    ap.add_argument("--out", type=Path, default=UPDATES_PATH)
    args = ap.parse_args(argv)
    vendor = {
        "olink": vat.read_olink_list(vat.olink_list(args.olink_list)),
        "soma": vat.read_somascan_menu(args.menu),
    }
    analysed = {p: analysed_assays(p) for p in vendor}
    hgnc_path = fetch_pinned(HGNC_URL, args.cache_dir / Path(HGNC_URL).name, HGNC_SHA256)
    rows, release, entries = build_rows(vendor, analysed, UniProt(args.cache_dir), read_hgnc(hgnc_path))
    rows = resolve_not_in_uniprot(rows, ha.read_hgnc_names(hgnc_path))
    table = write_table(rows, release, args.out)
    print(f"[ok] {args.out}: {table.height} rows, UniProt release {release}")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=120):
        print(table.group_by("platform", "reason").len().sort("platform", "reason"))
    per_platform = {p: vendor_rows(p, t) for p, t in vendor.items()}
    ids = check_hgnc_ids(final_genes(per_platform, rows), entries, ha.read_hgnc_names(hgnc_path))
    print(f"[ok] {len(ids)} gene symbols: HGNC IDs from UniProt equal those from HGNC {HGNC_RELEASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
