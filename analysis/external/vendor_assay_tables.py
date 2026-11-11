"""Build the assay annotation: the vendor assay lists (base layer) with the UniProt gene-update layer on top.

    pixi run python analysis/external/vendor_assay_tables.py

**Base layer: the vendor files, as published**, read from ``data/vendor/`` (or ``--menu`` /
``--olink-list``), each SHA-256 pinned and verified before it is read:

* Olink's public Explore HT assay list (Doc 1562, v2026-06-24, ``OLINK_URL``, no login): columns
  ``Olink ID*``, ``UniProt ID``, ``Gene name``; several targets are joined with ``_`` in both. Without
  a copy in ``data/vendor/`` it is downloaded through the pin (``olink_list``).
* The SomaScan 11K v5.0 plasma/serum annotated menu, SL00000906 Rev 10 (2026-03), sheet
  ``Annotations``, rows with ``Type`` Protein and ``Organism`` Human: ``SeqId``, ``UniProt ID``,
  ``Entrez Gene Name``, ``HGNC ID``, ``Entrez Gene ID``, several values joined with `` | ``. It needs a
  SomaLogic account (https://menu.somalogic.com) and is marked SomaLogic confidential; a reader
  downloads it and passes ``--menu``.

Columns are selected by name.

**Update layer: ``analysis/reference/assay_gene_updates.tsv``**, written by
``analysis/external/assay_gene_updates.py`` from UniProt, which this build never calls. It resolves
every multi-target assay one accession at a time and records each gene that changes (renames, blanks
filled, vendor codes, analyte labels, filled accessions) or is kept for review. Every row must match the
vendor files (assay, accession, vendor gene) and every multi-target assay must have one row per
accession, or the build raises.

**HGNC release** (``analysis.external.hgnc_aliases``): the HGNC complete set, monthly archive 2026-09-04,
fetched through its pin into ``data/external/hgnc/`` (or passed with ``--hgnc``). From it this stage takes
each gene symbol's HGNC ID and writes the HGNC alias table; the release and the table are gitignored.

Outputs (``data/external/vendor_assay_tables/``, gitignored):

* ``assay_annotation.tsv``: one row per (platform, assay, vendor accession), vendor order:
  ``ANNOTATION_COLUMNS``. ``Gene_Symbol`` joins with ``|`` the genes of one accession (identical
  proteins from paralogous loci, such as P0DN86 = CGB3|CGB5|CGB8). ``Display_Label`` is the assay's
  figure label (``analysis.protein_labels.protein_label``) and ``Is_Analyte_Label`` marks the analyte
  assays (an update reason part ``analyte label``), on every row of the assay.
  ``HGNC_ID`` holds the HGNC ID of each gene of ``Gene_Symbol``, ``|``-joined in the same order (an
  empty element for a gene without one, such as HERVK-113 or an Olink code; null when no gene has
  one). ``Excluded_From_Gene_Level`` marks the assays left out of gene-level steps (pathways, Fig 3b):
  the assays marked ``SHARED_GENE_REASON`` (distinct proteins sharing one gene).
* ``olink_explore_ht_annotation.tsv`` (``OlinkID``) and ``somascan_11k_annotation.tsv`` (``SeqId``):
  one row per assay, ``UniProt_ID`` and ``Gene_Symbol`` pipe-joined in accession order,
  ``Display_Label`` and ``HGNC_ID``, derived from the long table.
* ``hgnc_aliases.tsv``: the HGNC alias table (``hgnc_aliases.build_alias_table``): our symbols with
  their HGNC IDs, and the current, previous and alias HGNC names of those IDs; the pathway step translates
  gene-set library members through it (``read_alias_table``).
* ``assay_annotation.inputs.json``: the SHA-256 of the four inputs (vendor files, update layer, HGNC
  release) and of the alias table written beside the annotation; ``read_annotation`` refuses an annotation
  built from other inputs, ``read_alias_table`` an alias table that is not this build's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl
import polars as pl

from analysis.external import hgnc_aliases as ha
from analysis.external.pinned import check_sha256, fetch_pinned, sha256_file

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = ROOT / "data" / "vendor"

# SomaLogic SomaScan 11K v5.0 plasma/serum annotated menu, SL00000906 Rev 10, 2026-03.
MENU_PATH = VENDOR_DIR / "SL00000906_SomaScan_11K_v5.0_Plasma_Serum_Annotated_Menu.xlsx"
MENU_SHA256 = "4284504d27bee6726e00ca3561913c12a01b9e2da9cd1ea489f73dd87c0102bf"
MENU_SHEET = "Annotations"
MENU_HEADER_CELL = "SeqId"
MENU_TYPE, MENU_ORGANISM = "Type", "Organism"
MENU_KEEP = {MENU_TYPE: "Protein", MENU_ORGANISM: "Human"}
MENU_COLUMNS = {
    "SeqId": "SeqId",
    "UniProt ID": "UniProt_ID",
    "Entrez Gene Name": "Gene_Symbol",
    "HGNC ID": "HGNC_ID",
    "Entrez Gene ID": "Entrez_Gene_ID",
}

# Olink Explore HT assay list (Doc 1562), the vendor's public download, v2026-06-24.
OLINK_URL = (
    "https://7074596.fs1.hubspotusercontent-na1.net/hubfs/7074596/000-documents/"
    "10-excel%20file/olink-explore-ht-assay-list.xlsx"
)
OLINK_LIST_PATH = VENDOR_DIR / "olink-explore-ht-assay-list.xlsx"
OLINK_SHA256 = "a48a0f7d364dfe0e9e66357954ad2d50be40f4d4a4685e894192c2a945fcb6a8"
OLINK_HEADER_CELL = "Olink ID*"
OLINK_COLUMNS = {"Olink ID*": "OlinkID", "UniProt ID": "UniProt_ID", "Gene name": "Gene_Symbol"}
OLINK_MULTI_SEPARATOR = "_"

UPDATES_PATH = ROOT / "analysis" / "reference" / "assay_gene_updates.tsv"
ABSENT_REASON = "assay absent from vendor list"

OUTPUT_DIR = ROOT / "data" / "external" / "vendor_assay_tables"
ANNOTATION_PATH = OUTPUT_DIR / "assay_annotation.tsv"
OLINK_ANNOTATION_PATH = OUTPUT_DIR / "olink_explore_ht_annotation.tsv"
SOMA_ANNOTATION_PATH = OUTPUT_DIR / "somascan_11k_annotation.tsv"
INPUTS_RECORD = OUTPUT_DIR / "assay_annotation.inputs.json"
ANNOTATION_COLUMNS = (
    "Platform",
    "AssayID",
    "Vendor_UniProt",
    "UniProt",
    "Vendor_Gene",
    "Gene_Symbol",
    "Update_Reason",
    "Display_Label",
    "Is_Analyte_Label",
    "HGNC_ID",
    "Excluded_From_Gene_Level",
)
# The update-layer reason (a "; "-joined part of it) that marks an analyte assay: INS-CPEPTIDE, NT-proBNP.
ANALYTE_LABEL_REASON = "analyte label"
# The update-layer reason that marks assays sharing one gene; gene-level steps leave the marked assays out.
SHARED_GENE_REASON = "review: distinct proteins would share one gene, vendor gene kept"
# Assays whose figure label is the vendor's assay name, not built from genes: the analyte assays and the
# assays marked ``SHARED_GENE_REASON``. On Olink the name is the list's gene cell, its "_" kept.
def _is_vendor_label(reason: pl.Expr) -> pl.Expr:
    parts = reason.fill_null("").str.split("; ")
    return parts.list.eval(pl.element().str.starts_with(ANALYTE_LABEL_REASON)).list.any() | parts.list.contains(SHARED_GENE_REASON)


# Our symbols that have no HGNC ID, pinned: Olink codes, HERV-K113 and its codes, and names HGNC does not hold.
# The build raises an error if the set changes.
SYMBOLS_WITHOUT_HGNC_ID = frozenset(
    {"F231L", "YP010", "YK038", "YU005", "YO027", "YP029", "YH007", "YB057", "YM017",
     "HERVK-113", "BARGIN", "L1RE1", "FLJ44635"}
)
# Platform -> (assay ID column of its wide table, wide table path).
WIDE = {"olink": ("OlinkID", OLINK_ANNOTATION_PATH), "soma": ("SeqId", SOMA_ANNOTATION_PATH)}


# ---------------------------------------------------------------------------
# Base layer: the vendor files
# ---------------------------------------------------------------------------


def _sheet_table(xlsx: Path, sheet: str | None, header_cell: str) -> tuple[list[str], list[tuple]]:
    """The header row (the first with a cell equal to ``header_cell``) and the rows below it."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list((wb[sheet] if sheet else wb.worksheets[0]).iter_rows(values_only=True))
    wb.close()
    at = next(i for i, row in enumerate(rows) if header_cell in row)
    header = [None if h is None else str(h).strip() for h in rows[at]]
    return header, rows[at + 1 :]


def _select(xlsx: Path, header: list[str], rows: list[tuple], columns: list[str]) -> pl.DataFrame:
    """The named columns of ``rows`` as text, with blank cells null.

    Raises:
        ValueError: When a column is not in ``header``.
    """
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"{xlsx} is missing columns {missing}")
    index = [header.index(c) for c in columns]
    records = [[None if row[i] is None else str(row[i]).strip() or None for i in index] for row in rows]
    return pl.DataFrame(records, schema=dict.fromkeys(columns, pl.String), orient="row")


def read_somascan_menu(xlsx: Path) -> pl.DataFrame:
    """``MENU_COLUMNS`` of the menu's human proteins, in sheet order, multi-values pipe-joined.

    Raises:
        RuntimeError: When ``xlsx`` does not match ``MENU_SHA256``.
        ValueError: When the sheet lacks a column this reads.
    """
    check_sha256(xlsx, MENU_SHA256)
    header, rows = _sheet_table(xlsx, MENU_SHEET, MENU_HEADER_CELL)
    table = _select(xlsx, header, rows, [*MENU_KEEP, *MENU_COLUMNS])
    for column, value in MENU_KEEP.items():
        table = table.filter(pl.col(column) == value)
    renamed = list(MENU_COLUMNS.values())[1:]
    return table.select(list(MENU_COLUMNS)).rename(MENU_COLUMNS).with_columns(
        pl.col(renamed).str.replace_all(r"\s*\|\s*", "|")
    )


def read_olink_list(xlsx: Path) -> pl.DataFrame:
    """``OlinkID``, ``UniProt_ID``, ``Gene_Symbol`` of the public Explore HT assay list, in list order.

    Raises:
        RuntimeError: When ``xlsx`` does not match ``OLINK_SHA256``.
        ValueError: When the sheet lacks a column this reads.
    """
    check_sha256(xlsx, OLINK_SHA256)
    header, rows = _sheet_table(xlsx, None, OLINK_HEADER_CELL)
    table = _select(xlsx, header, rows, list(OLINK_COLUMNS)).drop_nulls(OLINK_HEADER_CELL)
    return table.rename(OLINK_COLUMNS).with_columns(
        pl.col("UniProt_ID", "Gene_Symbol").str.replace_all(OLINK_MULTI_SEPARATOR, "|")
    )


def olink_list(path: Path = OLINK_LIST_PATH) -> Path:
    """The verified Olink list at ``path``, downloaded from ``OLINK_URL`` first when absent."""
    return fetch_pinned(OLINK_URL, path, OLINK_SHA256)


# ---------------------------------------------------------------------------
# Base + update layer
# ---------------------------------------------------------------------------


def read_updates(path: Path = UPDATES_PATH) -> pl.DataFrame:
    """The update layer, every column as text (an empty field is null)."""
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def _genes(col: str) -> pl.Expr:
    return pl.col(col).str.split("|").list.eval(pl.element().filter(pl.element() != ""))


def annotate(platform: str, table: pl.DataFrame, updates: pl.DataFrame, *, hgnc_ids: dict[str, str]) -> pl.DataFrame:
    """One platform's long annotation: the vendor table, one row per accession, with the update layer applied.

    Args:
        platform: ``olink`` or ``soma`` (the ``platform`` value of the update rows to apply).
        table: Vendor table; its first column is the assay ID, with ``UniProt_ID`` and ``Gene_Symbol``.
        updates: The update layer (``read_updates``), both platforms.
        hgnc_ids: Gene symbol -> HGNC ID (``hgnc_aliases.hgnc_ids`` of the alias table).

    Returns:
        ``ANNOTATION_COLUMNS`` in vendor order, then the assays absent from the vendor table.
        Without an update row the vendor value stands (``Vendor_Gene`` = ``Gene_Symbol``, the assay's
        gene string, ``Update_Reason`` null).

    Raises:
        ValueError: When a multi-target assay lacks a row for any accession, an update row matches no
            vendor (assay, accession) or names a vendor gene the vendor file does not give, a key
            repeats, or an absent-assay row names a listed assay.
    """
    assay_col = table.columns[0]
    base = (
        table.select(pl.col(assay_col).alias("AssayID"), "UniProt_ID", pl.col("Gene_Symbol").alias("_assay_genes"))
        .with_columns(
            pl.col("UniProt_ID").str.split("|").alias("Vendor_UniProt"),
            pl.col("UniProt_ID").str.split("|").list.len().alias("_n_targets"),
        )
        .explode("Vendor_UniProt")
    )
    rows = updates.filter(pl.col("platform") == platform).select(
        pl.col("assay_id").alias("AssayID"),
        pl.col("vendor_accession").alias("Vendor_UniProt"),
        "accession",
        "vendor_gene",
        "updated_gene",
        "reason",
    )
    duplicated = rows.filter(rows.select("AssayID", "Vendor_UniProt").is_duplicated())
    if duplicated.height:
        raise ValueError(f"{platform}: repeated update rows {duplicated.select('AssayID', 'Vendor_UniProt').rows()}")
    absent = rows.filter(pl.col("reason") == ABSENT_REASON)
    rows = rows.filter(pl.col("reason") != ABSENT_REASON)
    listed = absent.filter(pl.col("AssayID").is_in(base["AssayID"].implode()))
    if listed.height:
        raise ValueError(f"{platform}: assays marked absent are in the vendor list: {sorted(set(listed['AssayID']))}")
    stale = rows.join(base, on=["AssayID", "Vendor_UniProt"], how="anti")
    if stale.height:
        raise ValueError(f"{platform}: update rows match no vendor assay and accession: {stale.select('AssayID', 'Vendor_UniProt').rows()}")
    joined = base.join(rows, on=["AssayID", "Vendor_UniProt"], how="left", maintain_order="left")
    unresolved = joined.filter((pl.col("_n_targets") > 1) & pl.col("reason").is_null())
    if unresolved.height:
        raise ValueError(
            f"{platform}: multi-target assays without a resolution row per accession (regenerate "
            f"{UPDATES_PATH.name}): {sorted(set(unresolved['AssayID']))}"
        )
    single = pl.col("_n_targets") == 1
    vendor_genes = _genes("_assay_genes")
    wrong_gene = joined.filter(
        pl.col("reason").is_not_null()
        & pl.when(single)
        .then(~pl.col("vendor_gene").eq_missing(pl.col("_assay_genes")))
        .otherwise(~_genes("vendor_gene").fill_null([]).list.set_difference(vendor_genes).list.len().eq(0))
    )
    if wrong_gene.height:
        raise ValueError(
            f"{platform}: update rows name a vendor gene the vendor file does not give: "
            f"{wrong_gene.select('AssayID', 'Vendor_UniProt', 'vendor_gene', '_assay_genes').rows()}"
        )
    updated = pl.col("reason").is_not_null()
    long = joined.select(
        pl.lit(platform).alias("Platform"),
        "AssayID",
        "Vendor_UniProt",
        pl.coalesce("accession", "Vendor_UniProt").alias("UniProt"),
        pl.when(updated).then(pl.col("vendor_gene")).otherwise(pl.col("_assay_genes")).alias("Vendor_Gene"),
        pl.when(updated).then(pl.col("updated_gene")).otherwise(pl.col("_assay_genes")).alias("Gene_Symbol"),
        pl.col("reason").alias("Update_Reason"),
    )
    added = absent.select(
        pl.lit(platform).alias("Platform"),
        "AssayID",
        "Vendor_UniProt",
        pl.col("accession").alias("UniProt"),
        pl.col("vendor_gene").alias("Vendor_Gene"),
        pl.col("updated_gene").alias("Gene_Symbol"),
        pl.col("reason").alias("Update_Reason"),
    )
    return with_display_labels(pl.concat([long, added]), hgnc_ids)


def with_display_labels(long: pl.DataFrame, ids: dict[str, str]) -> pl.DataFrame:
    """Add ``Display_Label`` (the assay's figure label, ``analysis.protein_labels.protein_label``),
    ``Is_Analyte_Label`` (a reason part starts with ``ANALYTE_LABEL_REASON``), ``HGNC_ID`` (each gene's
    ID from ``ids``, aligned) and ``Excluded_From_Gene_Level`` (a reason part is ``SHARED_GENE_REASON``)
    to every row of an assay. An analyte assay and an assay of ``SHARED_GENE_REASON`` are labeled by the
    vendor's name (``_is_vendor_label``; on Olink the list's ``_`` restored: HERVK-113_VP113)."""
    from analysis.protein_labels import protein_label

    is_analyte = (
        pl.col("Update_Reason").fill_null("").str.split("; ").list.eval(
            pl.element().str.starts_with(ANALYTE_LABEL_REASON)
        ).list.any()
    )
    excluded = pl.col("Update_Reason").fill_null("").str.split("; ").list.contains(SHARED_GENE_REASON)
    gene_ids = _genes("Gene_Symbol").list.eval(pl.element().replace_strict(ids, default="")).list.join("|")
    flagged = long.with_columns(
        is_analyte.alias("Is_Analyte_Label"),
        pl.when(gene_ids.str.replace_all("|", "", literal=True) != "").then(gene_ids).alias("HGNC_ID"),
    )
    flagged = flagged.with_columns(excluded.any().over("Platform", "AssayID").alias("Excluded_From_Gene_Level"))
    vendor_name = (
        pl.when(pl.col("Platform") == "olink")
        .then(pl.col("Vendor_Gene").str.replace_all("|", OLINK_MULTI_SEPARATOR, literal=True))
        .otherwise(pl.col("Vendor_Gene"))
    )
    per_assay = flagged.group_by("Platform", "AssayID", maintain_order=True).agg(
        _genes("Gene_Symbol").explode().drop_nulls().alias("_genes"),
        vendor_name.filter(_is_vendor_label(pl.col("Update_Reason"))).first().alias("_analyte"),
        pl.col("Is_Analyte_Label").any().alias("_any_analyte"),
    )
    labels = per_assay.select(
        "Platform",
        "AssayID",
        pl.struct("_genes", "_analyte")
        .map_elements(lambda r: protein_label(r["_genes"] or [], r["_analyte"]), return_dtype=pl.String)
        .alias("Display_Label"),
        pl.col("_any_analyte").alias("Is_Analyte_Label"),
    )
    return flagged.drop("Is_Analyte_Label").join(labels, on=["Platform", "AssayID"], how="left", maintain_order="left").select(
        ANNOTATION_COLUMNS
    )


def wide_annotation(long: pl.DataFrame, assay_col: str) -> pl.DataFrame:
    """One row per assay: accessions and their genes pipe-joined in accession order (genes once each),
    with the genes' HGNC IDs aligned."""
    pairs = long.select(
        "AssayID",
        _genes("Gene_Symbol").alias("_gene"),
        pl.col("HGNC_ID").fill_null("").str.split("|").alias("_id"),
    ).filter(pl.col("_gene").list.len() > 0).with_columns(
        pl.when(pl.col("_id").list.len() == pl.col("_gene").list.len()).then(pl.col("_id"))
        .otherwise(pl.col("_gene").list.eval(pl.lit(""))).alias("_id")
    ).explode("_gene", "_id").unique(["AssayID", "_gene"], maintain_order=True)
    ids = pairs.group_by("AssayID", maintain_order=True).agg(pl.col("_id").str.join("|").alias("HGNC_ID"))
    return (
        long.group_by("AssayID", maintain_order=True).agg(
            pl.col("UniProt").str.join("|").alias("UniProt_ID"),
            _genes("Gene_Symbol").explode().drop_nulls().unique(maintain_order=True).str.join("|").alias("Gene_Symbol"),
            pl.col("Display_Label").first(),
        )
        .join(ids, on="AssayID", how="left", maintain_order="left")
        .with_columns(
            pl.col("Gene_Symbol").replace("", None),
            pl.when(pl.col("HGNC_ID").str.replace_all("|", "", literal=True) != "").then(pl.col("HGNC_ID")).alias("HGNC_ID"),
        )
        .rename({"AssayID": assay_col})
    )


def check_symbols_without_hgnc_id(symbols, ids: dict[str, str], expected: frozenset[str]) -> None:
    """Raise unless the symbols without an HGNC ID are exactly ``expected`` (``SYMBOLS_WITHOUT_HGNC_ID``).

    Raises:
        ValueError: Naming the unexpected and the missing names.
    """
    without = set(symbols) - set(ids)
    if without != set(expected):
        raise ValueError(
            f"symbols without an HGNC ID differ from the pinned set: new {sorted(without - expected)}, "
            f"missing {sorted(set(expected) - without)}"
        )


def build_aliases(
    names: pl.DataFrame, symbols, expected_without_id: frozenset[str] | None = None
) -> tuple[dict[str, str], pl.DataFrame]:
    """Our symbols' HGNC IDs and the alias table, from the HGNC release (``hgnc_aliases``).

    Raises:
        ValueError: When two of our symbols differ only by case (``hgnc_aliases.case_collisions``), or the
            symbols without an ID are not ``expected_without_id``.
    """
    ids = ha.symbol_ids_from_hgnc(symbols, names)
    check_symbols_without_hgnc_id(
        symbols, ids, SYMBOLS_WITHOUT_HGNC_ID if expected_without_id is None else expected_without_id
    )
    aliases = ha.build_alias_table(names, ids, ha.HGNC_RELEASE)
    collisions = ha.case_collisions(aliases)
    if collisions:
        raise ValueError(f"our symbols that differ only by case: {collisions}")
    return ids, aliases


def _inputs(updates: Path) -> dict[str, str]:
    """The pins of the vendor files and the HGNC release (each verified when read) and the update layer's digest."""
    return {
        "menu_sha256": MENU_SHA256,
        "olink_sha256": OLINK_SHA256,
        "updates_sha256": sha256_file(updates),
        "hgnc_sha256": ha.HGNC_SHA256,
    }


def _record(updates: Path, aliases: Path) -> dict[str, str]:
    """The inputs record: ``_inputs`` and the digest of the alias table written by the same build."""
    return {**_inputs(updates), "aliases_sha256": sha256_file(aliases)}


def _check_inputs(path: Path) -> dict[str, str]:
    """Raise when the annotation at ``path`` was built from another update layer or a vendor or HGNC pin.

    Returns:
        The inputs record.

    Raises:
        RuntimeError: Naming the stage to rerun.
    """
    record = json.loads((path.parent / INPUTS_RECORD.name).read_text())
    built = {k: v for k, v in record.items() if k != "aliases_sha256"}
    expected = _inputs(UPDATES_PATH)
    if built != expected:
        raise RuntimeError(f"{path} is stale (built from {built}, inputs are {expected}); rerun stage vendor_assay_tables")
    return record


def read_alias_table(path: Path = ha.ALIASES_PATH) -> pl.DataFrame:
    """The HGNC alias table this build wrote beside the annotation; refused when stale or not this build's.

    Raises:
        RuntimeError: Naming the stage to rerun.
    """
    record = _check_inputs(path)
    if sha256_file(path) != record["aliases_sha256"]:
        raise RuntimeError(f"{path} is not the alias table built with the annotation; rerun stage vendor_assay_tables")
    return ha.read_aliases(path)


def read_annotation(path: Path = ANNOTATION_PATH) -> pl.DataFrame:
    """The long annotation, every column as text; refused when stale (``_check_inputs``)."""
    _check_inputs(path)
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def assay_genes(platform: str, path: Path = ANNOTATION_PATH, *, gene_level: bool = False) -> dict[str, list[str]]:
    """Assay ID -> its genes over all its accessions (each once, accession order); empty when it has none.

    With ``gene_level``, the assays ``Excluded_From_Gene_Level`` are left out (the gene-level steps:
    pathways and Fig 3b).
    """
    long = read_annotation(path).filter(pl.col("Platform") == platform)
    if gene_level:
        long = long.filter(pl.col("Excluded_From_Gene_Level") != "true")
    grouped = long.group_by("AssayID", maintain_order=True).agg(
        _genes("Gene_Symbol").explode().drop_nulls().unique(maintain_order=True).alias("genes")
    )
    return {assay: genes for assay, genes in grouped.iter_rows()}


def summary(long: pl.DataFrame) -> pl.DataFrame:
    """Per platform: assays, (assay, accession) rows, rows updated, kept for review and unresolved."""
    reason = pl.col("Update_Reason").fill_null("")
    changed = ~pl.col("Gene_Symbol").eq_missing(pl.col("Vendor_Gene")) | (pl.col("UniProt") != pl.col("Vendor_UniProt"))
    return long.group_by("Platform", maintain_order=True).agg(
        pl.col("AssayID").n_unique().alias("assays"),
        pl.len().alias("rows"),
        changed.sum().alias("updated"),
        reason.str.contains("review:").sum().alias("review"),
        reason.str.starts_with("unresolved").sum().alias("unresolved"),
        reason.str.starts_with("multi-target").sum().alias("multi_target_rows"),
    )


def main(argv: list[str] | None = None) -> int:
    """Write the long and wide annotation tables and their inputs record."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--menu", type=Path, default=MENU_PATH, help="the SomaLogic annotated menu (data/vendor/ or a download)")
    ap.add_argument("--olink-list", type=Path, default=OLINK_LIST_PATH, help="the Olink assay list (fetched when absent)")
    ap.add_argument("--updates", type=Path, default=UPDATES_PATH, help="the UniProt update layer")
    ap.add_argument("--hgnc", type=Path, default=ha.HGNC_PATH, help="the HGNC complete set (fetched through its pin when absent)")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    updates = read_updates(args.updates)
    tables = {"olink": read_olink_list(olink_list(args.olink_list)), "soma": read_somascan_menu(args.menu)}
    names = ha.read_hgnc_names(fetch_pinned(ha.HGNC_URL, args.hgnc, ha.HGNC_SHA256))
    # Two passes: the genes first (no IDs), then their HGNC IDs from the release, then the annotation with them.
    genes = pl.concat([annotate(platform, table, updates, hgnc_ids={}) for platform, table in tables.items()])
    symbols = {g for cell in genes["Gene_Symbol"].drop_nulls() for g in cell.split("|") if g}
    ids, aliases = build_aliases(names, symbols)
    ha.write_alias_table(aliases, args.out_dir / ha.ALIASES_PATH.name)
    print(f"[ok] {args.out_dir / ha.ALIASES_PATH.name} ({aliases.height} names, HGNC {ha.HGNC_RELEASE})")
    long = pl.concat([annotate(platform, table, updates, hgnc_ids=ids) for platform, table in tables.items()])
    long.write_csv(args.out_dir / ANNOTATION_PATH.name, separator="\t")
    for platform, (assay_col, path) in WIDE.items():
        wide = wide_annotation(long.filter(pl.col("Platform") == platform), assay_col)
        wide.write_csv(args.out_dir / path.name, separator="\t")
        print(f"[ok] {args.out_dir / path.name} ({wide.height} assays)")
    (args.out_dir / INPUTS_RECORD.name).write_text(
        json.dumps(_record(args.updates, args.out_dir / ha.ALIASES_PATH.name), indent=2) + "\n"
    )
    print(f"[ok] {args.out_dir / ANNOTATION_PATH.name} ({long.height} assay-accession rows)")
    with pl.Config(tbl_cols=-1):
        print(summary(long))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
