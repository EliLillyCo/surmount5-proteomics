"""Fetch and parse the published cis-pQTL catalogs.

``build_catalogs`` downloads each publisher's table into ``data/external/pqtl/raw/``
(``fetch_pinned``: SHA-256 checked before it lands and on every reuse), parses it with
``analysis.external.pqtl_parsers`` and writes the parquets into
``analysis.qc_paths.PQTL_CATALOG_DIR``, the directory ``analysis.qc.pqtl_reference``
resolves the configs' catalog names against. The one table not downloaded is Fenland's:
the Science supplementary workbook is downloaded in a browser and read from ``FENLAND_XLSX``
(``data/vendor/``) or ``--fenland-xlsx``, SHA-256 checked (``FENLAND_XLSX_SHA256``) before any
download starts. Sources and licenses are listed in ``data/external/README.md``.

Each parsed frame must match its pinned content digest (``frame_sha256``, which a parquet
writer version does not change; ``CATALOG_FRAME_SHA256``), and a mismatch raises before any
parquet is written. For the Fenland catalog, the frame without ``eaf`` is checked against its
``CATALOG_FRAME_SHA256`` entry and the full frame against its ``FULL_FRAME_SHA256`` entry
(``EXTRA_COLUMNS`` names the column). The QC reads ``eaf`` only to orient palindromic SNPs
when run with ``palindromic = "rescue"``.

Some catalogs annotate their rows from others, so the build runs in this order: UKB-PPP,
JHS, Genes & Health Olink, Olink's Explore 3072 Fixed LOD file (a UniProt source only, not
written), MESA Olink (UniProt by OlinkID from UKB-PPP, the Explore 3072 LOD file, JHS), CKB Olink (UniProt and OlinkID by
gene symbol from UKB-PPP, MESA Olink, Genes & Health Olink); deCODE, ARIC, CKB SomaScan
(gene and UniProt by SeqId from deCODE, ARIC), MESA SomaScan (UniProt by SeqId from deCODE,
ARIC, CKB SomaScan), Genes & Health SomaScan, Fenland (lifted to hg38 and checked against
the trial PVARs), CKB Atlas, INTERVAL (lifted to hg38) and AGES (lifted to hg38; UniProt by SeqId from
the deCODE workbook). The three hg19 catalogs are lifted with Ensembl's assembly mapping
(``analysis.external.liftover``). The 2025 CKB SomaScan catalog supplies UniProt to MESA SomaScan; no QC config or table
selects it. Chromosomes are written as ``1``..``22`` and ``X``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.external.fenland_liftover import liftover_fenland, pvar_files
from analysis.external.liftover import Liftover, RsidLocator, fetch_ensembl_mapping
from analysis.external.pinned import check_sha256, fetch_pinned
from analysis.external.pqtl_parsers import (
    ckb_gene_annotation,
    ckb_seqid_annotation,
    decode_seqid_uniprot,
    parse_ages,
    parse_aric,
    parse_ckb_atlas,
    parse_ckb_olink,
    parse_ckb_soma,
    parse_fenland,
    parse_genes_health,
    parse_interval,
    parse_jhs_mesa,
    parse_mesa,
    parse_olink_3072_lod,
    uniprot_by_platform_id,
)
from analysis.qc.pqtl import normalize_chrom
from analysis.qc.pqtl_reference import parse_decode, parse_ukbppp

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external" / "pqtl" / "raw"

_SPRINGER = "https://static-content.springer.com/esm/"


@dataclass(frozen=True)
class RawFile:
    """A published file: where it is downloaded from, its local name and its SHA-256."""

    url: str
    name: str
    sha256: str


RAW_FILES: dict[str, RawFile] = {
    # Sun et al. 2023, Nature, doi:10.1038/s41586-023-06592-6, Supplementary Tables (ST9, ST11).
    "ukbppp": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41586-023-06592-6/MediaObjects/41586_2023_6592_MOESM3_ESM.xlsx",
        "41586_2023_6592_MOESM3_ESM.xlsx",
        "e174964b81e0d16f84e485593d4b349e9c24790f60983724d4f9f37681938cc7",
    ),
    # Tahir et al. 2024, JCI, doi:10.1172/JCI181802, Supplementary Data Tables 1-12 (ST2, ST3).
    "jhs": RawFile(
        "https://dm5migu4zj3pb.cloudfront.net/manuscripts/181000/181802/JCI181802.sdt1-12.xlsx",
        "JCI181802.sdt1-12.xlsx",
        "006857936601d2c872d86b52a361dddfcd90b3bf2f813e7b13fbcd1da9b2d442",
    ),
    # Pietzner et al. 2026, Nat Genet, doi:10.1038/s41588-026-02697-6, Supplementary Tables (ST2, ST3, ST14).
    "genes_health": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41588-026-02697-6/MediaObjects/41588_2026_2697_MOESM4_ESM.xlsx",
        "41588_2026_2697_MOESM4_ESM.xlsx",
        "8bd52a3c4bedc317f154247c3d19931ad8c1e1f2bbae8f3f13ff023860e299d2",
    ),
    # Nicholas et al. 2026, Nat Commun, doi:10.1038/s41467-025-67814-1; Zenodo record 17642645.
    "mesa_olink": RawFile(
        "https://zenodo.org/api/records/17642645/files/olink_cis_trans_pQTL_combined.tsv.gz/content",
        "olink_cis_trans_pQTL_combined.tsv.gz",
        "771b365d08e1515695ade53e35682e6bfb6b9e31c55a96dc9be0d48dacc2aa3a",
    ),
    "mesa_soma": RawFile(
        "https://zenodo.org/api/records/17642645/files/soma_cis_trans_pQTL_combined.tsv.gz/content",
        "soma_cis_trans_pQTL_combined.tsv.gz",
        "749391ee6f20b177fbf6b2743bf784ac3b8a2008090afc82757b61f2ba38f424",
    ),
    # Nat Commun 2025;16:1869, doi:10.1038/s41467-025-56935-2, Supplementary Data 4 (sheets Olink, SomaScan).
    "ckb": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41467-025-56935-2/MediaObjects/41467_2025_56935_MOESM7_ESM.xlsx",
        "41467_2025_56935_MOESM7_ESM.xlsx",
        "c6a6f4cfa58a2ee886a86482d7e08913c5c2aed4e42c3aa07f17c625d4fd6764",
    ),
    # Olink Explore 3072 Fixed LOD file (Olink, December 2024): OlinkID -> UniProt for the MESA Olink backfill.
    "olink_3072_lod": RawFile(
        "https://7074596.fs1.hubspotusercontent-na1.net/hubfs/7074596/000-documents/10-excel%20file/"
        "Explore%203072_Fixed%20LOD_2024-12-19.csv",
        "olink_explore_3072_fixed_lod_2024-12-19.csv",
        "0c9543ac6f49fa2c6ef4f098ddf03dd4f91829704d806e50f7ac0397d518840c",
    ),
    # Ferkingstad et al. 2021, Nat Genet, doi:10.1038/s41588-021-00978-w, Supplementary Tables (ST02).
    "decode": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41588-021-00978-w/MediaObjects/41588_2021_978_MOESM4_ESM.xlsx",
        "41588_2021_978_MOESM4_ESM.xlsx",
        "e3f907a11b452f979db8241ed852c1542d00ed5373c39b53c6a79702f8b80ef6",
    ),
    # Zhang et al. 2022, Nat Genet, doi:10.1038/s41588-022-01051-w, Supplementary Tables (ST3.1, ST3.2).
    "aric": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41588-022-01051-w/MediaObjects/41588_2022_1051_MOESM4_ESM.xlsx",
        "41588_2022_1051_MOESM4_ESM.xlsx",
        "473e0cb77d813b3e945a446827b1bb7cacab86e83bb10596fdea8837e9f5b02c",
    ),
    # Gudjonsson et al. 2022, Nat Commun, doi:10.1038/s41467-021-27850-z, Supplementary Data 3 (sheet Data 3).
    "ages": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41467-021-27850-z/MediaObjects/41467_2021_27850_MOESM5_ESM.xlsx",
        "41467_2021_27850_MOESM5_ESM.xlsx",
        "f2f4eafe8d3e74e447c5f6c5c0392f351c5c3e1539ea8892aadbc9d4d05f2dca",
    ),
    # Sun et al. 2018, Nature, doi:10.1038/s41586-018-0175-2, Supplementary Table 4 (sheet "ST4 - pQTL summary").
    "interval": RawFile(
        _SPRINGER + "art%3A10.1038%2Fs41586-018-0175-2/MediaObjects/41586_2018_175_MOESM4_ESM.xlsx",
        "41586_2018_175_MOESM4_ESM.xlsx",
        "d7bfada3eab1e2be15fdde9bc5a62e2d2563c111db5b2e7f242dfb9ea492b9d5",
    ),
    # CKB Atlas: Pozarickij A. et al., Science Bulletin (2026), doi:10.1016/j.scib.2026.10.060.
    # Zenodo record 21503247 (CC BY 4.0), doi:10.5281/zenodo.21503247: Supplementary Tables (ST2, ST3).
    "ckb_atlas": RawFile(
        "https://zenodo.org/api/records/21503247/files/Supplementary%20Tables.xlsx/content",
        "ckb_atlas_supplementary_tables.xlsx",
        "22aeba9e2111ca8bc2a9827e89c239cd9de02b1129ba254d1f36a580e8df0d3a",
    ),
}

# Pietzner et al. 2021, Science, doi:10.1126/science.abj1541, Supplementary Tables S1 to S9 (ST2):
# downloaded by hand from science.org (scripted requests are refused).
FENLAND_XLSX = ROOT / "data" / "vendor" / "science.abj1541_tables_s1_to_s9.xlsx"
FENLAND_XLSX_SHA256 = "4d242a5057510de09ff645469fd83414048a7e3fc7d52a6c94818c2dac2c5c51"

# ``frame_sha256`` of each parsed catalog. The parquet bytes depend on the writer's version;
# the content does not.
CATALOG_FRAME_SHA256: dict[str, str] = {
    "ukbppp_sun2023.parquet": "bbbe542fba03dd4f0e74ed41a78530920f4ec25219469f1b8a42602f0e9e5e35",
    "jhs_tahir2024.parquet": "9dcdb95dcaadeee079d7b13c34ef29b95052f4ee553eeb8e6ead397425f5bd35",
    "genes_health_olink_2026.parquet": "faa17d901d7110e84082afc026dc6b08e55f82afbd327fc2d43904c4e0bcc6de",
    "mesa_olink_2026.parquet": "78ca6a097b730bb6b7e704e144598c2fd2ebce1bd385d903a604e5c0ff18ce20",
    "ckb_olink_2025.parquet": "9aad3da83564451d6a0989e87ff81d5a59f216654b813ac18ac92001bd56dc82",
    "decode_ferkingstad2021.parquet": "70f46d61a518c4f497b75bfdac48b360aa5f4c58fb9b9b84dc21c861eccc3b1f",
    "aric_zhang2022.parquet": "d35c1ab7fc846a60a10c72d18a172d072960b74938809fe4ce64327c0846cd17",
    "ckb_soma_2025.parquet": "9af2b3ac37fad5c74523c4c351347613612ac7512b521f64619706a835ca3483",
    "ckb_soma_2026.parquet": "87d7cb0e903ddd57cc1232ed799c920146c69b9c3bee18abfdaa9c39d62f9eb1",
    "ages_gudjonsson2022.parquet": "3b06c60047ddcf1a31c78451b0c6fad06d9c8f4ec6b666fe611605ccd3a73de7",
    "interval_sun2018_hg38.parquet": "630f14d215e69e24eacae4f5578ce3f8b99852135138ced455c8205243618905",
    "mesa_soma_2026.parquet": "adbf8d8475c6496d0e5729a27c1f14a0d5088ae19a19445554a61be9f2ce30c2",
    "genes_health_soma_2026.parquet": "3b47d0e5c5c3fb7525a098ee90d4c42b355b6cf89a0c897bfea2080e04a7fa23",
    "fenland_pietzner2021_hg38.parquet": "b652acaf0c2d94166bb42e25a0b7b3b4627b6ff60d7fab7c6b016bf08d3746dd",
}

# Columns checked separately from ``CATALOG_FRAME_SHA256``: that digest is checked on the frame
# without them, ``FULL_FRAME_SHA256`` on the frame with them.
EXTRA_COLUMNS: dict[str, tuple[str, ...]] = {
    "fenland_pietzner2021_hg38.parquet": ("eaf",),
}
FULL_FRAME_SHA256: dict[str, str] = {
    "fenland_pietzner2021_hg38.parquet": "2ac0929154e2c6054be994d7dbe8996156e03afe73b2f0e0cfd507534ee557fb",
}


def frame_sha256(df: pl.DataFrame) -> str:
    """SHA-256 of a frame's content: column names, dtypes, null masks and value bytes, in order.

    Two frames have the same digest exactly when they hold the same columns, dtypes and
    values (floats compared bit for bit), whichever writer stored them.
    """
    h = hashlib.sha256(f"{df.height}\x1d".encode())
    for s in df.get_columns():
        h.update(f"{s.name}\x1f{s.dtype}\x1e".encode())
        h.update(s.is_null().to_numpy().tobytes())
        if s.dtype.is_numeric():
            h.update(s.fill_null(0).to_numpy().tobytes())
        else:
            h.update("\x1f".join(s.cast(pl.String).fill_null("").to_list()).encode())
        h.update(b"\x1d")
    return h.hexdigest()


def normalize_chrom_column(df: pl.DataFrame) -> pl.DataFrame:
    """``chrom`` recoded to ``1``..``22``, ``X`` (``chr`` dropped, ``23`` -> ``X``)."""
    return df.with_columns(pl.col("chrom").map_elements(normalize_chrom, return_dtype=pl.String))


def parse_catalogs(
    raw: dict[str, Path], pvars: dict[str, list[Path]], liftover: Liftover, rsids: RsidLocator
) -> dict[str, pl.DataFrame]:
    """Parse the fourteen catalogs, in the build order of the module docstring.

    Args:
        raw: ``RAW_FILES`` key -> the verified local file, and ``"fenland"`` -> the verified
            Science workbook (``FENLAND_XLSX``).
        pvars: Chromosome -> trial PVARs (``pvar_files``), for the Fenland allele check.
        liftover: The hg19 -> hg38 chain (``analysis.external.liftover``).
        rsids: The rsID lookup for variants the chain does not lift.

    Returns:
        Parquet file name (``analysis.qc.pqtl_reference.CATALOG_NAMES`` values) -> frame.
    """
    norm = normalize_chrom_column
    out: dict[str, pl.DataFrame] = {}
    out["ukbppp_sun2023.parquet"] = ukbppp = norm(parse_ukbppp(raw["ukbppp"]))
    out["jhs_tahir2024.parquet"] = jhs = norm(parse_jhs_mesa(raw["jhs"]))
    out["genes_health_olink_2026.parquet"] = gh_olink = norm(parse_genes_health(raw["genes_health"], "Olink"))
    olink_3072 = parse_olink_3072_lod(raw["olink_3072_lod"])
    out["mesa_olink_2026.parquet"] = mesa_olink = norm(
        parse_mesa(
            raw["mesa_olink"],
            platform="olink",
            source="mesa_olink",
            uniprot=uniprot_by_platform_id([ukbppp, olink_3072, jhs]),
        )
    )
    out["ckb_olink_2025.parquet"] = norm(
        parse_ckb_olink(raw["ckb"], ckb_gene_annotation([ukbppp, mesa_olink, gh_olink]))
    )
    out["decode_ferkingstad2021.parquet"] = decode = norm(parse_decode(raw["decode"]))
    out["aric_zhang2022.parquet"] = aric = norm(parse_aric(raw["aric"]))
    out["ckb_soma_2025.parquet"] = ckb_soma = norm(parse_ckb_soma(raw["ckb"], ckb_seqid_annotation([decode, aric])))
    out["mesa_soma_2026.parquet"] = norm(
        parse_mesa(
            raw["mesa_soma"],
            platform="soma",
            source="mesa_soma",
            uniprot=uniprot_by_platform_id([decode, aric, ckb_soma]),
        )
    )
    out["genes_health_soma_2026.parquet"] = norm(parse_genes_health(raw["genes_health"], "SomaLogic"))
    # ``eaf`` (the frequency of ``a1``) passes through the liftover unchanged: the lift moves
    # ``chrom`` and ``pos`` and leaves ``a1`` and ``a2`` as published.
    out["fenland_pietzner2021_hg38.parquet"] = norm(
        liftover_fenland(norm(parse_fenland(raw["fenland"])), pvars, liftover, rsids)
    )
    print(f"[ok] lifted fenland_pietzner2021_hg38.parquet ({out['fenland_pietzner2021_hg38.parquet'].height} rows)")
    out["ckb_soma_2026.parquet"] = norm(parse_ckb_atlas(raw["ckb_atlas"]))
    out["interval_sun2018_hg38.parquet"] = norm(parse_interval(raw["interval"], liftover, rsids))
    print(f"[ok] lifted interval_sun2018_hg38.parquet ({out['interval_sun2018_hg38.parquet'].height} rows)")
    out["ages_gudjonsson2022.parquet"] = norm(
        parse_ages(raw["ages"], decode_seqid_uniprot(raw["decode"]), liftover, rsids)
    )
    print(f"[ok] lifted ages_gudjonsson2022.parquet ({out['ages_gudjonsson2022.parquet'].height} rows)")
    return out


def write_verified(
    frames: dict[str, pl.DataFrame],
    parsed_dir: Path,
    pins: dict[str, str],
    extra_columns: dict[str, tuple[str, ...]] | None = None,
    full_pins: dict[str, str] | None = None,
) -> list[Path]:
    """Write each frame as ``parsed_dir/<name>`` once every frame matches its pinned digest.

    A frame is checked against ``pins`` without the columns ``extra_columns`` names for it
    and, when it has such columns, against ``full_pins`` as a whole.

    Raises:
        RuntimeError: Naming each frame whose digest differs from ``pins`` or ``full_pins``
            (or has none); nothing is written then.
    """
    extra_columns, full_pins = extra_columns or {}, full_pins or {}
    bad = []
    for name, df in frames.items():
        extra = extra_columns.get(name, ())
        digest = frame_sha256(df.drop(extra, strict=False))
        if digest != pins.get(name):
            bad.append(f"{name}: {digest} (pinned {pins.get(name)})")
        if extra and frame_sha256(df) != full_pins.get(name):
            bad.append(f"{name} with {', '.join(extra)}: {frame_sha256(df)} (pinned {full_pins.get(name)})")
    if bad:
        raise RuntimeError("parsed catalogs differ from their pinned digests:\n  " + "\n  ".join(bad))
    parsed_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in frames.items():
        path = parsed_dir / name
        df.write_parquet(path)
        print(f"[ok] {path} ({df.height} rows)")
        written.append(path)
    return written


def build_catalogs(
    parsed_dir: Path, pvar_dir: Path, pvar_pattern: str, raw_dir: Path = RAW_DIR, fenland_xlsx: Path = FENLAND_XLSX
) -> list[Path]:
    """Fetch the published tables, parse them, and write the parquets once every digest matches its pin.

    Args:
        parsed_dir: Where the parquets are written.
        pvar_dir: The trial's PGEN directory.
        pvar_pattern: The PVAR naming family the Fenland check reads (``pvar_files``).
        raw_dir: Where the downloads are kept.
        fenland_xlsx: The Fenland Science supplementary workbook (manual download).

    Returns:
        The parquet paths.

    Raises:
        RuntimeError: When ``fenland_xlsx`` does not match ``FENLAND_XLSX_SHA256`` (checked
            before anything is downloaded).
    """
    check_sha256(fenland_xlsx, FENLAND_XLSX_SHA256)
    raw = {key: fetch_pinned(f.url, raw_dir / f.name, f.sha256) for key, f in RAW_FILES.items()}
    raw["fenland"] = fenland_xlsx
    liftover = Liftover(fetch_ensembl_mapping())
    frames = parse_catalogs(raw, pvar_files(pvar_dir, pvar_pattern), liftover, RsidLocator())
    return write_verified(frames, parsed_dir, CATALOG_FRAME_SHA256, EXTRA_COLUMNS, FULL_FRAME_SHA256)
