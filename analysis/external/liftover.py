"""Lift hg19 (GRCh37) catalog positions to hg38 (GRCh38) with Ensembl's assembly mapping.

A position is lifted through Ensembl's ``GRCh37_to_GRCh38.chain.gz`` (``Liftover``). A variant the
chain does not cover is placed by its rsID in Ensembl's GRCh38 variation release
(``RsidLocator``), and takes the strand of the nearest aligned block. ``lift`` applies the two in
that order; a variant neither places has no hg38 position.

The chain and the variation indexes are Ensembl data, downloaded into ``data/external/ensembl_mapping/``
(not tracked) and SHA-256 pinned (``CHAIN_SHA256``, ``VARIATION_INDEX_SHA256``). The variation files
are read from release ``ENSEMBL_RELEASE``.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam

from analysis.external.pinned import fetch_pinned

ROOT = Path(__file__).resolve().parents[2]
MAPPING_DIR = ROOT / "data" / "external" / "ensembl_mapping"

ENSEMBL_RELEASE = 116
# The chain's URL is unversioned; the digest fixes its content.
CHAIN_URL = "https://ftp.ensembl.org/pub/assembly_mapping/homo_sapiens/GRCh37_to_GRCh38.chain.gz"
CHAIN_NAME = "GRCh37_to_GRCh38.chain.gz"
CHAIN_SHA256 = "351de3cd4a01d9fcffd38881981767b697090d2eba876740891b96d5c546b100"
VARIATION_URL = (
    f"https://ftp.ensembl.org/pub/release-{ENSEMBL_RELEASE}/variation/vcf/homo_sapiens/homo_sapiens-chr{{chrom}}.vcf.gz"
)
# How far on either side of the estimated hg38 position the rsID search looks.
RSID_WINDOW = 5_000_000


def fetch_ensembl_mapping(directory: Path = MAPPING_DIR) -> Path:
    """The verified Ensembl GRCh37-to-GRCh38 chain, downloaded into ``directory`` when absent."""
    return fetch_pinned(CHAIN_URL, directory / CHAIN_NAME, CHAIN_SHA256)


def _chr_name(name: str) -> str:
    """An Ensembl sequence name with the ``chr`` prefix the catalog parsers use (``MT`` is ``chrM``)."""
    if name.startswith("chr"):
        return name
    return "chrM" if name == "MT" else f"chr{name}"


class Liftover:
    """A point liftover through a chain file.

    The base must fall inside an aligned block; when several chains cover it, the
    highest-scoring chain wins and file order breaks ties. Sequence names are read with a
    ``chr`` prefix (``1`` becomes ``chr1``).

    Args:
        chain_path: Gzipped chain file.
    """

    def __init__(self, chain_path: str | Path) -> None:
        blocks: dict[str, list[tuple]] = {}
        with gzip.open(chain_path, "rt") as fh:
            header: list[str] = []
            t = q = order = 0
            for line in fh:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "chain":
                    header = fields
                    header[2], header[7] = _chr_name(header[2]), _chr_name(header[7])
                    t, q = int(fields[5]), int(fields[10])
                    order += 1
                    continue
                size = int(fields[0])
                blocks.setdefault(header[2], []).append(
                    (t, t + size, q, float(header[1]), order, header[7], int(header[8]), header[9])
                )
                if len(fields) == 3:
                    t += size + int(fields[1])
                    q += size + int(fields[2])
        self._chroms: dict[str, dict] = {}
        for chrom, rows in blocks.items():
            arr = list(zip(*rows, strict=True))
            self._chroms[chrom] = {
                "t_start": np.asarray(arr[0], dtype=np.int64),
                "t_end": np.asarray(arr[1], dtype=np.int64),
                "q_start": np.asarray(arr[2], dtype=np.int64),
                "score": np.asarray(arr[3], dtype=np.float64),
                "order": np.asarray(arr[4], dtype=np.int64),
                "q_name": list(arr[5]),
                "q_size": np.asarray(arr[6], dtype=np.int64),
                "q_strand": list(arr[7]),
            }

    def convert(self, chrom: str, pos: int) -> tuple[str, int, str] | None:
        """Lifts a 1-based position.

        Returns:
            ``(chrom, pos, strand)``, or None when the base is in no aligned block.
        """
        c = self._chroms.get(chrom)
        if c is None:
            return None
        p = pos - 1
        hits = np.flatnonzero((c["t_start"] <= p) & (p < c["t_end"]))
        if hits.size == 0:
            return None
        best = hits[np.lexsort((c["order"][hits], -c["score"][hits]))[0]]
        q0 = int(c["q_start"][best] + (p - c["t_start"][best]))
        strand = c["q_strand"][best]
        if strand == "-":
            q0 = int(c["q_size"][best]) - 1 - q0
        return c["q_name"][best], q0 + 1, strand

    def nearest(self, chrom: str, pos: int) -> tuple[int, str] | None:
        """The estimated hg38 position and strand from the aligned block nearest a 1-based position.

        Only blocks that map to the same chromosome are used. The estimate extends the block's
        offset to the position.

        Returns:
            ``(pos, strand)``, or None when no block maps ``chrom`` to itself.
        """
        c = self._chroms.get(chrom)
        if c is None:
            return None
        same = np.array([name == chrom for name in c["q_name"]])
        if not same.any():
            return None
        p = pos - 1
        gap = np.where(p < c["t_start"], c["t_start"] - p, np.where(p >= c["t_end"], p - c["t_end"] + 1, 0))
        gap = np.where(same, gap, np.iinfo(np.int64).max)
        best = int(np.lexsort((c["order"], -c["score"], gap))[0])
        strand = c["q_strand"][best]
        q0 = int(c["q_start"][best] + (p - c["t_start"][best]))
        if strand == "-":
            q0 = int(c["q_size"][best]) - 1 - q0
        return q0 + 1, strand


# SHA-256 of each chromosome's index (``<file>.vcf.gz.csi``) in release ``ENSEMBL_RELEASE``.
VARIATION_INDEX_SHA256: dict[str, str] = {
    "1": "df0f66da4dfc7c8a8174929e005ce5ccf381bb47887263e442e269657939c463",
    "2": "fc53231a5b57178bbba529b757bc4ba4b6a0358fb0732da2cee26c9b03ee2a51",
    "3": "c887911f4edf42c5ee0fbb037fa6b194cd5e8955f999a817846b9c7ea2bd18d7",
    "4": "e3d2dcddc950c034f677f339cb2f39bcd10cec1d0e89cbbebc26d3727143d613",
    "5": "492fadfe6cf7dd805f7a7b5a72d19478f93dc8ef52cc43b1d7f939e95586039f",
    "6": "080d435eff2bf5ab999817e12e5d83b1f7d00b86e6ddccb4307ab28d29bdff3e",
    "7": "f26280030790bea1ccfecff95af065349bada5032f663c0d4fa26bac86d331d1",
    "8": "5bb8a7f4ddb0ad8cc929c859234cde489a014b66643f2e4995b4d15e458fe180",
    "9": "cb358f05cc560f5de1c34daf362dba79dca39a727c9c204ede05cdc61208696d",
    "10": "e73dd834ceea8ad1887e1548e47dd0518bb4b07ad8c37e1472c3c281158df2e0",
    "11": "65eb40d5a3dbd5d1c23501d25ab615e0a5449de75b0a0dd4a3d1239159ed3af7",
    "12": "6a75c7354c005dea9d50b3d7b168a0fc2d5d6533097220e176afcd4d2e37fd01",
    "13": "ed49ff17449ab41c1fbe5991c04f5c29620b4787edc863b5bfcc3554d4101e65",
    "14": "a4e922c15b1d5cbba9cdd9dc55e5aa622391310efba924578933993588d6cd82",
    "15": "c8dd0f70338ee8c40b7b1464cf462f83391bb14be66cf2c9bd4779a0df6f38cc",
    "16": "4a4feb7bec2945157dbd7746c205d007f7a35a2d856a0f994f518ccf2e41ec84",
    "17": "5cac0cffd638b64b75dfa7452e9c70f934f46f69ba75c324840b31354592146c",
    "18": "d7282e24672e2ce59b26fcd7a4528425f4cded10a2b4654226e42775ba51b74f",
    "19": "d0e649a0f99b8569e26c3c589e6f27be5d1662567a6102787f1b9d683389c2ea",
    "20": "91a5b2026a7b44b9586d0afe765ff057c7b5bb906d546dddb5dce6954398dde4",
    "21": "b80a79c95f84b743763c7c47491cdfa8b4c49cd1f99d639af90125a36e240594",
    "22": "0803709ee7c13c2a9e504651cf9f414d8abd73116ec62cc1a31b89161edc0c56",
    "X": "4d5640f235b29529b87c0a595d1bfefe759774a9670723a0e204e5f91a5d1cd2",
    "Y": "622864877fe6f849ad200235eacec3e03658fb147608a000c72d89578f153ae6",
}


class RsidLocator:
    """GRCh38 positions of rsIDs in Ensembl's variation release (``VARIATION_URL``).

    Each chromosome's file is read over HTTPS through its index. The index is downloaded into
    ``index_dir`` once and checked against ``VARIATION_INDEX_SHA256``.

    Args:
        index_dir: Where the per-chromosome index files are kept.
    """

    def __init__(self, index_dir: Path = MAPPING_DIR) -> None:
        self._index_dir = index_dir
        self._files: dict[str, pysam.VariantFile] = {}

    def _file(self, chrom: str) -> pysam.VariantFile:
        if chrom not in self._files:
            url = VARIATION_URL.format(chrom=chrom)
            index = fetch_pinned(
                url + ".csi", self._index_dir / f"homo_sapiens-chr{chrom}.vcf.gz.csi", VARIATION_INDEX_SHA256[chrom]
            )
            self._files[chrom] = pysam.VariantFile(url, index_filename=str(index))
        return self._files[chrom]

    def locate(self, rsid: str, chrom: str, near: int) -> int | None:
        """The GRCh38 position of ``rsid`` on ``chrom`` within ``RSID_WINDOW`` of ``near``.

        The search starts at ``near`` and widens tenfold from 10 kb until the rsID is found.

        Returns:
            The 1-based position, or None when the rsID is not found there or is found at more
            than one position.
        """
        width = 10_000
        while True:
            found = {
                rec.pos
                for rec in self._file(chrom).fetch(chrom, max(0, near - width), near + width)
                if rec.id and rsid in rec.id.split(";")
            }
            if found or width >= RSID_WINDOW:
                return found.pop() if len(found) == 1 else None
            width = min(width * 10, RSID_WINDOW)


@dataclass(frozen=True)
class Lifted:
    """A lifted position: hg38 ``pos``, the ``strand`` of the mapping and its ``method`` (``chain`` or ``rsid``)."""

    pos: int
    strand: str
    method: str


def lift(liftover: Liftover, rsids: RsidLocator, chrom: str, pos: int, rsid: str | None) -> Lifted | None:
    """Lifts an hg19 variant to hg38 on the same chromosome: by the chain, else by its rsID.

    Args:
        liftover: The chain.
        rsids: The rsID lookup.
        chrom: Chromosome without the ``chr`` prefix.
        pos: 1-based hg19 position.
        rsid: The variant's rsID, or None.

    Returns:
        The hg38 position, or None when the chain does not place it on ``chrom`` and the rsID
        does not either.
    """
    hit = liftover.convert(f"chr{chrom}", pos)
    if hit is not None and hit[0] == f"chr{chrom}":
        return Lifted(hit[1], hit[2], "chain")
    if not rsid or not rsid.startswith("rs"):
        return None
    estimate = liftover.nearest(f"chr{chrom}", pos)
    if estimate is None:
        return None
    found = rsids.locate(rsid, chrom, estimate[0])
    return Lifted(found, estimate[1], "rsid") if found is not None else None
