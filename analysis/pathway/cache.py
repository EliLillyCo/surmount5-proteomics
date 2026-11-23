"""Gene-set cache: reads the pinned Enrichr libraries from disk, downloading a missing one on request."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import requests

from analysis.external.hgnc_aliases import translate

from .constants import CACHE_DIR, GENE_SET_SHA256, UNTRANSLATED_SETS
from .verify import verify_bytes, verify_sha256

logger = logging.getLogger(__name__)

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
CORRECTIONS_PATH = Path(__file__).resolve().parents[1] / "reference" / "gene_set_corrections.tsv"


def read_corrections(path: Path = CORRECTIONS_PATH) -> pl.DataFrame:
    """The tracked member corrections, every column as text."""
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def apply_corrections(name: str, lib: dict[str, list[str]], corrections: pl.DataFrame) -> dict[str, list[str]]:
    """The library with the corrections naming it applied, before any symbol translation.

    Each row replaces ``from_symbol`` by ``to_symbol`` in its term, in place.  A term absent from the
    library, a ``from_symbol`` not in the term or a ``to_symbol`` already in it raises, so a library
    update that fixes or moves the member surfaces.  Corrections for other libraries are ignored.
    """
    out = {term: list(members) for term, members in lib.items()}
    for row in corrections.filter(pl.col("library") == name).iter_rows(named=True):
        term, old, new = row["term"], row["from_symbol"], row["to_symbol"]
        if term not in out:
            raise KeyError(f"Gene-set correction: term '{term}' is not in library '{name}'")
        if old not in out[term]:
            raise ValueError(f"Gene-set correction: '{old}' is not a member of '{term}' in '{name}'")
        if new in out[term]:
            raise ValueError(f"Gene-set correction: '{new}' is already a member of '{term}' in '{name}'")
        out[term] = [new if m == old else m for m in out[term]]
    return out


def _download_enrichr_library(name: str) -> dict:
    """Download a gene-set library from Enrichr, returning {set_name: [genes]}.

    The response is read as UTF-8.
    """
    url = f"{ENRICHR_URL}/geneSetLibrary?mode=text&libraryName={name}"
    response = requests.get(url, timeout=60, stream=True)
    if not response.ok:
        raise RuntimeError(
            f"Failed to download gene set library '{name}' from Enrichr "
            f"(HTTP {response.status_code})"
        )
    response.encoding = "utf-8"
    genesets: dict = {}
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        parts = line.strip().split("\t")
        set_name = parts[0]
        # Column 1 is description (often empty); genes start at column 2
        genes = [g.split(",")[0] for g in parts[2:] if g]
        genesets[set_name] = genes
    return genesets


def translate_library(
    name: str, lib: dict[str, list[str]], translation: dict[str, str], mouse: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """The library with each set's members renamed to our symbols (``hgnc_aliases.translate``), each once.

    Sets whose name ends with ``UNTRANSLATED_SETS[name]`` (CellMarker's mouse sets) never go through the
    HGNC aliases: with ``mouse`` (upper-cased member -> our symbol, ``hcop.mouse_translation``) each member
    takes its HCOP-verified human ortholog and a member without one is dropped; without it they are
    kept as they are.
    """
    suffix = UNTRANSLATED_SETS.get(name)
    out = {}
    for term, members in lib.items():
        if suffix and term.endswith(suffix):
            out[term] = (
                list(members)
                if mouse is None
                else list(dict.fromkeys(mouse[m.upper()] for m in members if m.upper() in mouse))
            )
        else:
            out[term] = translate(members, translation)
    return out


class GeneSetCache:
    """The pinned gene-set libraries, read from disk and kept in memory.

    Reads ``<library>.json`` from *cache_dir* (default ``CACHE_DIR``,
    ``data/external/gene_sets/``, which the gene_sets stage fills); with
    ``download_missing`` an absent library is downloaded from Enrichr and saved
    there.  A loaded library is kept in memory for the life of the object.

    Every library is SHA-256 verified against *pins* (default
    ``GENE_SET_SHA256``) before use, from disk and after a download (before it
    is written).  An unpinned name or a digest mismatch raises.  A fresh
    Enrichr download passes only if Enrichr still serves the pinned content.

    With *translation* (name -> our symbol, ``hgnc_aliases.symbol_translation``), every
    library is translated after its pin is checked (``translate_library``); the pathway
    CLI passes the alias table built with the annotation's, and *mouse* (HCOP-verified
    orthologs) for CellMarker's mouse sets.

    *corrections* (default ``gene_set_corrections.tsv``) replace a member symbol in named
    terms of the verified library before translation; the pinned files are never modified.
    """

    DEFAULT_DIR = CACHE_DIR

    def __init__(
        self,
        cache_dir: Path | None = None,
        pins: dict[str, str] | None = None,
        translation: dict[str, str] | None = None,
        mouse: dict[str, str] | None = None,
        corrections: pl.DataFrame | None = None,
    ):
        self.corrections = read_corrections() if corrections is None else corrections
        self._pins = GENE_SET_SHA256 if pins is None else pins
        self.translation = translation
        self.mouse = mouse
        self._mem: dict[str, dict] = {}
        self._dir = cache_dir or self.DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self._dir / f"{safe}.json"

    def _pin(self, name: str) -> str:
        if name not in self._pins:
            raise KeyError(f"No pinned SHA-256 for gene-set library '{name}'")
        return self._pins[name]

    def get(self, name: str, *, download_missing: bool = False) -> dict:
        """Return a cached gene-set library, optionally downloading it on demand."""
        # 1. memory
        if name in self._mem:
            return self._mem[name]

        # 2. disk
        path = self._disk_path(name)
        if path.exists():
            verify_sha256(path, self._pin(name))
            logger.info(f"Loading gene set library from cache: {name}")
            with open(path) as f:
                lib = self._translated(name, apply_corrections(name, json.load(f), self.corrections))
            self._mem[name] = lib
            logger.info(f"  -> {len(lib)} sets loaded from {path.name}")
            return lib

        if not download_missing:
            raise FileNotFoundError(
                f"Gene-set library '{name}' not found at {path}. "
                "Run `pixi run stage run gene_sets` to fetch the pinned libraries, "
                "or pass --download-missing."
            )

        # 3. download & persist
        logger.info(f"Downloading gene set library: {name}")
        lib = _download_enrichr_library(name)
        logger.info(f"  -> {len(lib)} sets downloaded")
        payload = json.dumps(lib)
        verify_bytes(payload.encode(), self._pin(name), f"downloaded Enrichr library {name}")
        path.write_text(payload)
        logger.info(f"  -> cached to {path}")
        lib = self._translated(name, apply_corrections(name, lib, self.corrections))
        self._mem[name] = lib
        return lib

    def _translated(self, name: str, lib: dict) -> dict:
        if self.translation is None:
            return lib
        return translate_library(name, lib, self.translation, self.mouse)

    def prefetch(self, names: list[str], *, download_missing: bool = False):
        """Ensure all requested libraries are available."""
        for name in names:
            self.get(name, download_missing=download_missing)
