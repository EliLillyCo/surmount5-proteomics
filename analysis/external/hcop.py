"""HCOP human-mouse orthologs for CellMarker's mouse gene sets (stage ``hcop``).

    pixi run python analysis/external/hcop.py [--path <file>]

The CellMarker 2024 (Enrichr) mouse sets have members that are upper-cased mouse symbols (``PDGFRB``,
``CD59A``). The pathway step resolves each member through HGNC's HCOP orthology predictions
(``human_mouse_hcop_fifteen_column.txt.gz``; Eyre et al., Nucleic Acids Res 2007; Yates et al. 2021), read
here; ``fetch`` downloads it.

A member's orthologs are the human genes (HGNC ID, symbol) HCOP pairs with every mouse gene whose symbol,
upper-cased, is the member (``read_orthologs``). Then (``resolve_member``):

* ``KEPT``: the member is, ignoring case, the symbol of one of its human orthologs (PDGFRB);
* ``MAPPED``: it has exactly one human ortholog, of another symbol (CD59A -> CD59);
* dropped otherwise (no ortholog, or several and none of them its symbol: H2-K1 -> HLA-A, -B, ...).

A kept or mapped member takes our symbol for the ortholog's HGNC ID (the alias table), or HCOP's symbol for
an ID we do not measure (``mouse_translation``). The sets' minimum size is 5 measured genes.

HCOP's file is updated in place every month; the URL pins the generation the analysis used and the file's
SHA-256 is checked on every read. When that generation is gone, a reader with a saved copy passes ``--path``.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import polars as pl

from analysis.external.pinned import check_sha256, fetch_pinned

ROOT = Path(__file__).resolve().parents[2]
HCOP_URL = (
    "https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz"
    "?generation=1790858706025272"
)
# Release of 2026-10-01 (object generation above), 3,454,647 bytes.
HCOP_SHA256 = "0cfb78e4eb273751e0557e8230445a60e844bfaf958e32740d4a266fb41cdb35"
HCOP_PATH = ROOT / "data" / "external" / "hcop" / "human_mouse_hcop_fifteen_column.txt.gz"
KEPT, MAPPED = "kept", "mapped"
MISSING = "-"


def fetch(path: Path = HCOP_PATH) -> Path:
    """The verified HCOP file at ``path``, downloaded from ``HCOP_URL`` first when absent."""
    return fetch_pinned(HCOP_URL, path, HCOP_SHA256)


def read_orthologs(path: Path = HCOP_PATH) -> dict[str, dict[str, str]]:
    """Upper-cased mouse symbol -> {HGNC ID: human symbol} of its HCOP human orthologs.

    Raises:
        RuntimeError: When the file does not match ``HCOP_SHA256``.
    """
    check_sha256(path, HCOP_SHA256)
    with gzip.open(path, "rt") as fh:
        table = pl.read_csv(fh, separator="\t", infer_schema_length=0, quote_char=None)
    pairs = table.filter((pl.col("hgnc_id") != MISSING) & (pl.col("mouse_symbol") != MISSING)).select(
        pl.col("mouse_symbol").str.to_uppercase().alias("mouse"), "hgnc_id", "human_symbol"
    ).unique(maintain_order=True)
    out: dict[str, dict[str, str]] = {}
    for mouse, hgnc_id, human in pairs.iter_rows():
        out.setdefault(mouse, {})[hgnc_id] = human
    return out


def resolve_member(member: str, orthologs: dict[str, dict[str, str]]) -> tuple[str, str] | None:
    """``(HGNC ID, KEPT | MAPPED)`` of one mouse-set member, or None when it is dropped."""
    found = orthologs.get(member.upper(), {})
    same = sorted(i for i, s in found.items() if s.upper() == member.upper())
    if same:
        return same[0], KEPT
    if len(found) == 1:
        return next(iter(found)), MAPPED
    return None


def mouse_translation(
    orthologs: dict[str, dict[str, str]], id_symbol: dict[str, str], translation: dict[str, str]
) -> dict[str, str]:
    """Upper-cased mouse member -> our symbol, for every kept or mapped member (dropped ones are absent).

    Args:
        orthologs: ``read_orthologs``.
        id_symbol: HGNC ID -> our symbol (the alias table's ``symbol`` rows).
        translation: Upper-cased name -> our symbol (``hgnc_aliases.symbol_translation``), for an ortholog
            we do not measure.
    """
    out = {}
    for member in sorted(orthologs):
        resolved = resolve_member(member, orthologs)
        if resolved is None:
            continue
        hgnc_id, _ = resolved
        human = orthologs[member][hgnc_id]
        out[member] = id_symbol.get(hgnc_id) or translation.get(human.upper(), human)
    return out


def main(argv: list[str] | None = None) -> int:
    """Fetch (or verify) the pinned HCOP file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", type=Path, default=HCOP_PATH, help="where the file is cached (or a saved copy)")
    args = ap.parse_args(argv)
    path = fetch(args.path)
    print(f"[ok] {path} ({len(read_orthologs(path))} mouse symbols with a human ortholog)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
