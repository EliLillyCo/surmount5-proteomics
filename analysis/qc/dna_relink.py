"""The ``dna_relink`` correction: it moves a genome to the participant it belongs to.

A ``dna_relink`` entry names the participant a genome is filed under
(``from_usubjid_sha256``) and the participant it belongs to (``usubjid_sha256``), each
by the SHA-256 of the USUBJID. Applying it files the genome under its owner; the
participant it was filed under is left with no genome. It is a mapping change only: no
genotype, tube or value changes.

The entry is subject-level, so it is written in the corrections TOML of both platform
runs. The pQTL identity check applies it to the genotype labels
(``pqtl_reference.pqtl_inputs``), and the pharmacogenomics stage to the pheno-geno
linking table (``analysis.pharmacogenomics.run``); both resolve it with ``relink_map``.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.qc.corrections import resolve_digest

# config imports pqtl_reference, which imports this module.
if TYPE_CHECKING:
    from analysis.qc.config import Correction

DNA_RELINK = "dna_relink"


def read_corrections(path: str | Path) -> tuple[Correction, ...]:
    """Parses the ``[[corrections]]`` entries of a corrections TOML (a ``<study>_qc_corrections`` path).

    Args:
        path: The TOML file; a file with no entries gives an empty tuple.

    Returns:
        The validated entries, every type included.
    """
    from analysis.qc.config import _parse_corrections

    return _parse_corrections(tomllib.loads(Path(path).read_text()).get("corrections", []))


def relink_map(
    corrections: Iterable[Correction], genotyped: Iterable[str], subjects: Iterable[str]
) -> dict[str, str]:
    """Resolves the ``dna_relink`` entries to ``{USUBJID filed under: owner USUBJID}``.

    Args:
        corrections: The run's corrections; other types are ignored.
        genotyped: The USUBJIDs genomes are filed under (the PSAM IIDs, or the linking
            table's subjects); ``from_usubjid_sha256`` is resolved against them.
        subjects: The study's USUBJIDs (the tube-labeled subjects and ADSL);
            ``usubjid_sha256`` is resolved against them.

    Returns:
        The mapping; empty when there is no ``dna_relink`` entry.

    Raises:
        ValueError: If a digest matches zero or several IDs, two entries move one genome
            or give one participant two genomes, or an owner already has a genome filed
            under them (a stale correction, or a second genome for one participant; two
            genomes exchanged are refused too). Errors never name an ID.
    """
    entries = [c for c in corrections if c.type == DNA_RELINK]
    if not entries:
        return {}
    genotyped = sorted(set(genotyped))
    subjects = sorted(set(subjects))
    pairs = [
        (
            resolve_digest(
                e.from_usubjid_sha256,
                genotyped,
                "from_usubjid_sha256 among the genotyped participants",
            ),
            resolve_digest(e.usubjid_sha256, subjects, "usubjid_sha256 in ADSL"),
        )
        for e in entries
    ]
    for side, what in (
        (0, "move the genome filed under one participant"),
        (1, "give one participant two genomes"),
    ):
        repeated = max(Counter(p[side] for p in pairs).values())
        if repeated > 1:
            raise ValueError(f"{repeated} dna_relink entries {what}")
    mapping = dict(pairs)
    clash = set(genotyped) & set(mapping.values())
    if clash:
        raise ValueError(
            f"a dna_relink owner already has a genome filed under them ({len(clash)} owner(s)); "
            "a stale correction, or a second genome for one participant"
        )
    return mapping


def relabel(labels: list[str], mapping: Mapping[str, str]) -> list[str]:
    """Returns a new list: ``labels`` with each relinked genome's label replaced by its owner.

    The replacement is positional (the genome keeps its position, so the KING table, PCs and
    genotype columns stay aligned); ``labels`` itself is not modified.
    """
    return [mapping.get(u, u) for u in labels]
