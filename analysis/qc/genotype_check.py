"""Checks that the PGEN genotype directory is complete.

A run with pQTL enabled calls ``check_genotype_dir`` first and stops on a missing or
empty file.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.qc.pqtl import normalize_chrom, pgen_prefixes, read_psam


def check_genotype_dir(
    genotype_dir: str | Path, expected_chromosomes, pgen_pattern: str | None = None
) -> None:
    """Checks that every expected chromosome has intact PGEN, PVAR and PSAM files.

    Checks, on the naming family ``pqtl.pgen_prefixes`` selects: every expected
    chromosome is present; each PGEN prefix has a non-empty ``.pgen``, ``.pvar`` and
    ``.psam``; and all PSAM files list the same samples.

    Args:
        genotype_dir: The directory of the PGEN files.
        expected_chromosomes: The chromosomes the run needs.
        pgen_pattern: PGEN family to check (``pqtl.pgen_prefixes``).

    Raises:
        FileNotFoundError: If a chromosome, or a PVAR or PSAM beside a PGEN, is
            missing.
        ValueError: If a file is empty, or the PSAM sample lists differ.
    """
    directory = Path(genotype_dir).expanduser()
    prefixes = pgen_prefixes(directory, pgen_pattern)
    expected = [normalize_chrom(c) for c in expected_chromosomes]
    absent = [c for c in expected if c not in prefixes]
    if absent:
        raise FileNotFoundError(
            f"{directory} lacks PGEN files for chromosome(s) {', '.join(absent)}; "
            "the genotype directory is incomplete"
        )
    missing, empty = [], []
    for chrom in expected:
        for prefix in prefixes[chrom]:
            for ext in ("pgen", "pvar", "psam"):
                path = Path(f"{prefix}.{ext}")
                if not path.is_file():
                    missing.append(path.name)
                elif path.stat().st_size == 0:
                    empty.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"{directory} lacks {len(missing)} file(s): {', '.join(missing[:5])}"
        )
    if empty:
        raise ValueError(f"{directory} has {len(empty)} empty file(s): {', '.join(empty[:5])}")
    reference: pl.Series | None = None
    for chrom in expected:
        for prefix in prefixes[chrom]:
            iids = read_psam(prefix)["IID"].sort()
            if reference is None:
                reference = iids
            elif not iids.equals(reference):
                raise ValueError(
                    f"{Path(prefix).name}.psam lists different samples from the other PSAM files "
                    f"({len(iids)} against {len(reference)})"
                )
