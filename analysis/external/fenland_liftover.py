"""Lift the Fenland catalog from hg19 to hg38 and check its alleles against the trial PVARs.

Positions are lifted by ``analysis.external.liftover.lift``; rows that do not lift are dropped.
Each lifted row then gets the trial PVAR's ``REF``/``ALT`` at its hg38 position and
``allele_check``: ``match`` when {a1, a2} equals {REF, ALT}, ``mismatch`` otherwise,
``no_pvar_variant`` when the PVAR has no row there. The QC scores with the catalog columns
only; the PVAR columns record the check.

The lift changes ``chrom`` and ``pos`` and nothing else in the catalog columns: ``a1``, ``a2``
and ``eaf`` (the frequency of ``a1``) are carried as published, whatever the PVAR's ``REF``/``ALT``
order or the mapping's strand, so ``eaf`` never needs ``1 - eaf`` here. A row whose alleles do not
fit the PVAR is marked ``mismatch``, not rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from analysis.external.liftover import Liftover, RsidLocator, lift
from analysis.qc.pqtl_reference import CATALOG_SCHEMA


def pvar_files(pvar_dir: Path, pattern: str) -> dict[str, list[Path]]:
    """Chromosome -> the PVARs in ``pvar_dir`` of one naming family, sorted by name.

    ``pattern`` is a file name without ``.pvar`` in which ``{chrom}`` is the chromosome and
    any other ``{field}`` (a chunk start or end) is a number, e.g. ``<prefix>.chr{chrom}`` or
    ``<prefix>.chr{chrom}_{start}-{end}``. Only names the whole pattern matches are used, so
    two families in one directory never mix.

    Raises:
        FileNotFoundError: If no file matches.
    """
    regex = re.compile(
        "".join(
            "(?P<chrom>[0-9XY]+)" if part == "{chrom}" else r"\d+" if part.startswith("{") else re.escape(part)
            for part in re.split(r"(\{[^}]+\})", pattern)
        )
        + r"\.pvar"
    )
    out: dict[str, list[Path]] = {}
    for f in sorted(Path(pvar_dir).glob("*.pvar")):
        m = regex.fullmatch(f.name)
        if m:
            out.setdefault(m.group("chrom"), []).append(f)
    if not out:
        raise FileNotFoundError(f"no {pattern}.pvar files in {pvar_dir}")
    return out


def read_pvar_sites(pvars: dict[str, list[Path]], sites: pl.DataFrame) -> pl.DataFrame:
    """PVAR ``chrom, pos, REF, ALT`` rows at the (chrom, pos) of ``sites``."""
    frames = []
    for (chrom,), sub in sites.group_by("chrom", maintain_order=True):
        if chrom not in pvars:
            continue
        positions = sub["pos"].unique().to_list()
        for path in pvars[chrom]:
            frames.append(
                pl.scan_csv(
                    path,
                    separator="\t",
                    comment_prefix="##",
                    schema_overrides={"#CHROM": pl.String, "POS": pl.Int64},
                    infer_schema_length=0,
                )
                .filter(pl.col("POS").is_in(positions))
                .select(pl.lit(chrom).alias("chrom"), pl.col("POS").alias("pos"), "REF", "ALT")
                .collect()
            )
    if not frames:
        return pl.DataFrame(schema={"chrom": pl.String, "pos": pl.Int64, "REF": pl.String, "ALT": pl.String})
    return pl.concat(frames)


def liftover_fenland(
    catalog: pl.DataFrame, pvars: dict[str, list[Path]], liftover: Liftover, rsids: RsidLocator
) -> pl.DataFrame:
    """``catalog`` (hg19) lifted to hg38, with the PVAR ``REF``, ``ALT`` and ``allele_check``.

    ``chrom``/``pos`` become hg38 and the originals are kept as ``chrom_hg19``/``pos_hg19``.
    Where the PVAR has several rows at a site (a split multiallelic), a matching one is kept,
    so the rows are those of ``catalog`` that lift, in order. The catalog columns are the
    ``CATALOG_SCHEMA`` columns ``catalog`` has.
    """
    lifted = [
        lift(liftover, rsids, c, p, r)
        for c, p, r in zip(catalog["chrom"], catalog["pos"], catalog["rsid"], strict=True)
    ]
    df = (
        catalog.rename({"chrom": "chrom_hg19", "pos": "pos_hg19"})
        .with_row_index("row_id")
        .with_columns(
            pl.col("chrom_hg19").alias("chrom"),
            pl.Series("pos", [h.pos if h else None for h in lifted], dtype=pl.Int64),
        )
        .filter(pl.col("pos").is_not_null())
    )
    sites = read_pvar_sites(pvars, df.select("chrom", "pos"))
    match = ((pl.col("a1") == pl.col("REF")) & (pl.col("a2") == pl.col("ALT"))) | (
        (pl.col("a1") == pl.col("ALT")) & (pl.col("a2") == pl.col("REF"))
    )
    checked = df.join(sites, on=["chrom", "pos"], how="left").with_columns(
        pl.when(pl.col("REF").is_null())
        .then(pl.lit("no_pvar_variant"))
        .when(match)
        .then(pl.lit("match"))
        .otherwise(pl.lit("mismatch"))
        .alias("allele_check")
    )
    return (
        checked.sort("row_id", pl.col("allele_check") != "match", "REF", "ALT", nulls_last=True, maintain_order=True)
        .unique(subset=["row_id"], keep="first", maintain_order=True)
        .select(
            *(c for c in CATALOG_SCHEMA if c in catalog.columns),
            "chrom_hg19",
            "pos_hg19",
            "REF",
            "ALT",
            "allele_check",
        )
    )
