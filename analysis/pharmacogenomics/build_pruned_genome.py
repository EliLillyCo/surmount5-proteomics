#!/usr/bin/env python
"""PLINK2 LD-pruning to produce the autosomal pruned genome for SNPRelate PCA."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis.design import STUDY_STEM
from paths import required_value

AUTOSOMES = tuple(range(1, 23))


def _maf_tag(value: float) -> str:
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pruned autosomal genome used for the SURMOUNT-5 rs1800437 "
            "pharmacogenomic PCA step from the merged QC PLINK chromosome files."
        )
    )
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--study-name", default=STUDY_STEM, help="stem of the files this step writes")
    parser.add_argument(
        "--qa-file-stem",
        default=None,
        help="study tag in the names of the merged chromosome files read (default: paths.yaml surmount5_qa_file_stem)",
    )
    parser.add_argument("--dist-version", required=True)
    parser.add_argument("--maf", type=float, default=0.01)
    parser.add_argument("--hwe", type=float, default=1e-4)
    parser.add_argument("--window-kb", type=int, default=200)
    parser.add_argument("--step-variant-count", type=int, default=5)
    parser.add_argument("--r2", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chromosomes", type=int, nargs="+", default=list(AUTOSOMES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    qa_file_stem = args.qa_file_stem or required_value("surmount5_qa_file_stem")

    plink2 = shutil.which("plink2")
    if plink2 is None and not args.dry_run:
        raise FileNotFoundError("plink2 is required but was not found on PATH.")
    plink2 = plink2 or "plink2"

    merged_root = (
        args.qa_root
        / "2000_genomic_data_processing"
        / "2200_analysis_data"
        / "merged_chr"
    )
    pruned_root = (
        args.qa_root
        / "2000_genomic_data_processing"
        / "2200_analysis_data"
        / "pruned_genome"
    )
    pruned_root.mkdir(parents=True, exist_ok=True)

    maf_tag = _maf_tag(args.maf)
    exclude_file = (
        merged_root
        / f"2201_{qa_file_stem}_data{args.dist_version}_exclude_meanDP_snps.txt"
    )
    # The variants the genotype QC set aside for mean depth are always excluded: a missing
    # list stops the run, so pruning cannot go ahead on the unfiltered variants.
    if not args.dry_run and not exclude_file.exists():
        raise FileNotFoundError(f"Missing variant exclusion list: {exclude_file}")

    pruned_prefixes: list[Path] = []
    for chrom in args.chromosomes:
        input_prefix = (
            merged_root
            / f"chr{chrom}"
            / f"2202_{qa_file_stem}_data{args.dist_version}_qced_chr{chrom}"
        )
        if not args.dry_run and not input_prefix.with_suffix(".pgen").exists():
            raise FileNotFoundError(
                f"Missing merged chromosome PLINK file: {input_prefix}.pgen"
            )

        prune_prefix = (
            merged_root
            / f"chr{chrom}"
            / f"2202_{args.study_name}_data{args.dist_version}_qced_maf{maf_tag}_pruned_chr{chrom}"
        )
        prune_command = [
            plink2,
            "--pfile",
            str(input_prefix),
            "--maf",
            str(args.maf),
            "--hwe",
            str(args.hwe),
        ]
        prune_command.extend(
            [
                "--exclude",
                str(exclude_file),
                "--indep-pairwise",
                str(args.window_kb),
                str(args.step_variant_count),
                str(args.r2),
                "--threads",
                str(args.threads),
                "--out",
                str(prune_prefix),
            ]
        )
        _run(prune_command, dry_run=args.dry_run)

        extract_command = [
            plink2,
            "--pfile",
            str(input_prefix),
            "--extract",
            str(prune_prefix.with_suffix(".prune.in")),
            "--make-bed",
            "--threads",
            str(args.threads),
            "--out",
            str(prune_prefix),
        ]
        _run(extract_command, dry_run=args.dry_run)
        pruned_prefixes.append(prune_prefix)

    merged_prefix = (
        pruned_root
        / f"2203_{args.study_name}_data{args.dist_version}_qced_maf{maf_tag}_pruned_genome"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as handle:
        for prefix in pruned_prefixes:
            handle.write(f"{prefix}\n")
        handle.flush()

        merge_command = [
            plink2,
            "--pmerge-list",
            handle.name,
            "bfile",
            "--make-bed",
            "--threads",
            str(args.threads),
            "--out",
            str(merged_prefix),
        ]
        _run(merge_command, dry_run=args.dry_run)

    make_pgen_command = [
        plink2,
        "--bfile",
        str(merged_prefix),
        "--make-pgen",
        "--threads",
        str(args.threads),
        "--out",
        str(merged_prefix),
    ]
    _run(make_pgen_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
