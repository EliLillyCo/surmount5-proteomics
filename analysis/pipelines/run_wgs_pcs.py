#!/usr/bin/env python
"""Build LD-pruned genome and compute SNPRelate principal components."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import GENO_ROOT, QA_ROOT

SUPPORT_DIR = Path(__file__).resolve().parent.parent / "pharmacogenomics"
PRUNING_SCRIPT = SUPPORT_DIR / "build_pruned_genome.py"
PCA_SCRIPT = SUPPORT_DIR / "compute_wgs_pcs.R"
AUTOSOMES = tuple(range(1, 23))


def _default_dist_version() -> str:
    parent_name = GENO_ROOT.parent.name
    if re.fullmatch(r"\d{8}_\d{6}", parent_name):
        return parent_name

    chr1_dir = (
        QA_ROOT
        / "2000_genomic_data_processing"
        / "2200_analysis_data"
        / "merged_chr"
        / "chr1"
    )
    matches = sorted(
        list(chr1_dir.glob("2202_surmount5_data*_qced_chr1.pgen"))
        + list(chr1_dir.glob("2202_surmount5_data*_qced_chr1.pgen"))
    )
    if not matches:
        raise FileNotFoundError(
            "Could not infer the genotype distribution version from qa_root. "
            "Pass --dist-version explicitly."
        )

    match = re.search(
        r"2202_surmount5_data(\d{8}_\d{6})_qced_chr1\.pgen$",
        matches[-1].name,
    )
    if match is None:
        raise ValueError(
            f"Could not parse a data-version stamp from {matches[-1].name}. "
            "Pass --dist-version explicitly."
        )
    return match.group(1)


def _maf_tag(value: float) -> str:
    return f"{value:g}"


def _build_pruned_prefix(
    qa_root: Path, study_name: str, dist_version: str, maf: float
) -> Path:
    return (
        qa_root
        / "2000_genomic_data_processing"
        / "2200_analysis_data"
        / "pruned_genome"
        / f"2203_{study_name}_data{dist_version}_qced_maf{_maf_tag(maf)}_pruned_genome"
    )


def _build_pca_prefix(
    qa_root: Path, study_name: str, dist_version: str, maf: float
) -> Path:
    return (
        qa_root
        / "3000_genomic_data_summary"
        / "3100_pca"
        / f"3101_{study_name}_data{dist_version}_qced_maf{_maf_tag(maf)}_pruned_genome"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pruned autosomal genome and SNPRelate principal components "
            "for the SURMOUNT-5 rs1800437 pharmacogenomic analysis."
        )
    )
    parser.add_argument("--qa-root", type=Path, default=QA_ROOT)
    parser.add_argument("--study-name", default="surmount5")
    parser.add_argument("--dist-version", default=None)
    parser.add_argument("--maf", type=float, default=0.01)
    parser.add_argument("--hwe", type=float, default=1e-4)
    parser.add_argument("--window-kb", type=int, default=200)
    parser.add_argument("--step-variant-count", type=int, default=5)
    parser.add_argument("--r2", type=float, default=0.1)
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chromosomes", type=int, nargs="+", default=list(AUTOSOMES))
    parser.add_argument("--skip-pruned-genome", action="store_true")
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    dist_version = args.dist_version or _default_dist_version()

    if not args.skip_pruned_genome:
        pruning_command = [
            sys.executable,
            str(PRUNING_SCRIPT),
            "--qa-root",
            str(args.qa_root),
            "--study-name",
            args.study_name,
            "--dist-version",
            dist_version,
            "--maf",
            str(args.maf),
            "--hwe",
            str(args.hwe),
            "--window-kb",
            str(args.window_kb),
            "--step-variant-count",
            str(args.step_variant_count),
            "--r2",
            str(args.r2),
            "--threads",
            str(args.threads),
            "--chromosomes",
            *[str(chrom) for chrom in args.chromosomes],
        ]
        if args.dry_run:
            pruning_command.append("--dry-run")
        _run(pruning_command, dry_run=args.dry_run)

    if not args.skip_pca:
        pruned_prefix = _build_pruned_prefix(
            qa_root=args.qa_root,
            study_name=args.study_name,
            dist_version=dist_version,
            maf=args.maf,
        )
        pca_prefix = _build_pca_prefix(
            qa_root=args.qa_root,
            study_name=args.study_name,
            dist_version=dist_version,
            maf=args.maf,
        )
        pca_command = [
            "Rscript",
            str(PCA_SCRIPT),
            "--bed-prefix",
            str(pruned_prefix),
            "--out-prefix",
            str(pca_prefix),
            "--num-pcs",
            str(args.num_pcs),
            "--threads",
            str(args.threads),
        ]
        _run(pca_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
