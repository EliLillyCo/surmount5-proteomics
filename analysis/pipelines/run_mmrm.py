"""Prepare analysis frames from SURMOUNT-5 QC outputs and run the manuscript MMRM workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.platform_inputs import (
    BASELINE_VISITNUM,
    MMRM_COVARIATE_CONFIG,
    VISIT_LABELS,
    prepare_mmrm_inputs,
)
from analysis.mmrm import run_mmrm_pipeline
from paths import MMRM_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        action="append",
        choices=["olink", "soma"],
        help="Repeat to run multiple platforms. Defaults to both.",
    )
    parser.add_argument("--output-root", type=Path, default=MMRM_ROOT)
    parser.add_argument(
        "--study-name",
        default="surmount5",
        choices=("surmount5",),
        help="Study identifier used in output directories and file names.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunksize", type=int, default=10)
    return parser


def _selected_platforms(raw: list[str] | None) -> list[str]:
    return raw or ["olink", "soma"]


def main() -> int:
    args = build_parser().parse_args()
    for platform in _selected_platforms(args.platform):
        prepared = prepare_mmrm_inputs(platform)
        output_dir = args.output_root / f"{args.study_name}_{platform}"
        print(f"\n{'=' * 72}")
        print(f"{args.study_name} {platform.upper()} MMRM -> {output_dir}")
        print(f"{'=' * 72}")
        print("Running primary covariate model AGE_SEX")
        run_mmrm_pipeline(
            data=prepared.data,
            output_dir=output_dir,
            annotation=prepared.annotation,
            cat_covariates=list(MMRM_COVARIATE_CONFIG["cat_covariates"]),
            cont_covariates=list(MMRM_COVARIATE_CONFIG["cont_covariates"]),
            covar_label=str(MMRM_COVARIATE_CONFIG["covar_label"]),
            study_name=args.study_name,
            baseline_visitnum=BASELINE_VISITNUM,
            visit_labels=VISIT_LABELS,
            response_var=prepared.response_var,
            response_var_bl=prepared.response_var_bl,
            protein_id_col=prepared.protein_id_col,
            n_jobs=args.n_jobs,
            chunksize=args.chunksize,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
