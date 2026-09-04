"""Prepare analysis frames from SURMOUNT-5 QC outputs and run the manuscript mediation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.platform_inputs import (
    BASELINE_VISITNUM,
    CONTROL_VALUE,
    MEDIATION_COVARIATE_CONFIG,
    POST_BASELINE_VISITS,
    TREAT_VALUE,
    prepare_mediation_inputs,
)
from analysis.mediation import run_mediation_pipeline
from paths import MMRM_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        action="append",
        choices=["olink", "soma"],
        help="Repeat to run multiple platforms. Defaults to both.",
    )
    parser.add_argument("--visit", action="append", type=int, dest="visits")
    parser.add_argument("--output-root", type=Path, default=MMRM_ROOT)
    parser.add_argument(
        "--study-name",
        default="surmount5",
        choices=("surmount5",),
        help="Study identifier used in output directories and file names.",
    )
    parser.add_argument("--mediator", default="PCTCHG_WGT")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sims", type=int, default=100_000)
    return parser


def _selected_platforms(raw: list[str] | None) -> list[str]:
    return raw or ["olink", "soma"]


def main() -> int:
    args = build_parser().parse_args()
    visits = args.visits or list(POST_BASELINE_VISITS)
    for platform in _selected_platforms(args.platform):
        prepared = prepare_mediation_inputs(platform)
        output_dir = args.output_root / f"{args.study_name}_{platform}"
        print(f"\n{'=' * 72}")
        print(f"{args.study_name} {platform.upper()} mediation -> {output_dir}")
        print(f"{'=' * 72}")
        run_mediation_pipeline(
            data=prepared.data,
            output_dir=output_dir,
            annotation=prepared.annotation,
            mediator=args.mediator,
            visits=visits,
            cat_covariates=list(MEDIATION_COVARIATE_CONFIG["cat_covariates"]),
            cont_covariates=list(MEDIATION_COVARIATE_CONFIG["cont_covariates"]),
            covar_label=str(MEDIATION_COVARIATE_CONFIG["covar_label"]),
            study_name=args.study_name,
            baseline_visitnum=BASELINE_VISITNUM,
            response_var=prepared.response_var,
            response_var_bl=prepared.response_var_bl,
            protein_id_col=prepared.protein_id_col,
            control_value=CONTROL_VALUE,
            treat_value=TREAT_VALUE,
            n_jobs=args.n_jobs,
            sims=args.sims,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
