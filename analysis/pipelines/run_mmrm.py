"""Prepare analysis frames from SURMOUNT-5 QC outputs and run the manuscript MMRM workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.design import BASELINE_VISITNUM, CONTROL_VALUE, PLATFORMS, VISIT_LABELS
from analysis.design import STUDY_STEM as STUDY_NAME
from analysis.mmrm import run_mmrm_pipeline
from analysis.platform_inputs import MMRM_COVARIATE_CONFIG, prepare_mmrm_inputs
from paths import MMRM_ROOT



def build_parser() -> argparse.ArgumentParser:
    """The command-line parser: ``--platform`` (repeatable), ``--output-root``, ``--n-jobs``, ``--chunksize``.

    Returns:
        The parser; ``--output-root`` defaults to ``mmrm_root`` from ``paths.yaml``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        action="append",
        choices=list(PLATFORMS),
        help="Repeat to run multiple platforms. Defaults to both.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MMRM_ROOT,
        help="Root for <surmount5>_<platform>/covar_AGE_SEX trees (default: mmrm_root in paths.yaml, "
        "the root the figures, tables and mediation read).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunksize", type=int, default=10)
    return parser


def _selected_platforms(raw: list[str] | None) -> list[str]:
    return raw or list(PLATFORMS)


def main() -> int:
    """Fit the primary age- and sex-adjusted MMRM for each selected platform (both by default).

    For each platform it builds the frame with ``prepare_mmrm_inputs`` and runs
    ``run_mmrm_pipeline`` (change-from-baseline and baseline models), which writes
    ``<output-root>/surmount5_<platform>/covar_AGE_SEX/`` (``rawOutput/`` and the formatted
    outputs).

    Returns:
        0 when every platform has run.
    """
    args = build_parser().parse_args()
    for platform in _selected_platforms(args.platform):
        prepared = prepare_mmrm_inputs(platform)
        output_dir = args.output_root / f"{STUDY_NAME}_{platform}"
        print(f"\n{'=' * 72}")
        print(f"{STUDY_NAME} {platform.upper()} MMRM -> {output_dir}")
        print(f"{'=' * 72}")
        print("Running primary covariate model AGE_SEX")
        run_mmrm_pipeline(
            data=prepared.data,
            output_dir=output_dir,
            annotation=prepared.annotation,
            control_trt=CONTROL_VALUE,
            cat_covariates=list(MMRM_COVARIATE_CONFIG["cat_covariates"]),
            cont_covariates=list(MMRM_COVARIATE_CONFIG["cont_covariates"]),
            covar_label=str(MMRM_COVARIATE_CONFIG["covar_label"]),
            study_name=STUDY_NAME,
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
