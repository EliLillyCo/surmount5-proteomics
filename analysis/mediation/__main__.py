"""CLI entrypoint for the mediation wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .constants import (
    CAT_COVARIATES,
    CONT_COVARIATES,
    DEFAULT_CONTROL_VALUE,
    DEFAULT_RESPONSE_VAR,
    DEFAULT_RESPONSE_VAR_BL,
    DEFAULT_SIMS,
    DEFAULT_TREAT_COL,
    DEFAULT_TREAT_VALUE,
    DEFAULT_VISITS,
)
from .pipeline import run_mediation_pipeline


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(path, separator=separator)
    raise ValueError(f"Unsupported input format for {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Long-format analysis parquet/csv/tsv"
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Marker annotation parquet/csv/tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory root"
    )
    parser.add_argument("--mediator", default="PCTCHG_WGT")
    parser.add_argument(
        "--visit",
        action="append",
        type=int,
        dest="visits",
        help="Visit number to analyze",
    )
    parser.add_argument("--study-name", default="STUDY")
    parser.add_argument("--covar-label", default="none")
    parser.add_argument("--baseline-visitnum", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--response-var", default=DEFAULT_RESPONSE_VAR)
    parser.add_argument("--response-var-bl", default=DEFAULT_RESPONSE_VAR_BL)
    parser.add_argument("--protein-id-col", default="OlinkID")
    parser.add_argument("--treat-col", default=DEFAULT_TREAT_COL)
    parser.add_argument("--control-value", default=DEFAULT_CONTROL_VALUE)
    parser.add_argument("--treat-value", default=DEFAULT_TREAT_VALUE)
    parser.add_argument("--cat-covariate", action="append", dest="cat_covariates")
    parser.add_argument("--cont-covariate", action="append", dest="cont_covariates")
    parser.add_argument(
        "--sims",
        type=int,
        default=DEFAULT_SIMS,
        help="Bootstrap replicates per protein",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = _read_table(args.data)
    annotation = _read_table(args.annotation)
    run_mediation_pipeline(
        data=data,
        output_dir=args.output_dir,
        annotation=annotation,
        mediator=args.mediator,
        visits=args.visits or list(DEFAULT_VISITS),
        cat_covariates=args.cat_covariates or list(CAT_COVARIATES),
        cont_covariates=args.cont_covariates or list(CONT_COVARIATES),
        covar_label=args.covar_label,
        study_name=args.study_name,
        baseline_visitnum=args.baseline_visitnum,
        n_jobs=args.n_jobs,
        response_var=args.response_var,
        response_var_bl=args.response_var_bl,
        protein_id_col=args.protein_id_col,
        treat_col=args.treat_col,
        control_value=args.control_value,
        treat_value=args.treat_value,
        sims=args.sims,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
