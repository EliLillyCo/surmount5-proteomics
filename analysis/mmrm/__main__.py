"""CLI entrypoint for the MMRM wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .pipeline import run_mmrm_pipeline


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(path, separator=separator)
    raise ValueError(f"Unsupported input format for {path}")


def _parse_visit_labels(values: list[str] | None) -> dict[int, int] | None:
    if not values:
        return None
    labels: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("visit labels must be VISITNUM=DISPLAY")
        visit, display = value.split("=", 1)
        labels[int(visit)] = int(display)
    return labels


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
    parser.add_argument("--study-name", default="STUDY")
    parser.add_argument("--covar-label", default="none")
    parser.add_argument("--baseline-visitnum", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunksize", type=int)
    parser.add_argument("--response-var", default="NPX")
    parser.add_argument("--response-var-bl", default="NPXBL")
    parser.add_argument("--protein-id-col", default="OlinkID")
    parser.add_argument("--cat-covariate", action="append", dest="cat_covariates")
    parser.add_argument("--cont-covariate", action="append", dest="cont_covariates")
    parser.add_argument("--visit-label", action="append", dest="visit_labels")
    parser.add_argument("--no-placebo-only", action="store_true")
    parser.add_argument(
        "--emmeans-weights", default="proportional", choices=["proportional", "equal"]
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = _read_table(args.data)
    annotation = _read_table(args.annotation)
    run_mmrm_pipeline(
        data=data,
        output_dir=args.output_dir,
        annotation=annotation,
        cat_covariates=args.cat_covariates,
        cont_covariates=args.cont_covariates,
        covar_label=args.covar_label,
        study_name=args.study_name,
        baseline_visitnum=args.baseline_visitnum,
        n_jobs=args.n_jobs,
        visit_labels=_parse_visit_labels(args.visit_labels),
        chunksize=args.chunksize,
        response_var=args.response_var,
        response_var_bl=args.response_var_bl,
        protein_id_col=args.protein_id_col,
        placebo_comparisons_only=not args.no_placebo_only,
        emmeans_weights=args.emmeans_weights,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
