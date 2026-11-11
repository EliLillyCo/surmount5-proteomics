"""Fit the treatment x visit x sex MMRM on each platform and write the interaction results."""

from __future__ import annotations

import argparse

from analysis.mmrm.sex_interaction import run_sex_interaction
from analysis.mmrm_outputs import sexint_csv
from analysis.design import BASELINE_VISITNUM, PLATFORMS
from analysis.platform_inputs import prepare_mmrm_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        action="append",
        choices=list(PLATFORMS),
        help="Repeat to run multiple platforms. Defaults to both.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunksize", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for platform in args.platform or list(PLATFORMS):
        prepared = prepare_mmrm_inputs(platform)
        out_path = sexint_csv(platform)
        print(f"\n{'=' * 72}\nSURMOUNT-5 {platform.upper()} sex interaction -> {out_path}\n{'=' * 72}")
        rows = run_sex_interaction(
            prepared.data,
            annotation=prepared.annotation,
            response_var=prepared.response_var,
            response_var_bl=prepared.response_var_bl,
            protein_id_col=prepared.protein_id_col,
            baseline_visitnum=BASELINE_VISITNUM,
            n_jobs=args.n_jobs,
            chunksize=args.chunksize,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows.write_csv(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
