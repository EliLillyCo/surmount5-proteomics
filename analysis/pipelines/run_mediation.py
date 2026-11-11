"""Per-platform causal mediation of the tirzepatide proteomic effect (SURMOUNT-5).

For each platform (Olink, SomaScan) separately, decomposes the TZP-vs-SEMA effect on
each protein into a mediated (ACME) and a direct (ADE) component through percent weight
change, using the ``analysis.mediation`` engine (per-visit numpy nonparametric bootstrap,
Sobel/Wald p-values for FDR — see ``analysis.mediation.core.run_mediation_single_numpy``)
at each post-baseline VISITNUM (``analysis.design.POST_BASELINE_VISITS``).

Covariates in all three models: SEX, AGE, baseline weight (``WGTBL``) and the baseline
protein level the engine always adds. Mediator: percent weight change from baseline.

Output: ``<out>/surmount5_<platform>/covar_AGE_SEX/mediationRes/
surmount5_covar_AGE_SEX_mediation_PCTCHG_WGT.csv``, read by
``analysis.mediation.results.load_platform``.
"""

from __future__ import annotations

import argparse

import polars as pl
from analysis.mediation import run_mediation_pipeline
from analysis.mediation.cli import add_mediation_args
from analysis.mediation.constants import (
    CAT_COVARIATES,
    CONT_COVARIATES,
    CONTROL_VALUE,
    COVAR_LABEL,
    DEFAULT_SIMS,
    MEDIATOR,
    POST_BASELINE_VISITS,
    STUDY_NAME,
    TREAT_VALUE,
)
from analysis.platform_inputs import prepare_mediation_inputs
from analysis.qc_contract import PLATFORMS
from paths import MMRM_ROOT


def main() -> int:
    """Run the weight-mediation decomposition per platform from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_mediation_args(parser, out=MMRM_ROOT)
    parser.add_argument(
        "--platform",
        action="append",
        choices=PLATFORMS,
        help="Repeat to run several platforms. Defaults to both.",
    )
    parser.add_argument(
        "--visit",
        action="append",
        type=int,
        dest="visits",
        help="VISITNUM to fit; repeat for several. Defaults to the post-baseline visits of analysis.design.",
    )
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    visits = args.visits or list(POST_BASELINE_VISITS)

    for platform in args.platform or PLATFORMS:
        prepared = prepare_mediation_inputs(platform)
        data = prepared.data
        if args.protein_subset:
            data = data.filter(pl.col(prepared.protein_id_col).is_in(args.protein_subset))
        output_dir = args.out / f"{STUDY_NAME}_{platform}"
        print(f"\n{'=' * 72}")
        print(f"{STUDY_NAME} {platform.upper()} mediation -> {output_dir}")
        print(f"{'=' * 72}")
        run_mediation_pipeline(
            data=data,
            output_dir=output_dir,
            annotation=prepared.annotation,
            mediator=MEDIATOR,
            visits=visits,
            cat_covariates=list(CAT_COVARIATES),
            cont_covariates=list(CONT_COVARIATES),
            covar_label=COVAR_LABEL,
            study_name=STUDY_NAME,
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
