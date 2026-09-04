"""CLI: file analysis orchestrator, batch processing, and argument parsing."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import GeneSetCache, _gene_set_cache
from .constants import (
    CACHE_DIR,
    DEFAULT_GENE_SETS,
    SEED,
)
from .loaders import load_olink_mapping, load_somascan_mapping
from .methods import run_camera, run_gsea, run_ora
from .preparation import prepare_data
from .reporting import print_summary_table, save_combined

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: analyse one results file
# ---------------------------------------------------------------------------


def analyse_file(
    results_file: str,
    output_dir: str,
    olink_mapping: dict[str, list[str]],
    somascan_mapping: dict[str, list[str]],
    gene_set_names: list[str],
    cache: GeneSetCache | None = None,
    qc_filter: str = "ALL",
    weighting: str = "equal",
    include_gsea: bool = False,
    skip_ora: bool = False,
    skip_camera: bool = False,
    skip_genes: list[str] | None = None,
    n_jobs: int = 1,
    download_missing: bool = False,
) -> bool:
    """Run all requested pathway analyses for a single MMRM result file.

    Returns True on success.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranked, sig = prepare_data(
        results_file, olink_mapping, somascan_mapping, qc_filter, weighting=weighting
    )
    if not ranked:
        logger.error(f"No genes for {results_file}")
        return False

    cache = cache or _gene_set_cache

    # Pre-fetch gene sets (avoids re-download inside loop)
    cache.prefetch(gene_set_names, download_missing=download_missing)

    gsea_res: list[pd.DataFrame] = []
    ora_res: list[pd.DataFrame] = []
    cam_res: list[pd.DataFrame] = []

    # Build task list: (gene_set, method)
    tasks: list[tuple[str, str]] = []
    for gs in gene_set_names:
        if include_gsea:
            tasks.append((gs, "gsea"))
        if not skip_ora:
            tasks.append((gs, "ora"))
        if not skip_camera:
            tasks.append((gs, "camera"))

    # Execute tasks (parallel across gene sets if requested)
    effective_jobs = min(max(n_jobs, 1), len(tasks))

    if effective_jobs > 1:
        logger.info(f"Running {len(tasks)} tasks with {effective_jobs} workers")
        with ProcessPoolExecutor(
            max_workers=effective_jobs,
            mp_context=mp.get_context("forkserver"),
        ) as pool:
            futures = {}
            for gs, method in tasks:
                cached_dict = cache.get(gs, download_missing=download_missing)
                fut = pool.submit(
                    _run_one_task,
                    gs,
                    method,
                    ranked,
                    sig,
                    SEED,
                    skip_genes,
                    cached_dict,
                    download_missing,
                )
                futures[fut] = (gs, method)
            for fut in as_completed(futures):
                gs, method = futures[fut]
                df = fut.result()
                if df is not None:
                    df["Gene_Set_Database"] = gs
                    {"gsea": gsea_res, "ora": ora_res, "camera": cam_res}[
                        method
                    ].append(df)
    else:
        # Sequential: reuse cached gene-set dicts
        for gs, method in tasks:
            logger.info(f"{method.upper()} -> {gs}")
            df = None
            cached_dict = cache.get(gs, download_missing=download_missing)
            if method == "gsea":
                df = run_gsea(ranked, gs, seed=SEED, gene_set_dict=cached_dict)
            elif method == "ora":
                df = run_ora(sig, ranked, gs, gene_set_dict=cached_dict)
            elif method == "camera":
                df = run_camera(
                    ranked,
                    gs,
                    gene_set_dict=cached_dict,
                    skip_genes=skip_genes,
                    download_missing=download_missing,
                )
            if df is not None:
                df["Gene_Set_Database"] = gs
                {"gsea": gsea_res, "ora": ora_res, "camera": cam_res}[method].append(df)

    # Save combined results
    save_combined(gsea_res, "gsea", "FDR q-val", out)
    save_combined(ora_res, "ora", "Adjusted P-value", out)
    save_combined(cam_res, "camera", "FDR q-val", out)

    # Print summary table
    print_summary_table(gsea_res, ora_res, cam_res)

    logger.info(f"Results saved to {out}")
    return True


def _run_one_task(
    gs,
    method,
    ranked,
    sig,
    seed,
    skip_genes=None,
    gene_set_dict=None,
    download_missing=False,
):
    """Worker for parallel gene-set tasks (must be picklable)."""
    if method == "gsea":
        return run_gsea(ranked, gs, seed=seed, gene_set_dict=gene_set_dict)
    elif method == "ora":
        return run_ora(sig, ranked, gs, gene_set_dict=gene_set_dict)
    elif method == "camera":
        return run_camera(
            ranked,
            gs,
            gene_set_dict=gene_set_dict,
            skip_genes=skip_genes,
            download_missing=download_missing,
        )
    return None


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def find_result_files(base_dir: Path, result_type: str) -> list[Path]:
    """Find all acTrt, withinTrt, or pcblRes result files."""
    return sorted(base_dir.glob(f"**/finalRes/{result_type}/*_py.csv"))


def parse_filename(filepath: Path, result_type: str) -> dict[str, str] | None:
    """Extract study / covar / comparison from path structure.

    Expected layout::

        {study}/covar_{covar}/finalRes/{type}/filename.csv
    """
    parts = filepath.parts
    try:
        idx = parts.index("finalRes")
        study = parts[idx - 2]
        covar_dir = parts[idx - 1]
        if not covar_dir.startswith("covar_"):
            return None
        covar = covar_dir[6:]
    except (ValueError, IndexError):
        return None

    filename = filepath.name
    if result_type == "acTrt":
        m = re.match(
            r"^.+?_covar_.+?_proteomics_olinkAnalysis_acrossTrts_resCmps_(.+?)@(.+?)_py\.csv$",
            filename,
        )
    elif result_type == "withinTrt":
        m = re.match(
            r"^.+?_covar_.+?_proteomics_olinkAnalysis_withinTrt_resCmps_(.+?)@(.+?)_py\.csv$",
            filename,
        )
    elif result_type == "pcblRes":
        m = re.match(
            r"^.+?_covar_.+?_proteomics_olinkAnalysis_PCBLRes_(.+?)@(.+?)_py\.csv$",
            filename,
        )
    else:
        return None

    if not m:
        return None
    comp, time = m.groups()
    return {
        "study": study,
        "covar": covar,
        "comparison_label": f"{comp}@{time}",
        "type": result_type,
    }


def construct_output_path(filepath: Path, parsed: dict[str, str]) -> Path:
    """Build output path by replacing ``finalRes`` with ``pathwayRes``.

    Input:  .../study/covar_X/finalRes/type/filename.csv
    Output: .../study/covar_X/pathwayRes/type/comparison_label/
    """
    parts = list(filepath.parts)
    try:
        idx = parts.index("finalRes")
    except ValueError:
        raise ValueError(f"'finalRes' not found in path: {filepath}")
    parts[idx] = "pathwayRes"
    return Path(*parts[: idx + 2]) / parsed["comparison_label"]


# ---------------------------------------------------------------------------
# CLI argument parsers
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser):
    """Add arguments shared by both sub-commands."""
    p.add_argument(
        "--mapping",
        required=True,
        help="Olink mapping file",
    )
    p.add_argument(
        "--somascan-mapping",
        required=True,
        help="Somascan mapping file",
    )
    p.add_argument("--qc-filter", choices=["ALL", "PASS_WARN", "PASS"], default="ALL")
    p.add_argument(
        "--weighting",
        choices=["equal", "ivw"],
        default="equal",
        help="Gene-level aggregation: 'equal' (simple mean, default) "
        "or 'ivw' (inverse-variance weighted)",
    )
    p.add_argument(
        "--gene-sets",
        nargs="+",
        default=DEFAULT_GENE_SETS,
        help="Gene set databases",
    )
    p.add_argument(
        "--run-gsea",
        action="store_true",
        help="Include GSEA preranked analysis (off by default)",
    )
    p.add_argument("--skip-ora", action="store_true")
    p.add_argument("--skip-camera", action="store_true")
    p.add_argument(
        "--skip-genes",
        nargs="+",
        default=None,
        help="Gene symbols to exclude from cameraPR statistic vector "
        "(e.g. proteins with very strong statistics that bias "
        "inter-gene correlation)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers per file for gene-set tasks (-1 = all CPUs)",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cached gene set files before running",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Local directory for cached gene-set JSON files",
    )
    p.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing gene-set libraries from Enrichr and cache them locally",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Pathway analysis for MMRM proteomics results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- single ---------------------------------------------------------------
    p_single = sub.add_parser("single", help="Analyse a single result file")
    p_single.add_argument("results_file", help="MMRM results CSV")
    p_single.add_argument("--output-dir", default="pathway_analysis_results")
    _add_common_args(p_single)

    # -- batch ----------------------------------------------------------------
    p_batch = sub.add_parser("batch", help="Discover and analyse all result files")
    p_batch.add_argument("--base-dir", type=Path, default=Path("."))
    p_batch.add_argument("--studies", nargs="*", help="Limit to these studies")
    p_batch.add_argument("--covariates", nargs="*", help="Limit to these covariates")
    p_batch.add_argument(
        "--types",
        nargs="*",
        choices=["acTrt", "withinTrt", "pcblRes"],
        default=["acTrt", "pcblRes"],
    )
    p_batch.add_argument("--dry-run", action="store_true")
    p_batch.add_argument("--verbose", action="store_true")
    _add_common_args(p_batch)

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_single(args):
    """Handle the 'single' sub-command."""
    random.seed(SEED)
    np.random.seed(SEED)
    cache = GeneSetCache(args.cache_dir)

    if args.clear_cache:
        cache.clear()

    olink_map = load_olink_mapping(args.mapping)
    soma_map = load_somascan_mapping(args.somascan_mapping)

    n_jobs = args.n_jobs if args.n_jobs != -1 else cpu_count()

    ok = analyse_file(
        results_file=args.results_file,
        output_dir=args.output_dir,
        olink_mapping=olink_map,
        somascan_mapping=soma_map,
        gene_set_names=args.gene_sets,
        cache=cache,
        qc_filter=args.qc_filter,
        weighting=args.weighting,
        include_gsea=args.run_gsea,
        skip_ora=args.skip_ora,
        skip_camera=args.skip_camera,
        skip_genes=args.skip_genes,
        n_jobs=n_jobs,
        download_missing=args.download_missing,
    )
    if not ok:
        sys.exit(1)


def cmd_batch(args):
    """Handle the 'batch' sub-command."""
    random.seed(SEED)
    np.random.seed(SEED)
    cache = GeneSetCache(args.cache_dir)

    if args.clear_cache:
        cache.clear()

    # ---- discover files ----
    all_files: list[tuple[Path, str]] = []
    for rtype in args.types:
        for f in find_result_files(args.base_dir, rtype):
            all_files.append((f, rtype))
    if not all_files:
        logger.error(f"No result files in {args.base_dir}")
        return

    logger.info(f"Found {len(all_files)} result files")

    # ---- filter & plan ----
    jobs: list[tuple[Path, Path, dict]] = []
    for fpath, rtype in all_files:
        parsed = parse_filename(fpath, rtype)
        if not parsed:
            logger.warning(f"Could not parse: {fpath.name}")
            continue
        if args.studies and parsed["study"] not in args.studies:
            continue
        if args.covariates and parsed["covar"] not in args.covariates:
            continue
        out = construct_output_path(fpath, parsed)
        jobs.append((fpath, out, parsed))

    if not jobs:
        logger.warning("No files after filtering")
        return

    logger.info(f"Will process {len(jobs)} files")

    if args.dry_run:
        for fpath, out, parsed in jobs:
            print(f"[DRY RUN] {fpath} -> {out}")
        return

    # ---- load shared resources ONCE ----
    olink_map = load_olink_mapping(args.mapping)
    soma_map = load_somascan_mapping(args.somascan_mapping)

    # Pre-fetch gene sets so they're cached before the loop
    logger.info("Pre-fetching gene set libraries...")
    cache.prefetch(args.gene_sets, download_missing=args.download_missing)

    n_jobs_inner = args.n_jobs if args.n_jobs != -1 else cpu_count()

    # ---- process sequentially (mappings & cache shared in-process) ----
    successes, failures = 0, 0
    failed_files: list[str] = []

    try:
        from tqdm import tqdm

        progress = tqdm(jobs, desc="Analysing", unit="file", disable=args.verbose)
    except ImportError:
        progress = jobs

    for fpath, out, parsed in progress:
        label = (
            f"{parsed['study']}/{parsed['covar']}/{parsed['type']}/"
            f"{parsed['comparison_label']}"
        )
        logger.info(f"Processing {label}")
        try:
            ok = analyse_file(
                results_file=str(fpath),
                output_dir=str(out),
                olink_mapping=olink_map,
                somascan_mapping=soma_map,
                gene_set_names=args.gene_sets,
                cache=cache,
                qc_filter=args.qc_filter,
                weighting=args.weighting,
                include_gsea=args.run_gsea,
                skip_ora=args.skip_ora,
                skip_camera=args.skip_camera,
                skip_genes=args.skip_genes,
                n_jobs=n_jobs_inner,
                download_missing=args.download_missing,
            )
            if ok:
                successes += 1
            else:
                failures += 1
                failed_files.append(fpath.name)
        except Exception as e:
            logger.error(f"Failed {fpath.name}: {e}")
            failures += 1
            failed_files.append(fpath.name)

    # ---- summary ----
    skipped = len(all_files) - len(jobs)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total found:  {len(all_files)}")
    print(f"Processed:    {successes}")
    print(f"Failed:       {failures}")
    print(f"Skipped:      {skipped}")
    if failed_files:
        print("Failed files:")
        for fn in failed_files:
            print(f"  - {fn}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for _noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "single":
        cmd_single(args)
    elif args.mode == "batch":
        cmd_batch(args)
