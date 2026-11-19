"""CLI: contrast analysis orchestrator, batch processing, and argument parsing."""

from __future__ import annotations

import argparse
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from analysis.external import hcop
from analysis.external.hgnc_aliases import ALIASES_PATH, hgnc_ids, symbol_translation
from analysis.external.vendor_assay_tables import read_alias_table
from analysis.mmrm_outputs import RAW_OUTPUT, pathway_contrast_dir
from analysis.parallel import pin_blas_threads, resolve_n_jobs
from analysis.platform_inputs import EXCLUDED_ASSAYS
from analysis.qc_contract import PLATFORMS
from analysis.statistics import SEED

from .cache import GeneSetCache
from .constants import CACHE_DIR, DEFAULT_GENE_SETS
from .loaders import ANNOTATION_PATH, load_assay_genes
from .methods import run_camera, run_ora
from .preparation import contrast_statistics, gene_source_assays, prepare_data, source_groups
from .reporting import print_summary_table, save_combined

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Core: analyze one contrast
# ---------------------------------------------------------------------------


def analyse_contrast(
    stats: pl.DataFrame,
    output_dir: Path,
    mapping: dict[str, list[str]],
    gene_set_names: list[str],
    excluded_assays: tuple[str, ...] = (),
    *,
    cache: GeneSetCache,
    skip_ora: bool = False,
    skip_camera: bool = False,
    n_jobs: int = 1,
    download_missing: bool = False,
) -> None:
    """Run cameraPR and ORA for one contrast's statistics and write the combined CSVs.

    Args:
        stats: ``preparation.contrast_statistics`` frame (``marker, estimate, StdErr, pVal, fdr``).
        output_dir: Directory for ``camera_combined.csv`` and ``ora_combined.csv``.
        mapping: Assay id -> gene symbols.
        gene_set_names: Enrichr library names.
        excluded_assays: Assay ids dropped before ranking and from ORA (see ``EXCLUDED_ASSAYS``).
        cache: Gene-set cache built with the alias translation (``_setup``); required, so no library
            reaches cameraPR or ORA untranslated.
        skip_ora: Do not run ORA.
        skip_camera: Do not run cameraPR.
        n_jobs: Parallel workers over (library, method) tasks.
        download_missing: Allow downloading libraries missing from the cache.

    Raises:
        ValueError: When the contrast maps to no gene.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranked, sig = prepare_data(stats, mapping, excluded_assays)
    if not ranked:
        raise ValueError(f"No genes mapped for the contrast written to {out}")
    groups = source_groups(gene_source_assays(stats, mapping, excluded_assays))

    cache.prefetch(gene_set_names, download_missing=download_missing)

    tasks: list[tuple[str, str]] = []
    for gs in gene_set_names:
        if not skip_ora:
            tasks.append((gs, "ora"))
        if not skip_camera:
            tasks.append((gs, "camera"))

    results: dict[str, list[pl.DataFrame]] = {"ora": [], "camera": []}
    effective_jobs = min(max(n_jobs, 1), len(tasks))

    if effective_jobs > 1:
        import multiprocessing as mp

        logger.info(f"Running {len(tasks)} tasks with {effective_jobs} workers")
        # The fork server and its workers inherit the pinned environment: each worker's BLAS
        # runs single-threaded, as in forkserver_pool.
        pin_blas_threads()
        with ProcessPoolExecutor(
            max_workers=effective_jobs, mp_context=mp.get_context("forkserver")
        ) as pool:
            futures = {
                pool.submit(
                    _run_one_task, gs, method, ranked, sig, cache.get(gs), groups
                ): (gs, method)
                for gs, method in tasks
            }
            done = {futures[f]: f.result() for f in as_completed(futures)}
        for gs, method in tasks:  # task order, not completion order
            df = done[(gs, method)]
            if df is not None:
                results[method].append(df.with_columns(pl.lit(gs).alias("Gene_Set_Database")))
    else:
        for gs, method in tasks:
            logger.info(f"{method.upper()} -> {gs}")
            df = _run_one_task(gs, method, ranked, sig, cache.get(gs), groups)
            if df is not None:
                results[method].append(df.with_columns(pl.lit(gs).alias("Gene_Set_Database")))

    if not skip_ora:
        save_combined(results["ora"], "ora", "Adjusted P-value", out)
    if not skip_camera:
        save_combined(results["camera"], "camera", "FDR q-val", out)
    print_summary_table(results["ora"], results["camera"])
    logger.info(f"Results saved to {out}")


def _run_one_task(gs, method, ranked, sig, gene_set_dict, groups):
    """Worker for one (library, method) task (must be picklable); ``groups`` count same-source genes once."""
    if method == "ora":
        return run_ora(sig, ranked, gs, gene_set_dict=gene_set_dict, groups=groups)
    return run_camera(ranked, gs, gene_set_dict=gene_set_dict, groups=groups)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def find_mmrm_dirs(base_dir: Path) -> list[tuple[Path, str, str]]:
    """Find every ``<study>_<platform>/covar_<label>`` directory holding a ``rawOutput``.

    Returns:
        ``(covar_dir, study_dir_name, platform)`` sorted by path; platform is the
        ``olink``/``soma`` suffix of the study directory.
    """
    found = []
    for raw in sorted(Path(base_dir).glob(f"*/covar_*/{RAW_OUTPUT}")):
        covar_dir = raw.parent
        study = covar_dir.parent.name
        platform = study.rsplit("_", 1)[-1]
        if platform not in PLATFORMS:
            raise ValueError(f"Cannot read the platform (olink|soma) from study directory {study!r}")
        found.append((covar_dir, study, platform))
    return found


def construct_output_path(covar_dir: Path, label: str) -> Path:
    """``<covar_dir>/pathwayRes/acTrt/<label>/``."""
    return pathway_contrast_dir(covar_dir, label)


# ---------------------------------------------------------------------------
# CLI argument parsers
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser):
    """Add arguments shared by both sub-commands."""
    p.add_argument(
        "--annotation",
        type=Path,
        default=ANNOTATION_PATH,
        help="Assay annotation (stage vendor_assay_tables): assay -> genes for both platforms",
    )
    p.add_argument(
        "--aliases",
        type=Path,
        default=ALIASES_PATH,
        help="HGNC alias table (stage vendor_assay_tables): library members are translated to the annotation's symbols",
    )
    p.add_argument(
        "--hcop",
        type=Path,
        default=hcop.HCOP_PATH,
        help="HCOP human-mouse orthologs (stage hcop): CellMarker's mouse sets take their human orthologs",
    )
    p.add_argument("--gene-sets", nargs="+", default=DEFAULT_GENE_SETS, help="Gene set databases")
    p.add_argument("--skip-ora", action="store_true")
    p.add_argument("--skip-camera", action="store_true")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers per contrast for gene-set tasks (-1 = all CPUs but one)",
    )
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Gene-set cache directory")
    p.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing gene-set libraries from Enrichr and cache them locally",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Pathway analysis (cameraPR on z = estimate / SE, and ORA) for MMRM results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Analyze one contrast of one MMRM covar directory")
    p_single.add_argument("mmrm_dir", type=Path, help="<study>_<platform>/covar_<label> directory")
    p_single.add_argument("--platform", required=True, choices=PLATFORMS)
    p_single.add_argument(
        "--label", required=True, help="e.g. TZP15mgorMTDVSSEMA2.4mgorMTD@24"
    )
    p_single.add_argument("--output-dir", type=Path, required=True)
    _add_common_args(p_single)

    p_batch = sub.add_parser("batch", help="Analyze every contrast under an MMRM root")
    p_batch.add_argument("--base-dir", type=Path, required=True, help="Root holding <study>_<platform>/")
    p_batch.add_argument("--studies", nargs="*", help="Limit to these study directory names")
    p_batch.add_argument("--dry-run", action="store_true")
    _add_common_args(p_batch)

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _setup(args) -> tuple[GeneSetCache, dict, dict, int]:
    random.seed(SEED)
    np.random.seed(SEED)
    aliases = read_alias_table(args.aliases)
    translation = symbol_translation(aliases)
    id_symbol = {i: s for s, i in hgnc_ids(aliases).items()}
    mouse = hcop.mouse_translation(hcop.read_orthologs(args.hcop), id_symbol, translation)
    cache = GeneSetCache(args.cache_dir, translation=translation, mouse=mouse)
    mappings = {platform: load_assay_genes(platform, args.annotation) for platform in PLATFORMS}
    return cache, mappings, resolve_n_jobs(args.n_jobs)


def cmd_single(args):
    """Handle the 'single' sub-command."""
    cache, mappings, n_jobs = _setup(args)
    stats = contrast_statistics(args.mmrm_dir)[args.label]
    analyse_contrast(
        stats,
        args.output_dir,
        mappings[args.platform],
        args.gene_sets,
        excluded_assays=EXCLUDED_ASSAYS[args.platform],
        cache=cache,
        skip_ora=args.skip_ora,
        skip_camera=args.skip_camera,
        n_jobs=n_jobs,
        download_missing=args.download_missing,
    )


def cmd_batch(args):
    """Handle the 'batch' sub-command; any contrast failure stops the run."""
    cache, mappings, n_jobs = _setup(args)
    jobs = []
    for covar_dir, study, platform in find_mmrm_dirs(args.base_dir):
        if args.studies and study not in args.studies:
            continue
        for label, stats in contrast_statistics(covar_dir).items():
            jobs.append((study, platform, label, construct_output_path(covar_dir, label), stats))
    if not jobs:
        raise FileNotFoundError(f"No MMRM rawOutput contrasts found under {args.base_dir}")
    logger.info(f"Will process {len(jobs)} contrasts")

    if args.dry_run:
        for study, _, label, out, _ in jobs:
            print(f"[DRY RUN] {study}/{label} -> {out}")
        return

    cache.prefetch(args.gene_sets, download_missing=args.download_missing)
    for study, platform, label, out, stats in jobs:
        logger.info(f"Processing {study}/{label}")
        analyse_contrast(
            stats,
            out,
            mappings[platform],
            args.gene_sets,
            excluded_assays=EXCLUDED_ASSAYS[platform],
            cache=cache,
            skip_ora=args.skip_ora,
            skip_camera=args.skip_camera,
            n_jobs=n_jobs,
            download_missing=args.download_missing,
        )


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    args = build_parser().parse_args()
    if args.mode == "single":
        cmd_single(args)
    else:
        cmd_batch(args)
