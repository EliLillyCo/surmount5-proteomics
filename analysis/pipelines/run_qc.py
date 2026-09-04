"""Run the SURMOUNT-5 Olink or SomaScan QC pipeline from raw proteomics data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import QC_ROOT

from analysis.pipelines.surmount5_qc import (
    output_prefix,
    resolve_input_paths,
    run_olink_qc,
    run_soma_qc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal SURMOUNT-5 Olink/Soma QC pipeline used by the paper."
    )
    parser.add_argument(
        "--platform",
        action="append",
        choices=["olink", "soma"],
        help="Platform(s) to run. Defaults to both.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=QC_ROOT,
        help="Root directory for QC outputs.",
    )
    parser.add_argument(
        "--study-id",
        default="surmount5",
        choices=("surmount5",),
        help="Public study identifier used in output prefixes.",
    )
    parser.add_argument(
        "--study-label",
        default="SURMOUNT-5",
        help="Human-readable study label for QC logs.",
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        default=None,
        help="Optional override for the bridge manifest CSV.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for deterministic QC steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and output prefixes without running QC.",
    )
    return parser.parse_args()


def _selected_platforms(args: argparse.Namespace) -> list[str]:
    return args.platform or ["olink", "soma"]


def main() -> None:
    args = parse_args()
    platforms = _selected_platforms(args)

    for platform in platforms:
        prefix = output_prefix(
            platform, output_root=args.output_root, study_id=args.study_id
        )
        resolved = resolve_input_paths(
            platform, bridge_manifest_path=args.bridge_manifest
        )
        if args.dry_run:
            logger.info("platform=%s", platform)
            logger.info("  data=%s", resolved.data_path)
            logger.info("  manifest=%s", resolved.manifest_path)
            logger.info("  adsl=%s", resolved.adsl_path)
            logger.info("  adlb=%s", resolved.adlb_path)
            logger.info("  bridge_manifest=%s", resolved.bridge_manifest_path)
            logger.info("  save_prefix=%s", prefix)
            continue

        prefix.parent.mkdir(parents=True, exist_ok=True)
        if platform == "olink":
            output_file, _ = run_olink_qc(
                save_prefix=prefix,
                study_label=args.study_label,
                random_seed=args.random_seed,
                bridge_manifest_path=args.bridge_manifest,
            )
        else:
            output_file, _ = run_soma_qc(
                save_prefix=prefix,
                study_label=args.study_label,
                random_seed=args.random_seed,
                bridge_manifest_path=args.bridge_manifest,
            )
        logger.info("Wrote %s", output_file)


if __name__ == "__main__":
    main()
