"""Run the SURMOUNT-5 QC for the Olink and/or SomaScan platform run.

Each run is ``analysis.qc.study.run_study``: the thresholds and steps of
``analysis/qc/config_<platform>.toml`` and the inputs its ``paths:`` keys name. It
writes the files ``analysis.qc_contract`` describes under ``<qc_root>/<study>``, or under
``--output-dir/<study>`` when given.

    pixi run python -m analysis.pipelines.run_qc --study surmount5_olink \
        --output-dir <scratch>/qc

Before any data is read, the run checks that the ``qc_reference`` stage's outputs it needs exist
(``missing_reference_inputs``): the sex-association and ProtAge tables and the parsed cis-pQTL
catalogs each run's config names. A missing one stops the run with the list and the stage command.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from analysis.qc.pqtl_reference import CATALOG_NAMES
from analysis.qc.study import run_study, trial_config
from analysis.qc_contract import PLATFORM_STUDIES
from analysis.qc_paths import PQTL_CATALOG_DIR, PROTAGE_TABLE, SEX_TABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STUDIES: tuple[str, ...] = tuple(PLATFORM_STUDIES.values())
REFERENCE_STAGE_COMMAND = "pixi run stage run qc_reference"


def reference_inputs(study: str) -> list[Path]:
    """The ``qc_reference`` outputs one platform run reads.

    Args:
        study: ``surmount5_olink`` or ``surmount5_soma``.

    Returns:
        The sex-association and ProtAge tables, then the parsed catalog of each
        ``pqtl_reference_catalog`` entry of the run's config when pQTL is enabled (a catalog
        name resolves to its file in ``PQTL_CATALOG_DIR``, a ``.parquet`` value is taken as given).
    """
    config = trial_config(study)
    paths = [SEX_TABLE, PROTAGE_TABLE]
    if config.pqtl_enabled:
        catalogs = config.pqtl_reference_catalog
        for source in (catalogs,) if isinstance(catalogs, str) else catalogs:
            name = source.strip().lower()
            paths.append(PQTL_CATALOG_DIR / CATALOG_NAMES[name] if name in CATALOG_NAMES else Path(source))
    return paths


def missing_reference_inputs(studies: tuple[str, ...]) -> list[Path]:
    """The ``qc_reference`` outputs the runs need that do not exist, each once, in first-use order."""
    needed = dict.fromkeys(path for study in studies for path in reference_inputs(study))
    return [path for path in needed if not path.is_file()]


def require_reference_inputs(studies: tuple[str, ...]) -> None:
    """Stop before any data is read when a ``qc_reference`` output is missing.

    Raises:
        FileNotFoundError: Naming every missing file and the stage command that writes them.
    """
    missing = missing_reference_inputs(studies)
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            f"The QC reads the outputs of stage qc_reference, and {len(missing)} are missing:\n"
            f"{listing}\nRun `{REFERENCE_STAGE_COMMAND}` first."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the QC command-line arguments.

    Returns:
        Namespace with ``study`` (a run or ``both``) and ``output_dir`` (or None).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--study",
        choices=[*STUDIES, "both"],
        default="both",
        help="Platform run(s) to QC, one after the other. Defaults to both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write each run's outputs to <dir>/<study> instead of <qc_root>/<study>.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the QC for the requested platform run(s)."""
    args = parse_args(argv)
    studies = STUDIES if args.study == "both" else (args.study,)
    require_reference_inputs(studies)
    for study in studies:
        out = args.output_dir / study if args.output_dir is not None else None
        written = run_study(study, out)
        logger.info("%s: QC outputs in %s", study, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
