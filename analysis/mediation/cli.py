"""Command-line flags of the mediation stage."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_mediation_args(parser: argparse.ArgumentParser, *, out: Path) -> None:
    """Add the mediation driver's flags.

    Args:
        parser: The driver's parser.
        out: Default ``--out`` directory.
    """
    parser.add_argument(
        "--protein-subset", nargs="*", default=None, help="Protein IDs (OlinkID or SeqId) for a fast smoke run."
    )
    parser.add_argument("--out", type=Path, default=out)
