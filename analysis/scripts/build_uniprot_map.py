"""Build the cross-platform Olink/SomaScan marker-to-gene/UniProt mapping."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.tools.uniprot_map import OUTPUT_PATH_WIDE, build_uniprot_map


def main() -> None:
    build_uniprot_map()
    print(f"Wrote {OUTPUT_PATH_WIDE}")


if __name__ == "__main__":
    main()
