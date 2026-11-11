"""Build the cross-platform Olink/SomaScan marker-to-gene/UniProt mapping."""

from __future__ import annotations

from analysis.tools.uniprot_map import OUTPUT_PATH, build_uniprot_map


def main() -> None:
    build_uniprot_map()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
