"""Build all supplementary tables into a single Excel workbook.

Each table module exposes build() -> list[tuple[str, pl.DataFrame]].
MANIFEST defines sheet order, tab names, titles, and abbreviations.

Run: uv run python tables/build_all.py
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Sequence

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUT_DIR / "supplementary_tables.xlsx"

ABBREVIATIONS = [
    ("TZP", "tirzepatide"),
    ("SEMA", "semaglutide"),
    ("MTD", "maximum tolerated dose"),
    ("FC", "fold change"),
    ("FDR", "false discovery rate"),
    ("PCBL", "percent change from baseline"),
    ("SE", "standard error"),
    ("CI", "confidence interval"),
    ("CV", "coefficient of variation"),
    ("QC", "quality control"),
    ("NPX", "normalized protein expression (log2, Olink)"),
    ("RFU", "relative fluorescence units (SomaScan)"),
    ("MMRM", "mixed model for repeated measures"),
    ("BMI", "body mass index"),
    ("HbA1c", "glycated hemoglobin"),
    ("ACME", "average causal mediation effect (weight-mediated)"),
    ("ADE", "average direct effect (weight-independent)"),
    ("cameraPR", "competitive gene set test (pre-ranked)"),
    ("ORA", "over-representation analysis"),
]

# Sheet-specific footnotes rendered below the data block (italicized).
FOOTNOTES = {
    "Table S3": "Data are mean (SD) for continuous variables and n (%) for categorical variables unless otherwise indicated.",
    "Table S7": "Lab values from VISITNUM 1 (screening); proteomics from VISITNUM 2 (randomization) for Wk0.",
    "Table S8": "Lab values from VISITNUM 1 (screening); proteomics from VISITNUM 2 (randomization) for Wk0.",
}

# (tab_name, description, module, sheet_index_within_module)
MANIFEST = [
    ("Table S1", "Olink Explore HT assay QC", "suppl_assay_qc", 0),
    ("Table S2", "SomaScan 11K assay QC", "suppl_assay_qc", 1),
    (
        "Table S3",
        "Baseline characteristics of the proteomic substudy",
        "suppl_characteristics",
        0,
    ),
    ("Table S4", "Baseline comparison, TZP vs SEMA (Olink)", "suppl_baseline", 0),
    ("Table S5", "Baseline comparison, TZP vs SEMA (SomaScan)", "suppl_baseline", 1),
    ("Table S6", "Cross-platform concordance (Olink × SomaScan)", "suppl_concordance", 0),
    (
        "Table S7",
        "Biomarker correlation (Olink vs clinical labs)",
        "suppl_biomarker_correlation",
        0,
    ),
    (
        "Table S8",
        "Biomarker correlation (SomaScan vs clinical labs)",
        "suppl_biomarker_correlation",
        1,
    ),
    ("Table S9", "Temporal proteomic response (Olink)", "suppl_trajectory", 0),
    ("Table S10", "Temporal proteomic response (SomaScan)", "suppl_trajectory", 1),
    (
        "Table S11",
        "Treatment comparison, TZP vs SEMA (Olink)",
        "suppl_across_treatment",
        0,
    ),
    (
        "Table S12",
        "Treatment comparison, TZP vs SEMA (SomaScan)",
        "suppl_across_treatment",
        1,
    ),
    (
        "Table S13",
        "Pathway enrichment — cameraPR (TZP vs SEMA)",
        "suppl_pathway_enrichment",
        0,
    ),
    (
        "Table S14",
        "Pathway enrichment — ORA (TZP vs SEMA)",
        "suppl_pathway_enrichment",
        1,
    ),
    ("Table S15", "Weight-mediation results, TZP vs SEMA (Olink)", "suppl_mediation", 0),
    (
        "Table S16",
        "Weight-mediation results, TZP vs SEMA (SomaScan)",
        "suppl_mediation",
        1,
    ),
]


def _load_table_module(name: str):
    return importlib.import_module(name)


def build_workbook(out_path: Path = DEFAULT_OUTPUT) -> Path:
    out_path = out_path.with_suffix(".xlsx")
    style = importlib.import_module("style")
    module_results: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for *_, module_name, _ in MANIFEST:
        if module_name not in module_results:
            module = _load_table_module(module_name)
            module_results[module_name] = [
                (name, df.to_pandas()) for name, df in module.build()
            ]

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Contents", index=False)

        for tab_name, _, module_name, idx in MANIFEST:
            _, df = module_results[module_name][idx]
            df.to_excel(writer, sheet_name=tab_name, index=False)

        wb = writer.book
        toc_entries = []
        for tab_name, description, _, _ in MANIFEST:
            ws = wb[tab_name]
            title = f"{tab_name}: {description}"
            style.style_sheet(ws, title, footnote=FOOTNOTES.get(tab_name))
            toc_entries.append((tab_name, description))

        toc_ws = wb["Contents"]
        style.create_toc(toc_ws, toc_entries, abbreviations=ABBREVIATIONS)

    print(f"Wrote {out_path}")
    for tab_name, description, *_ in MANIFEST:
        print(f"  {tab_name}: {description}")
    return out_path


def _print_manifest() -> None:
    for tab_name, description, *_ in MANIFEST:
        print(f"{tab_name}\t{description}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render supplementary tables and assemble a single workbook."
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List workbook tabs in build order and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output workbook path (default: {DEFAULT_OUTPUT.name})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_tables:
        _print_manifest()
        return 0
    build_workbook(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
