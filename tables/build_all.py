"""Build all supplementary tables into a single Excel workbook.

Each table module exposes build() -> list[tuple[str, pl.DataFrame]].
MANIFEST defines sheet order, tab names, titles, and abbreviations.

Run: pixi run python tables/build_all.py
"""

from __future__ import annotations

import argparse
import importlib
import io
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from analysis.mediation.constants import DEFAULT_SIMS
from analysis.tools.provenance import record_build
from tables import style

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = OUT_DIR / "supplementary_tables.xlsx"

# openpyxl stamps the workbook's created and modified properties with the current time, and
# the zip with the time each member was written, so two builds of the same tables differ in
# their bytes. Both are fixed here (the zip epoch, which is the earliest zip time) so that the
# workbook is a function of its cells and styles alone.
WORKBOOK_TIMESTAMP = datetime(1980, 1, 1)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MODIFIED = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")

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
    "Table S1": "Cis_pQTL_Catalogs lists the independent published catalogs that report at least one cis association (within 1 Mb of the gene, p < 5e-8) for the assay. UKB-PPP, deCODE and Fenland publish only associations passing their own stricter thresholds (p < 1.7e-11, p < 1.8e-9 and p < 1e-11), so for these catalogs support reflects those thresholds. Olink catalogs: UKB-PPP (Sun et al., Nature 2023), JHS (Tahir et al., J Clin Invest 2024), MESA (Nicholas et al., Nat Commun 2026), CKB (Wang et al., Nat Commun 2025) and Genes & Health (Pietzner et al., Nat Genet 2026). SomaScan catalogs: deCODE (Ferkingstad et al., Nat Genet 2021), ARIC (Zhang et al., Nat Genet 2022), Fenland (Pietzner et al., Science 2021), CKB (Pozarickij A. et al., Science Bulletin (2026), doi:10.1016/j.scib.2026.10.060), MESA (Nicholas et al., Nat Commun 2026), Genes & Health (Pietzner et al., Nat Genet 2026), AGES (Gudjonsson et al., Nat Commun 2022) and INTERVAL (Sun et al., Nature 2018). UKB-PPP, JHS, MESA and CKB Olink used Olink Explore 3072 and were matched to Explore HT assays by UniProt; Genes & Health Olink is on Explore HT and was matched by OlinkID. SomaScan catalogs were matched by SeqId. Concordance (used for tier 1): posterior probability > 0.5 of the high-correlation component of a two-component Gaussian mixture fitted to SURMOUNT-5 baseline Spearman correlations of matched assay pairs; an assay is concordant if any partner pair is. Eldjarn2023_Tier uses the three tiers of Eldjarn et al., Nature 2023 (doi 10.1038/s41586-023-06563-x): tier 1, a cis-pQTL on both platforms and a concordant partner assay; tier 2, a cis-pQTL on either platform, not tier 1; tier 3, no cis-pQTL on either platform. Other_Platform_Cis_pQTL marks a cis-pQTL for the same protein on the other platform; the number of supporting catalogs is the number listed in Cis_pQTL_Catalogs, and pair-level concordance is in Table S6. HPA v25.1 bulk annotation (CC BY 4.0; data available from v25.proteinatlas.org) uses HGNC ID to Ensembl ID. HPA_Secretome_Location gives HPA's predicted secretion destination; HPA_Enriched_Tissues names the tissues in which HPA reports the gene's RNA as enriched or enhanced (blank when none). GH_Seer_MS_Rho is the Spearman correlation of the same assay with nanoparticle-enrichment mass spectrometry (Seer Proteograph) in Genes & Health (Pietzner et al., Nat Genet 2026; same Olink Explore HT and SomaScan 11K platforms), matched by assay ID; blank when no matched assay or N<100. When an assay matched several Seer protein groups with N≥100, the group quantified in the most participants is shown. HGNC_ID gives the HGNC identifier of each gene in Gene_Symbol, in the same order. Gene_Symbol is blank when the assay has no gene-symbol annotation in our UniProt-based mapping; HGNC_ID is blank when no HGNC identifier is available.",
    "Table S2": "Cis_pQTL_Catalogs lists the independent published catalogs that report at least one cis association (within 1 Mb of the gene, p < 5e-8) for the assay. UKB-PPP, deCODE and Fenland publish only associations passing their own stricter thresholds (p < 1.7e-11, p < 1.8e-9 and p < 1e-11), so for these catalogs support reflects those thresholds. Olink catalogs: UKB-PPP (Sun et al., Nature 2023), JHS (Tahir et al., J Clin Invest 2024), MESA (Nicholas et al., Nat Commun 2026), CKB (Wang et al., Nat Commun 2025) and Genes & Health (Pietzner et al., Nat Genet 2026). SomaScan catalogs: deCODE (Ferkingstad et al., Nat Genet 2021), ARIC (Zhang et al., Nat Genet 2022), Fenland (Pietzner et al., Science 2021), CKB (Pozarickij A. et al., Science Bulletin (2026), doi:10.1016/j.scib.2026.10.060), MESA (Nicholas et al., Nat Commun 2026), Genes & Health (Pietzner et al., Nat Genet 2026), AGES (Gudjonsson et al., Nat Commun 2022) and INTERVAL (Sun et al., Nature 2018). UKB-PPP, JHS, MESA and CKB Olink used Olink Explore 3072 and were matched to Explore HT assays by UniProt; Genes & Health Olink is on Explore HT and was matched by OlinkID. SomaScan catalogs were matched by SeqId. Concordance (used for tier 1): posterior probability > 0.5 of the high-correlation component of a two-component Gaussian mixture fitted to SURMOUNT-5 baseline Spearman correlations of matched assay pairs; an assay is concordant if any partner pair is. Eldjarn2023_Tier uses the three tiers of Eldjarn et al., Nature 2023 (doi 10.1038/s41586-023-06563-x): tier 1, a cis-pQTL on both platforms and a concordant partner assay; tier 2, a cis-pQTL on either platform, not tier 1; tier 3, no cis-pQTL on either platform. Other_Platform_Cis_pQTL marks a cis-pQTL for the same protein on the other platform; the number of supporting catalogs is the number listed in Cis_pQTL_Catalogs, and pair-level concordance is in Table S6. HPA v25.1 bulk annotation (CC BY 4.0; data available from v25.proteinatlas.org) uses HGNC ID to Ensembl ID. HPA_Secretome_Location gives HPA's predicted secretion destination; HPA_Enriched_Tissues names the tissues in which HPA reports the gene's RNA as enriched or enhanced (blank when none). GH_Seer_MS_Rho is the Spearman correlation of the same assay with nanoparticle-enrichment mass spectrometry (Seer Proteograph) in Genes & Health (Pietzner et al., Nat Genet 2026; same Olink Explore HT and SomaScan 11K platforms), matched by assay ID; blank when no matched assay or N<100. When an assay matched several Seer protein groups with N≥100, the group quantified in the most participants is shown. HGNC_ID gives the HGNC identifier of each gene in Gene_Symbol, in the same order. Gene_Symbol is blank when the assay has no gene-symbol annotation in our UniProt-based mapping; HGNC_ID is blank when no HGNC identifier is available.",
    "Table S3": "Data are mean (SD) for continuous variables and n (%) for categorical variables unless otherwise indicated. MTD denotes maximum tolerated dose.",
    "Table S4": "Baseline estimates compare TZP minus SEMA, adjusted for age and sex. log2FC and log2FC_SE are the difference and its standard error on the log₂ protein scale (Olink NPX or log₂-transformed SomaScan RFU); positive values indicate higher baseline levels in TZP. FDR is Benjamini–Hochberg-adjusted across all assays of this platform's baseline comparison.",
    "Table S5": "Baseline estimates compare TZP minus SEMA, adjusted for age and sex. log2FC and log2FC_SE are the difference and its standard error on the log₂ protein scale (Olink NPX or log₂-transformed SomaScan RFU); positive values indicate higher baseline levels in TZP. FDR is Benjamini–Hochberg-adjusted across all assays of this platform's baseline comparison.",
    "Table S6": "Spearman_P values of 0 are below machine precision. Each row pairs assays sharing a UniProt accession. Spearman_Rho and Spearman_P compare baseline Olink plate-control-normalized NPX with SomaScan log₂ RFU across the N paired participants; Prob_Concordant is the posterior probability of the higher-correlation mixture component, and Is_Concordant indicates whether that probability exceeds 0.5. Shared_Gene_Symbol gives the annotated genes of the shared accessions in the order of Shared_UniProt; accessions are separated by “;” and the genes of one accession by “|”.",
    "Table S7": "Baseline proteomics is paired with screening labs; week 24 and week 72 proteomics with labs of the same visit. Lab_Week=−2 denotes screening. Rows require at least ten participants with paired proteomic and laboratory measurements.",
    "Table S8": "Baseline proteomics is paired with screening labs; week 24 and week 72 proteomics with labs of the same visit. Spearman_P values of 0 are below machine precision. Lab_Week=−2 denotes screening. Rows require at least ten participants with paired proteomic and laboratory measurements.",
    "Table S9": "Arm identifies TZP or SEMA; each assay has one row per arm. PCBL_Wk24 and PCBL_Wk72 are within-arm percent changes from baseline; SE_Wk24 and SE_Wk72 are their standard errors in percentage points. log2FC_Wk72vsWk24 and SE_Wk72vsWk24 are the within-arm week-72 minus week-24 change and its standard error on the log₂ protein scale. FDR is Benjamini–Hochberg-adjusted across assays separately for each arm and comparison. Trajectory identifies Progressive ↑/↓, Sustained ↑/↓, Transient ↑/↓, Late-onset ↑/↓, Reversal ↑↓/↓↑, or Null, using the within-arm week-24, week-72 and week-72-minus-week-24 tests.",
    "Table S10": "Arm identifies TZP or SEMA; each assay has one row per arm. PCBL_Wk24 and PCBL_Wk72 are within-arm percent changes from baseline; SE_Wk24 and SE_Wk72 are their standard errors in percentage points. log2FC_Wk72vsWk24 and SE_Wk72vsWk24 are the within-arm week-72 minus week-24 change and its standard error on the log₂ protein scale. FDR is Benjamini–Hochberg-adjusted across assays separately for each arm and comparison. Trajectory identifies Progressive ↑/↓, Sustained ↑/↓, Transient ↑/↓, Late-onset ↑/↓, Reversal ↑↓/↓↑, or Null, using the within-arm week-24, week-72 and week-72-minus-week-24 tests.",
    "Table S11": "At each week, log2FC is the adjusted TZP minus SEMA contrast and log2FC_SE its standard error on the log₂ protein scale; positive values indicate higher levels with TZP. The model adjusts for age, sex and baseline protein. FDR is Benjamini–Hochberg-adjusted across assays separately within platform and week.",
    "Table S12": "At each week, log2FC is the adjusted TZP minus SEMA contrast and log2FC_SE its standard error on the log₂ protein scale; positive values indicate higher levels with TZP. The model adjusts for age, sex and baseline protein. FDR is Benjamini–Hochberg-adjusted across assays separately within platform and week.",
    "Table S13": "MMRM of change from baseline with treatment × visit × sex interaction, adjusted for baseline protein and age, separate unstructured covariance by sex; Kenward–Roger df. Interaction = (TZP − SEMA) in females minus (TZP − SEMA) in males; BH FDR within platform and week. Within-sex contrasts are descriptive. The female and male TZP−SEMA estimates, their standard errors, the female-minus-male interaction and its standard error are all on the log₂ protein scale. Direction compares the absolute magnitudes of the two within-sex estimates: opposite_sign means nonzero estimates with opposite signs; larger_in_F and smaller_in_F mean the absolute female estimate is larger or smaller, respectively; equal denotes equal magnitudes. The within-sex descriptive FDRs are adjusted across assays separately for each sex and week.",
    "Table S14": "MMRM of change from baseline with treatment × visit × sex interaction, adjusted for baseline protein and age, separate unstructured covariance by sex; Kenward–Roger df. Interaction = (TZP − SEMA) in females minus (TZP − SEMA) in males; BH FDR within platform and week. Within-sex contrasts are descriptive. The female and male TZP−SEMA estimates, their standard errors, the female-minus-male interaction and its standard error are all on the log₂ protein scale. Direction compares the absolute magnitudes of the two within-sex estimates: opposite_sign means nonzero estimates with opposite signs; larger_in_F and smaller_in_F mean the absolute female estimate is larger or smaller, respectively; equal denotes equal magnitudes. The within-sex descriptive FDRs are adjusted across assays separately for each sex and week.",
    "Table S17": f"Prop_Mediated is the proportion mediated, ACME divided by the total effect, with its 95% bootstrap percentile interval (Prop_Mediated_CI_lower, Prop_Mediated_CI_upper; {DEFAULT_SIMS:,} draws). It is unstable, and its interval wide, when the total effect is near zero.",
    "Table S18": f"Prop_Mediated is the proportion mediated, ACME divided by the total effect, with its 95% bootstrap percentile interval (Prop_Mediated_CI_lower, Prop_Mediated_CI_upper; {DEFAULT_SIMS:,} draws). It is unstable, and its interval wide, when the total effect is near zero.",
    "Table S15": "N_Measured, number of genes of the set measured on the platform and used in cameraPR; Lead_Genes lists the genes contributing most to the enrichment. Display_Label gives the figure label for terms included in the figure term mappings; it is blank otherwise.",
    "Table S16": "The ORA query and its overlap are gene-level: each gene's P value combines its assays' P values by Simes, followed by BH adjustment over genes (FDR < 0.05). Lead_Genes lists overlap genes that also have an assay at FDR < 0.05 in the same platform and week; N_Lead counts those listed genes and can be smaller than the overlap used to calculate the ORA P value. Display_Label gives the figure label for terms included in the figure term mappings; it is blank otherwise.",
}

# (tab_name, description, module, sheet_index_within_module)
MANIFEST = [
    ("Table S1", "Olink Explore HT assay annotations", "suppl_assay_qc", 0),
    ("Table S2", "SomaScan 11K assay annotations", "suppl_assay_qc", 1),
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
        "Sex × treatment interaction, TZP vs SEMA (Olink)",
        "suppl_sex_interaction",
        0,
    ),
    (
        "Table S14",
        "Sex × treatment interaction, TZP vs SEMA (SomaScan)",
        "suppl_sex_interaction",
        1,
    ),
    (
        "Table S15",
        "Pathway enrichment — cameraPR (TZP vs SEMA)",
        "suppl_pathway_enrichment",
        0,
    ),
    (
        "Table S16",
        "Pathway enrichment — ORA (TZP vs SEMA)",
        "suppl_pathway_enrichment",
        1,
    ),
    ("Table S17", "Weight-mediation results, TZP vs SEMA (Olink)", "suppl_mediation", 0),
    (
        "Table S18",
        "Weight-mediation results, TZP vs SEMA (SomaScan)",
        "suppl_mediation",
        1,
    ),
]


def _load_table_module(name: str):
    return importlib.import_module(f"tables.{name}")


def _fix_timestamps(path: Path) -> None:
    """Rewrite the workbook with every zip member dated ``WORKBOOK_TIMESTAMP``.

    openpyxl overwrites ``properties.modified`` with the current time when it saves, and zip
    members carry their write time; this sets the modified property back and the member times
    to the fixed epoch, leaving every member's bytes otherwise as written.
    """
    stamp = WORKBOOK_TIMESTAMP.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "docProps/core.xml":
                data, n = _MODIFIED.subn(rb"\g<1>" + stamp + rb"\g<2>", data)
                assert n == 1, "docProps/core.xml has no dcterms:modified"
            member = zipfile.ZipInfo(info.filename, date_time=_ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = info.external_attr
            target.writestr(member, data)
    path.write_bytes(buffer.getvalue())


def build_workbook(out_path: Path = DEFAULT_OUTPUT) -> Path:
    """Build every supplementary table and write them, styled, to one Excel workbook.

    Checks first that the ``analysis/outputs/`` files the table modules read are present
    (``require_stage_outputs``), then calls each module's ``build()`` once, writes a
    Contents sheet (table of contents and abbreviations) and one sheet per ``MANIFEST`` row,
    each with its title and any ``FOOTNOTES`` entry. The workbook properties and zip member times
    are fixed (``WORKBOOK_TIMESTAMP``), so the same tables give the same bytes.

    Args:
        out_path: Workbook path; the suffix is forced to ``.xlsx``.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If a manifest stage output a table reads is missing (the message
            names the stage that writes it).
    """
    from analysis.tools.manifest_stages import require_stage_outputs

    require_stage_outputs("tables")
    out_path = out_path.with_suffix(".xlsx")
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
        wb.properties.created = WORKBOOK_TIMESTAMP
        wb.properties.modified = WORKBOOK_TIMESTAMP
        toc_entries = []
        for tab_name, description, _, _ in MANIFEST:
            ws = wb[tab_name]
            title = f"{tab_name}: {description}"
            style.style_sheet(ws, title, footnote=FOOTNOTES.get(tab_name))
            toc_entries.append((tab_name, description))

        toc_ws = wb["Contents"]
        style.create_toc(toc_ws, toc_entries, abbreviations=ABBREVIATIONS)

    _fix_timestamps(out_path)
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
    """Command-line entry: ``--list-tables`` prints the tab order; otherwise builds the workbook.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        0.
    """
    args = _parse_args(argv)
    if args.list_tables:
        _print_manifest()
        return 0
    started = time.time()
    path = build_workbook(args.output)
    # Record provenance only for the default output path.
    if path == DEFAULT_OUTPUT:
        record_build("tables", outputs=[str(path)], scripts=[], started=started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
