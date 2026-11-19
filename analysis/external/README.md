# External inputs

`analysis/external/` fetches, verifies and prepares the third-party inputs the analysis
needs. Each downloaded input is pinned by SHA-256. The digest lives in the fetching module, and
every use re-verifies the file, cached copies included; a mismatch raises. The Ensembl variation
files of the hg19 lift are read over HTTPS through SHA-256-pinned indexes, and each parsed catalog
is checked against its content digest. Downloads are
gitignored. The preparation scripts write the external-input parquets in `data/external/`.
`data/external/README.md` gives every source with its citation and digest.

| Module | What it does | Command | Writes |
|---|---|---|---|
| `qc_reference.py` | Downloads the sex-association (Koprulu 2025, Table S2) and ProtAge (Argentieri 2024, Table S1) workbooks and writes the TSVs the QC parses; builds 14 cis-pQTL parquets (the 13 catalogs read by the QC and Tables S1-S2, 5 Olink and 8 SomaScan, and the 2025 CKB SomaScan catalog that supplies UniProt to MESA SomaScan) with `pqtl_catalogs.py` | `pixi run stage run qc_reference` (uses `paths.yaml` for the genotype directory, PVAR pattern and Fenland workbook) | `analysis/qc/data/*.tsv` (`qc_reference_dir`), `data/external/pqtl/parsed/` (`pqtl_catalog_dir`), downloads in `data/external/qc_reference/` and `data/external/pqtl/raw/` |
| `pqtl_catalogs.py` | Fetches the published pQTL tables (SHA-256 pinned; Fenland's Science workbook is a manual download read from `data/vendor/` (create it and put the browser-downloaded workbook there; it does not ship) or `--fenland-xlsx`, which the stage passes from `paths.yaml` `fenland_science_xlsx`), parses the catalogs in dependency order (CKB Atlas 2026, INTERVAL and AGES included; the hg19 Fenland, AGES and INTERVAL positions are lifted to hg38 by `liftover.py`), and checks each parsed catalog against its pinned content digest before writing it | (library; run through `qc_reference.py`) | `data/external/pqtl/parsed/*.parquet` |
| `pqtl_parsers.py` | One parser per publication's table, into the QC catalog schema | (library) | |
| `liftover.py` | hg19 to hg38 liftover with Ensembl's GRCh37-to-GRCh38 chain, and an rsID lookup in Ensembl's release 116 GRCh38 variation files for a variant the chain does not cover | (library) | `data/external/ensembl_mapping/` |
| `fenland_liftover.py` | The Fenland lift and its allele check against the trial PVARs | (library) | |
| `gene_sets.py` | Stage `gene_sets`: downloads the seven Enrichr gene-set libraries the pathway step reads, skips a library already present with its pinned digest and raises on a present file or a download with another digest | `pixi run python analysis/external/gene_sets.py` | `data/external/gene_sets/*.json` |
| `hpa_disease_atlas.py` | Fetches the HPA v25.1 blood disease differential-expression zip; `read_disease_de` reads it for Fig 3b | `pixi run python -m analysis.external.hpa_disease_atlas` | `data/external/hpa/blood_pea_disease_de.tsv.zip` |
| `recompute_hormone_atlas_pancreas.py` | Fetches the Hormone Cell Atlas pancreas h5ad (about 911 MB) and recomputes per-cell-type GIPR / GLP1R expression | `pixi run python analysis/external/recompute_hormone_atlas_pancreas.py` | `data/external/hormone_atlas_pancreas_gpcr.parquet` (ED Fig 7) |
| `science_aeb2672_tableS6A.py` | Extracts Table S6A from the Science supplementary workbook, which is downloaded in a browser. | `pixi run python analysis/external/science_aeb2672_tableS6A.py --xlsx <workbook>` | `data/external/science_aeb2672_tableS6A.parquet` (Fig 5a) |
| `pinned.py` | `fetch_pinned` and `check_sha256`: streams a download to a `.part` file and renames it only once it verifies | (library) | |
| `vendor_assay_tables.py` | Builds the assay annotation from the vendor assay lists (read from `data/vendor/` (create it and put the browser-downloaded workbook there; it does not ship) or `--menu` / `--olink-list`, SHA-256 pinned; the stage passes `--menu` from `paths.yaml` `somascan_menu_xlsx`) and the update layer; fetches the pinned HGNC release and writes the HGNC alias table beside the annotation | `pixi run stage run vendor_assay_tables` (passes `--menu` from `paths.yaml`) | `data/external/vendor_assay_tables/` (annotation, wide tables, `hgnc_aliases.tsv`, inputs record), `data/external/hgnc/` |
| `assay_gene_updates.py` | Writes the UniProt gene-update layer from UniProt release 2026_03 and the pinned HGNC release, and checks the HGNC IDs the build derives (not a stage; rerun only to re-pin) | `pixi run python analysis/external/assay_gene_updates.py` | `analysis/reference/assay_gene_updates.tsv` |
| `hgnc_aliases.py` | The HGNC alias table: our symbols' HGNC IDs from the release alone, and the approved, previous and alias names of those IDs, matched case-insensitively (library; built by `vendor_assay_tables.py`) | (library) | |
| `hcop.py` | Fetches HGNC's HCOP human-mouse orthologs (generation-pinned URL, SHA-256) for CellMarker's mouse sets; `--path` for a saved copy | `pixi run python analysis/external/hcop.py` | `data/external/hcop/human_mouse_hcop_fifteen_column.txt.gz` |

The STEP supplementary workbook (Maretty et al. 2025) is fetched through `fetch_pinned` by
`analysis/pipelines/maretty_step_data.py`. It is cached as `data/external/maretty_2025_supp.xlsx`.

The `paths.yaml` keys `qc_reference_dir` and `pqtl_catalog_dir` are optional overrides
(`analysis/qc_paths.py`). Without them the defaults above apply.

The QC and Tables S1-S2 cannot run without the QC reference tables. Every QC reference input
downloads with a plain HTTP GET except the Fenland workbook, a browser download whose path is
`paths.yaml` `fenland_science_xlsx`; the Fenland allele check also reads the trial PVARs
(`data/external/README.md`).
