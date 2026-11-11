# Reproducibility

What a rerun depends on: the environment, what makes it give the same result, the record each
stage writes, the inputs this repository cannot produce, and the thread and worker settings.
[README.md](README.md) says what to run and in what order.

## Environment

The environment is defined for `linux-64` only (`pixi.toml`). Every package is pinned to the
version `pixi.lock` records, so `pixi install --locked` installs the same package set from
conda-forge and bioconda; `somadata` is the one PyPI package. `pixi run check-r` fails unless
the eight R packages below are installed at exactly these versions. `pixi run setup-fonts`
downloads the Inter 4.0 release and checks its SHA-256 before installing it into the
environment's matplotlib font directory (nothing outside `.pixi/` is written); the figures raise
if Inter is not registered with matplotlib.

Commands run through `pixi run` or inside `pixi shell`, which set `PYTHONPATH` to the
repository root. Any other interpreter (an IDE or notebook kernel on `.pixi/envs/default`)
needs `PYTHONPATH` set to the repository root.

| Software | Version |
|---|---|
| Python | 3.13.13 |
| R | 4.5.3 |
| polars | 1.40.1 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| matplotlib-base | 3.10.9 |
| rpy2 | 3.6.7 |
| somadata | 1.2.5 |
| plink2 | 2.0.0a.6.9 |
| mmrm (R) | 0.3.19 |
| emmeans (R) | 2.0.3 |
| mediation (R) | 4.5.1 |
| limma (Bioconductor) | 3.66.0 |
| gdsfmt (Bioconductor) | 1.46.0 |
| SNPRelate (Bioconductor) | 1.44.0 |
| lme4 (R) | 2.0-1 |
| arrow (R) | 24.0.0 |

The R `mediation` package is a reference implementation: the mediation results come from the
numpy bootstrap in `analysis/mediation/`, and no stage calls the R package.

## Determinism

- **Seeds.** Every stochastic step uses seed 0 (`analysis.statistics.SEED`): the mediation
  bootstrap, the platform-concordance Gaussian mixture, the pathway CLI's `random` and numpy
  state, and the QC's PCA and age-model fits.
- **Fig 3b rotation null.** The disease-reversal P values are Monte Carlo: 1,000,000 rotation
  draws per disease (`ROTATION_DRAWS`) from a fresh `numpy.random.default_rng(0)` for each
  signature (`analysis/disease_reversal.py`). A rerun with the same inputs gives the same draws;
  the add-one P value has Monte Carlo standard error sqrt(P(1 − P) / 1,000,000).
- **Row order.** The MMRM sorts its inputs and pooled results on identifier columns ending in
  `USUBJID`, and sets factor levels explicitly (SEMA and week 24 as references), so a rerun
  writes the same rows in the same order. The pathway batch writes its (library,
  method) results in task order, not completion order.
- **BLAS threads.** The MMRM, sex-interaction and mediation pools start their workers through
  `analysis.parallel.forkserver_pool`, and the pathway batch calls `pin_blas_threads` before
  starting its workers, so each worker runs single-threaded BLAS. The Fig 3b rotation null
  (`analysis/pipelines/run_disease_reversal.py`) runs in one process and calls `pin_blas_threads`
  before it imports numpy, so its BLAS is single-threaded whatever the host.
- **Mediation reuse.** The mediation stage reuses an existing output or per-visit checkpoint
  only if the fingerprint file beside it matches the run's configuration, input rows,
  annotation and marker × visit keys; otherwise it exits and names what differs.
- **Figures.** Label placement uses adjustText with a fixed seed and an iteration cap, not a
  time budget (`figures/_internal/_text.py`), and `save_figure` writes PDFs without creation or
  modification dates (`figures/_internal/_theme.py`).
- **Supplementary workbook.** `openpyxl` stamps the workbook's created and modified properties
  with the current time and `zipfile` stamps each member with the write time, so two builds of
  the same tables would differ in their bytes. `tables/build_all.py` fixes the workbook properties
  and every member time to 1980-01-01, so the workbook's bytes are a function of its cells and
  styles.

## Provenance

`pixi run reproduce` runs every stage of `analysis/manifest.yaml` in order through
`analysis/tools/run_stage.py`, then the figure build and the table build. A stage's command is the
manifest's, run through `pixi run --frozen` with each `<paths.yaml:key>` filled from the path
configuration (every stage's placeholders are such keys); a command holding another kind of
`<placeholder>` could not be filled in and would be run by hand and then recorded. After each
stage the runner writes `analysis/outputs/provenance/<stage>.json` (gitignored with the rest of
`analysis/outputs/`) holding:

- the command as run, and the manifest's command;
- the git commit, the branch, whether tracked files differed from it (`dirty`, with the paths);
- the sha256 of `pixi.lock`;
- the Python, R and platform versions;
- the wall-clock duration and the peak resident set of the stage's largest process;
- a fingerprint of every declared input, taken before the stage ran, and of every declared
  output, taken after it.

A stage that fails, or that exits 0 without writing a declared output, leaves no record (its old
record is removed first). `pixi run stage run <id> [<id> ...]` runs the stages it names in the
order given. `pixi run reproduce` skips a stage only when its record and declared outputs exist,
its declared inputs still hash the same as the record fingerprinted them (contents hashed; a
record whose input list differs from the manifest's is not up to date) and no stage that
produces one of them ran in the same invocation; the figure and table builds always rebuild; `--force` runs everything; `--dry-run` prints the plan (deciding it hashes every recorded input). `pixi run paper` rebuilds the
figures and tables from existing outputs. The mediation stage resumes from its own output only
when the fingerprint file beside it matches (see Determinism); `reproduce --force` does not
delete that output, so a from-scratch mediation run starts by removing the CSV and its
checkpoints.

`reproduce` does not check stage code when deciding whether to skip. Run
`pixi run provenance-check`, rerun the stages it reports with `pixi run stage run <stage>`,
then run `pixi run provenance-check` again.

The figure build (`figures/build_figures.py`) and the table build (`tables/build_all.py`) write
a record when run on their own (`figures.json`, or `figures.<which>.json` for one registry, and
`tables.json`). Their inputs are the outputs of the stages whose `downstream` names them in the
manifest, fingerprinted when the build finishes; their outputs are the PDFs and the workbook.
Under `reproduce` the runner writes those records itself. A workbook built to another path with
`--output` is not recorded.

**Fingerprints.** A file of at most 1 GiB is hashed in full. A larger file is fingerprinted by
its size, its modification time and its first and last 64 MiB, and the entry records the method
(`size+mtime+head+tail sha256`), so a change in its unread middle that keeps size and mtime is
not detected. `provenance-check --quick` reads no contents. A directory records every file
under it with its own fingerprint and a sha256 over the sorted (path, size, file hash) lines, so
a changed, added or removed file is named. Hashes are of the bytes, so a rerun that writes the same table with another row order or
float rounding is a different fingerprint.

**Checking.** `pixi run provenance-check` fingerprints every record's declared inputs and
outputs as they are now and compares them with the record, naming each changed, missing or
added file; it also compares the stage's code between the record's commit and HEAD: a Python
file of the stage script's import closure (for the figure and table builds, the registered
figure scripts and table modules too) whose AST differs once docstrings are dropped, and any
added, removed or non-Python closure file that differs, makes the record stale, while a
comment- or docstring-only edit is a note. It exits 1 if any record is stale, if a stage whose declared outputs exist has no
record (those files have no record of what produced them), and if there are no records at all.
A stage with no record and no outputs is a note (it has not run here). `--quick` compares size
and modification time and reads no contents, so it reports a file rewritten with the same
contents as changed; the full check settles it. A record from a dirty tree (its code is compared with its commit), from another commit than HEAD, with a
`pixi.lock` that has since changed, or written after the fact is noted in the report; those
notes do not fail the check.

**Records written after the fact.** `pixi run stage record <id>` writes the record of a stage
that was run by hand, from the files as they are. It sets `recorded_after_the_fact: true` and
carries a `note`, because the record then shows what the files were when recorded and not that
this commit produced them; its command, duration and peak memory are null and its git entry is
the checkout at the time of recording, not of the run. `stage record <id> --allow-missing <ref>`
records a declared output that is absent under `outputs_missing_when_recorded`; a record whose
outputs are listed as missing is never taken as up to date by `reproduce`.

## Inputs the pipeline does not produce

- **Trial data**, through `paths.yaml`: the Olink NPX file and SomaScan ADAT (`surmount5_*`),
  the ADaM tables (`pheno_root`) and the genotype inputs named in `paths.example.yaml`
  (`geno_root`, `qa_root`).
- **QC reference tables and cis-pQTL catalogs** (`qc_reference` stage). The two published
  supplementary tables are downloaded from the publishers. The cis-pQTL catalogs are built here from
  their publications' tables (fourteen parquets: the thirteen catalogs of the QC and Tables S1-S2,
  five Olink and eight SomaScan, plus the 2025 CKB SomaScan catalog, which supplies the UniProt
  accessions for MESA SomaScan). Sources: Zenodo records (MESA, the CKB Atlas), publisher
  supplements, the Fenland Science supplementary workbook (a browser download, passed with
  `--fenland-xlsx` from `paths.yaml:fenland_science_xlsx`), and Ensembl's GRCh37-to-GRCh38
  chain and release 116 GRCh38 variation files for the Fenland, INTERVAL and AGES liftovers (`analysis/external/liftover.py`).
  Every download is SHA-256 pinned, the Ensembl variation files are read through pinned indexes, and each parsed catalog is
  checked against the content digest pinned in `analysis/external/pqtl_catalogs.py`
  (`CATALOG_FRAME_SHA256`); a mismatch raises before anything is written. The Fenland allele
  check reads the trial PVARs named by `paths.yaml:surmount5_fenland_pvar_pattern`. Without
  these inputs the QC cannot run and Tables S1–S2 cannot be built.
- **Pathway inputs.** The assay-to-gene mapping is the assay annotation (stage
  `vendor_assay_tables`), and the seven Enrichr gene-set libraries are downloaded by the
  `gene_sets` stage into `data/external/gene_sets/` (the pathway CLI's `--cache-dir` default),
  SHA-256 pinned in `analysis/pathway/constants.py`. Enrichr revises libraries in place, so a
  fresh download fails the digest check unless Enrichr still serves the same bytes.
- **Downloaded on first use, SHA-256 pinned**: the HPA pan-disease blood atlas table, the
  Ensembl release 116 gene-to-UniProt cross-references (Fig. 3b), the Maretty et al. 2025 STEP
  supplement, and the Hormone Cell Atlas pancreas h5ad.
- **Vendor assay lists**: Olink's public Explore HT assay list (fetched without a login) and the
  SomaLogic SomaScan 11K annotated menu (SL00000906 Rev 10; a browser download that needs a
  SomaLogic account). Both are SHA-256 pinned and checked before they are read. The stage
  fetches the Olink list and takes the menu with `--menu` from `paths.yaml:somascan_menu_xlsx`.
  The `vendor_assay_tables` stage builds
  the assay annotation from them and the tracked UniProt gene-update layer
  (`analysis/reference/assay_gene_updates.tsv`, written from UniProt release 2026_03 and the
  pinned HGNC complete set by `analysis/external/assay_gene_updates.py`; the build itself never
  calls UniProt). The build fetches the pinned HGNC monthly release and writes the HGNC
  alias table beside the annotation; the `hcop` stage fetches the pinned HCOP human-mouse orthologs.
  Neither release nor the alias table is tracked. The pathway step translates
  gene-set library members to the annotation's symbols through the alias table, and CellMarker's
  mouse sets through the HCOP orthologs. The UniProt
  map, the pathway step, Fig. 1 and Fig. 3b read the annotation.
- **Browser downloads**: the Science Table S6A workbook (`paths.yaml:science_aeb2672_xlsx`) and,
  above, the Fenland workbook (`fenland_science_xlsx`) and the SomaLogic menu
  (`somascan_menu_xlsx`). Each digest is checked before
  the file is read.
- **Derived tables with their own stage**: `data/external/science_aeb2672_tableS6A.parquet` and
  `data/external/hormone_atlas_pancreas_gpcr.parquet`, and
  `analysis/outputs/uniprot_map.parquet` and `analysis/outputs/qc_exclusion_breakdown.parquet`;
  each is written by its manifest stage. `data/external/README.md` lists every third-party
  source with its citation and digest.

## Compute

Pin BLAS to one thread for disease reversal. `--n-jobs -1` on the MMRM, mediation and pathway
stages uses every CPU but one.
