# QC

`analysis/qc/` is the sample and assay QC for both platforms. Each platform run is configured by its own file:
`config_olink.toml` (run `surmount5_olink`) or `config_soma.toml` (run `surmount5_soma`).

Steps: vendor sample and assay QC, CV, LOD and detection, SomaScan saturation and
signal-to-background, Olink duplicate assays, PCA and median/IQR outliers, sex and age
concordance (WARN only), pQTL genetic identity with reassignment, and the clinical rules
(treatment window, analysis visits, baseline). No QC report is written.

Value layers: the Olink QC reads `PCNormalizedNPX` (`data_col`). The SomaScan QC uses the
non-ANML RFU. Its pQTL score gives each selected assay equal weight (`weight_exponent = 0.0`) over eight cis catalogs: deCODE, ARIC, Fenland,
CKB Atlas (`ckb_soma`, the 2026 CKB Atlas), MESA, Genes & Health, AGES and INTERVAL.
Tables S1 and S2 read the same catalogs. The sign-exclusion list (`SIGN_EXCLUSIONS`) is
what `derive_sign_exclusions` returns for these catalogs.

## Running

```bash
pixi run python analysis/pipelines/run_qc.py --study both    # or surmount5_olink | surmount5_soma
pixi run python analysis/pipelines/run_qc.py --study surmount5_olink --output-dir <scratch>
```

`analysis/qc/study.py` (`run_study`) turns a config into driver arguments, and
`analysis/qc/driver.py` runs the steps.

## Inputs

The platform configurations resolve their inputs through `paths.yaml`; see
`paths.example.yaml` for the required settings.
- Reference tables and pQTL catalogs: fetched from their publications and parsed by
  `analysis/external/qc_reference.py` (catalogs: `analysis/external/pqtl_catalogs.py`) into
  `analysis/qc/data/` and `data/external/pqtl/parsed/`. Override these locations with
  `qc_reference_dir` and `pqtl_catalog_dir`. The stage fetches the 13 pQTL catalogs from
  their publications and parses each one, including the CKB Atlas, AGES and INTERVAL SomaScan
  catalogs, checking the downloads against byte and content pins.

## Corrections

Each run reads its `[[corrections]]` from the TOML its `<study>_qc_corrections` paths key names;
the TOML names tubes and participants by the SHA-256 of the ID, never by the ID.
`analysis/qc/config.py` (`CORRECTION_FIELDS`) lists the types. The manifest build applies the tube-level ones (`exclude`, `reassign`, `visit_set` and the rest), and the QC applies
`subject_set` and `expected_sex` to ADSL. A `dna_relink` entry (`analysis/qc/dna_relink.py`)
says the genome filed under one participant (`from_usubjid_sha256`) belongs to another
(`usubjid_sha256`): the pQTL identity check labels that genome with its owner, so the owner's tubes
are scored against it and the participant it was filed under is scored as ungenotyped. The KING
table and the ancestry PCs stay keyed by the PSAM IID. A relink is subject-level, so the same entry
is in both platform TOMLs, and the pharmacogenomics stage applies it to the linking table too. An owner that already has a genome raises, a swap included.

## Outputs

Each run writes to `<qc_root>/surmount5_<platform>/`: the sample and assay QC summaries,
the exclusion table, identity events, pQTL scores, the DNA link, `manifest.parquet`, and
the long value table. The Olink long table is `npx_long.parquet`, with `NPX` and
`PCNormalizedNPX` and their LODs. The SomaScan one is `rfu_long.parquet`, with the non-ANML `RFU`.

**Report-only vendor annotations.** Every sample summary carries nullable string columns
`SampleNotes`, `AssayNotes`, `AliquotingNotes` and `VOLUME_NOTE`. SomaScan joins the first
three verbatim (blank strings become null) from the ADAT sample metadata by `SampleId`; Olink
joins the manifest `VOLUME_NOTE` by `SampleID`. Identical repeated annotations for a tube
collapse; conflicting annotations on one tube key raise. No annotation changes a flag, an
exclusion, a status or a threshold.

Code outside `analysis/qc/` reads these files only through `analysis/qc_contract.py`. That
module defines the file names, required columns, the cohort rule (`Status != "FAIL"`) and
`FLAG_REGISTRY`.

