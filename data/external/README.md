# data/external

Third-party inputs. Each is written by a script or fetched, pinned by SHA-256 and verified on every
use. The vendor assay lists and the Fenland Science supplement are read from `data/vendor/` by default, or from
the paths `paths.yaml` names (`fenland_science_xlsx`, `somascan_menu_xlsx`), which the manifest stages pass as
`--fenland-xlsx` and `--menu` (see below). `data/vendor/` does not ship: create it and put each
browser-downloaded workbook there.
The digests live in the fetching code, are verified on every use (a cached copy included), and a
mismatch raises.

| File | Producer | Pinned input (SHA-256) | Read by |
|---|---|---|---|
| `science_aeb2672_tableS6A.parquet` | `analysis/external/science_aeb2672_tableS6A.py --xlsx <workbook>` | `science.aeb2672_tables_s1_to_s6.xlsx`, `c235a13f04ac71e7066e1e082221766f66785d13e7fd940b0b7eaef130aa6077` | `figures/_internal/_gip_panel.py` (Fig 5a) |
| `hormone_atlas_pancreas_gpcr.parquet` | `analysis/external/recompute_hormone_atlas_pancreas.py` | `Pancreas_annotated.h5ad`, `a1e638e8a1ba7ce2373d43f94c8e5d0ae89f083de9e6780d6f6146a2cfd278e3` | `figures/supp_fig_gipr_glp1r_celltype.py` |
| `maretty_2025_supp.xlsx` (gitignored) | `analysis/pipelines/maretty_step_data.py` | `4fd2be8d5488c3617384bf1763dd31f015a319e0e47a427d86b59db4a561a49d` | the STEP parquet, Fig 5, ED 5, Supp 6 |
| `hpa/blood_pea_disease_de.tsv.zip` (gitignored) | `analysis/external/hpa_disease_atlas.py` | `6f843164366d6863e63896c01e07d844e8a862c4c9bedcbc4a2cd015e4b23992` | Fig 3b |
| `vendor_assay_tables/` (gitignored) | `analysis/external/vendor_assay_tables.py` | the Olink Explore HT assay list `olink-explore-ht-assay-list.xlsx`, `a48a0f7d364dfe0e9e66357954ad2d50be40f4d4a4685e894192c2a945fcb6a8`, and the SomaLogic annotated menu `SL00000906_SomaScan_11K_v5.0_Plasma_Serum_Annotated_Menu.xlsx`, `4284504d27bee6726e00ca3561913c12a01b9e2da9cd1ea489f73dd87c0102bf` (both in `data/vendor/`); the UniProt update layer `analysis/reference/assay_gene_updates.tsv` | the UniProt map, the pathway step, Fig. 1, Fig. 3b |
| `ensembl/Homo_sapiens.GRCh38.116.uniprot.tsv.gz` (gitignored) | `analysis/external/ensembl_uniprot.py` | `f1e26db23b0771f7a5778067af20683de63d863b1c0834aa564d5d036ec85864` | Fig. 3b (HPA symbol aliases) |
| `gene_sets/` (written by the `gene_sets` stage) | `analysis/external/gene_sets.py` (stage `gene_sets`) | the seven Enrichr libraries, one JSON each; digests in `GENE_SET_SHA256` of `analysis/pathway/constants.py` | the pathway step |
| `qc_reference/` (gitignored) | `analysis/external/qc_reference.py` | see the module | the QC sex and age checks |
| `pqtl/raw/`, `pqtl/parsed/` (gitignored) | `analysis/external/qc_reference.py` (`pqtl_catalogs.py`) | below | the QC identity check |
| `ensembl_mapping/` (gitignored) | `analysis/external/liftover.py` | Ensembl `GRCh37_to_GRCh38.chain.gz`, `351de3cd4a01d9fcffd38881981767b697090d2eba876740891b96d5c546b100`, and the 24 GRCh38 variation VCF indexes (`VARIATION_INDEX_SHA256`) | the hg19 to hg38 lift of Fenland, INTERVAL and AGES |

## science_aeb2672_tableS6A.parquet

- **Source.** Fei, Huang-Doran et al., "A human hormone cell atlas", Science 2026,
  doi:10.1126/science.aeb2672, supplementary Table S6A, "Hormone receiving strengths at the
  fine-grained cell-type level across all 47 tissues" (sheet `Table S6A` of
  `science.aeb2672_tables_s1_to_s6.xlsx`). science.org refuses scripted downloads, so the workbook
  is fetched in a browser and is not stored in the repository.
- **Columns kept**, in sheet order: `Hormone_short`, `Tissue`,
  `Tissue level fine-grained cell type (celltype_level2)`, `Strength`; all text. Dropped: `Tier`,
  `Source Assay`, `Cell count from scRNA-seq`, `Cell count from snRNA-seq`.
- **Transformation.** The title and blank rows above the header are dropped, and `Strength` is the
  sheet's float rounded to nine decimals and written without trailing zeros.
- **Check.** `analysis/external/science_aeb2672_tableS6A.py` reads the pinned workbook and writes the
  selected columns.

## hormone_atlas_pancreas_gpcr.parquet

Per-cell-type GIPR and GLP1R mean log1p expression and percent expressing, recomputed from the
Hormone Cell Atlas pancreas h5ad (`https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/Pancreas_annotated.h5ad`,
955,186,752 bytes). The fetcher downloads it to `hormone_atlas/` (gitignored) and verifies its pinned digest
before reading.

## Assay annotation (vendor assay lists + UniProt update layer)

`analysis/external/vendor_assay_tables.py` (manifest stage `vendor_assay_tables`) builds the assay annotation
of both platforms in two layers.

**Base layer: the vendor files, as published.** Both are read from `data/vendor/` (or `--menu` /
`--olink-list`):

- **Olink**: Olink Explore HT assay list (Doc 1562), v2026-06-24, SHA-256
  `a48a0f7d364dfe0e9e66357954ad2d50be40f4d4a4685e894192c2a945fcb6a8`, downloaded without a login from
  `https://7074596.fs1.hubspotusercontent-na1.net/hubfs/7074596/000-documents/10-excel%20file/olink-explore-ht-assay-list.xlsx`
  (the stage fetches it into `data/vendor/` when absent). Columns `Olink ID*`, `UniProt ID`, `Gene name`; several
  targets are joined with `_` in both.
- **SomaScan**: SomaScan 11K v5.0 plasma/serum annotated menu, `SL00000906_SomaScan_11K_v5.0_Plasma_Serum_Annotated_Menu.xlsx`,
  SL00000906 Rev 10 (2026-03), SHA-256 `4284504d27bee6726e00ca3561913c12a01b9e2da9cd1ea489f73dd87c0102bf`. It is
  marked "SOMALOGIC CONFIDENTIAL INFORMATION" and needs a SomaLogic account: download it from
  `https://menu.somalogic.com` (file downloads; a signed link expires, so none is recorded) and save it to
  `data/vendor/` or pass `--menu <path>` (the stage passes `paths.yaml` `somascan_menu_xlsx`). Sheet `Annotations`, rows with `Type` Protein and `Organism` Human;
  columns `SeqId`, `UniProt ID`, `Entrez Gene Name`, `HGNC ID`, `Entrez Gene ID` (` | ` between values).

Columns are selected by name; each file's digest is checked before it is read.

**Update layer: `analysis/reference/assay_gene_updates.tsv`** (columns `platform`, `assay_id`,
`vendor_accession`, `accession`, `vendor_gene`, `updated_gene`, `reason`, `uniprot_release`). Written by
`pixi run python analysis/external/assay_gene_updates.py`, which queries UniProt REST for every accession of
both vendor files and records the UniProt release (`X-UniProt-Release`) on every row; the build never calls
UniProt, so the table is the pin. UniProt data are CC BY 4.0. **The gene symbol is the HGNC symbol recorded in
the UniProt 2026_03 entry's HGNC cross-reference.** A
read-through locus is dropped when the plain gene is linked too (AARSD1, BCL2L2, WFDC6), the locus type read from the
HGNC complete set, monthly archive
`https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2026-09-04.tsv`,
SHA-256 `6f43d6ff43aa9fdfa5fb2f20a20a7cace66e6e02e2a0dcf19d9b726e2e248d20` (fetched by the generator only). An
accession that links several genes (identical proteins from paralogous loci) takes every one of them, except pseudogene
loci, on both platforms, whatever the vendor lists (reason `paralog accession: every HGNC-linked gene`: DEFA1|DEFA1B, CGB3|CGB5|CGB8); an entry
without an HGNC link keeps UniProt's gene name. Pseudogene and non-coding symbols follow the same rule (ADGRF2P,
ANP32CP). The table holds:

- one row per accession of every multi-target assay (the unit is the assay and the accession): equal counts
  pair by position only when UniProt confirms each pair; otherwise each vendor gene goes to the accessions
  whose UniProt entry names it (gene symbol before synonym; on SomaScan the menu's HGNC and Entrez IDs break
  a tie); a target no vendor gene names takes its gene symbol, and one without any is listed as unresolved;
- every single-target change: a rename (the vendor gene is a UniProt name, synonym or former name, from UniSave,
  and the gene symbol differs), a blank filled, a vendor code replaced (`YP010`, `F231L`, `LOC...`), and the
  analyte assays (reason part `analyte label`), whose vendor name stays the assay label;
- the rows kept for review with the vendor gene: a different gene, no UniProt gene name, an accession not in the
  release, curated constituent-symbol rows, and assays that would share one gene, which are left out of
  gene-level steps (`Excluded_From_Gene_Level`). A vendor gene equal to the entry's UniProt mnemonic is a
  vendor code; codes whose entry has no gene (YB057, YM017) stay as given, and a SomaScan aptamer of
  the same accession, blank in the menu, takes the Olink code (34908-20 YU005, 35005-16 YO027);
- for an assay absent from the vendor list, the accession is read from the supplied assay summary
  (`assay absent from vendor list`).

**HGNC alias table: `vendor_assay_tables/hgnc_aliases.tsv`** (columns `name`, `kind`, `hgnc_id`, `symbol`,
`hgnc_release`), written by stage `vendor_assay_tables` from the pinned HGNC complete set and our gene symbols
(`analysis/external/hgnc_aliases.py`); the table and the release are gitignored. The
stage fetches the release into `hgnc/` from its dated public URL (above; `--hgnc` for a saved copy). Kinds: `symbol`
(each of our gene symbols and its HGNC ID, taken from the release alone: the approved symbol's ID, else the ID of a
previous symbol of exactly one ID not already one of ours;
`assay_gene_updates.py` checks these equal the IDs UniProt's cross-references record);
`approved` (HGNC's current symbol where it differs from ours); `previous` and `alias` (a previous or alias HGNC symbol of exactly one HGNC ID, one of
ours); `ambiguous` (a previous or alias symbol of several IDs, left as it is; `hgnc_id` lists them). A name resolves as
our symbol, then an approved symbol (so another gene's approved symbol is never translated), then a previous symbol, then
an alias. Matching ignores case (Enrichr upper-cases every symbol: C1ORF56 for our C1orf56): names are keyed upper-cased,
the rules apply after upper-casing (names of two IDs that differ only by case are `ambiguous`), rows other than `symbol`
hold the upper-cased name, and a match takes our exact symbol; two of our symbols that differ only by case would be
flagged and not translated (`case_collisions`). The build takes `HGNC_ID` from it, and the pathway step translates every gene-set library member through it
when the library loads, except CellMarker's mouse sets (upper-cased mouse symbols, below). HGNC data are CC0.

## HCOP human-mouse orthologs (CellMarker mouse sets)

`analysis/external/hcop.py` (stage `hcop`) fetches HGNC's HCOP predictions
`https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz?generation=1790858706025272`
(the release of 2026-10-01; HCOP replaces the file monthly, so the URL names the object generation), SHA-256
`0cfb78e4eb273751e0557e8230445a60e844bfaf958e32740d4a266fb41cdb35`, into `hcop/` (`--path` for a saved copy); it is
gitignored. CellMarker 2024's mouse sets hold upper-cased mouse symbols; each member keeps its
place when it is, ignoring case, the symbol of one of its HCOP human orthologs, takes the ortholog when it has exactly
one of another symbol, and is dropped otherwise; a kept or mapped member takes our symbol for the ortholog's HGNC ID.
The sets' minimum size is 5 measured genes. HCOP data are CC0.

**Outputs** (`vendor_assay_tables/`, gitignored): `assay_annotation.tsv`, one row per platform, assay and vendor
accession, with the vendor's UniProt and gene beside the annotated ones and the update reason;
`olink_explore_ht_annotation.tsv` and `somascan_11k_annotation.tsv`, one row per assay with the accessions and
genes pipe-joined in accession order (an accession with several genes, such as P0DN86 = CGB3|CGB5|CGB8, keeps
them all); the long and wide tables carry `Display_Label`, the assay's figure label (`analysis/protein_labels.py`:
one gene; `A/B`; `FIRST +N`, genes in alphabetical order, HGNC duplicate-copy symbols left out; an analyte assay's
vendor name, and the Olink assay name for the assays marked `Excluded_From_Gene_Level`), and the long table
`Is_Analyte_Label`, `HGNC_ID` (each gene's HGNC ID, `|`-joined in the order of
`Gene_Symbol`, an empty element for a gene without one; also on the wide tables) and `Excluded_From_Gene_Level` (true
for assays that would share one gene); and `assay_annotation.inputs.json`, the digests of the inputs (vendor files, update
layer, alias table), against which every reader checks that the annotation is current.

## Ensembl gene to UniProt cross-references

`analysis/external/ensembl_uniprot.py` downloads `Homo_sapiens.GRCh38.116.uniprot.tsv.gz` from Ensembl release
116 (`https://ftp.ensembl.org/pub/release-116/tsv/homo_sapiens/`), SHA-256
`f1e26db23b0771f7a5778067af20683de63d863b1c0834aa564d5d036ec85864`, into `ensembl/`. The Fig. 3b reversal
uses its Swiss-Prot and TrEMBL rows to give each Olink accession that carries one gene an Ensembl gene ID, so
that HPA and the assay annotation join through Ensembl gene IDs.

## cis-pQTL catalogs (QC identity check)

The QC scores each tube's genotype-predicted protein levels against thirteen published cis-pQTL
catalogs, five Olink and eight SomaScan (`reference_catalog` in `analysis/qc/config_{olink,soma}.toml`).
`analysis/external/qc_reference.py` downloads each publication's table into `pqtl/raw/`, checks its
SHA-256 (`RAW_FILES` in `analysis/external/pqtl_catalogs.py`), parses it
(`analysis/external/pqtl_parsers.py`) and writes `pqtl/parsed/<catalog>.parquet`. Every download is a
plain HTTP GET, except Fenland's workbook, a manual download (below).

AGES, INTERVAL and the CKB Atlas SomaScan catalog are parsed from their published workbooks like
the others: the AGES and INTERVAL hg19 positions are lifted to hg38 (see Lifting hg19 to hg38, below),
and AGES takes UniProt by SeqId from the deCODE workbook. `ckb_soma` is the CKB Atlas; the QC configs and
Supplementary Tables S1 and S2 read the same 13 catalogs (five Olink, eight SomaScan). The 2025 CKB
SomaScan catalog is built as well and supplies UniProt to MESA SomaScan; no QC config or table
selects it.

Each parsed frame must match its pinned content digest (`frame_sha256`, `CATALOG_FRAME_SHA256`), and a
mismatch raises before anything is written. The parquet bytes depend on the Polars writer
version, so the pin is on content, not on file bytes.

| Catalog (config name) | Source | Downloaded file (SHA-256) |
|---|---|---|
| UKB-PPP (`ukbppp`) | Sun BB, et al. *Nature* 622, 329–338 (2023), [doi:10.1038/s41586-023-06592-6](https://doi.org/10.1038/s41586-023-06592-6). Supplementary Tables 9 (European discovery) and 11 (other ancestries), cis rows | `41586_2023_6592_MOESM3_ESM.xlsx` (`e174964b…8cc7`) |
| JHS (`jhs`) | Tahir UA, et al. *J Clin Invest* 134, e181802 (2024), [doi:10.1172/JCI181802](https://doi.org/10.1172/JCI181802). ST3 (Jackson Heart Study discovery), cis rows; ST2 for UniProt | `JCI181802.sdt1-12.xlsx` (`00685793…d442`) |
| Genes & Health Olink (`gh`) and SomaScan (`gh_soma`) | Pietzner M, et al. *Nat Genet* 58, 1866–1876 (2026), [doi:10.1038/s41588-026-02697-6](https://doi.org/10.1038/s41588-026-02697-6). Supplementary Table 14, cis rows of each platform; ST2 and ST3 for the IDs (CC BY 4.0) | `41588_2026_2697_MOESM4_ESM.xlsx` (`8bd52a3c…99d2`) |
| MESA Olink (`mesa`) and SomaScan (`mesa_soma`) | Nicholas JC, et al. *Nat Commun* 17, 1054 (2026), [doi:10.1038/s41467-025-67814-1](https://doi.org/10.1038/s41467-025-67814-1). Zenodo record 17642645, [doi:10.5281/zenodo.17642645](https://doi.org/10.5281/zenodo.17642645) (CC-BY-4.0); lead cis variant per protein at p < 5e-8 | `olink_cis_trans_pQTL_combined.tsv.gz` (585 MB, `771b365d…aa3a`), `soma_cis_trans_pQTL_combined.tsv.gz` (729 MB, `749391ee…f424`) |
| China Kadoorie Biobank Olink (`ckb`) and SomaScan 2025 | Wang B, et al. *Nat Commun* 16, 1869 (2025), [doi:10.1038/s41467-025-56935-2](https://doi.org/10.1038/s41467-025-56935-2). Supplementary Data 4, sheets `Olink` and `SomaScan`, cis rows (CC BY 4.0); the 2025 SomaScan v4.1 effects are non-ANML; the SomaScan sheet is parsed into `ckb_soma_2025.parquet` | `41467_2025_56935_MOESM7_ESM.xlsx` (`c6a6f4cf…6764`) |
| CKB Atlas SomaScan 2026 (`ckb_soma`) | CKB Atlas, [doi:10.5281/zenodo.21503247](https://doi.org/10.5281/zenodo.21503247), [Supplementary Tables.xlsx](https://zenodo.org/api/records/21503247/files/Supplementary%20Tables.xlsx/content), ST2 aptamer annotation and ST3 cis sentinels (CC BY 4.0). SomaScan v4.1, GRCh38 EA/REF; ANML-derived rank-inverse-normal residual SD betas and SEs; `source` is `ckb_atlas` | `ckb_atlas_supplementary_tables.xlsx` (`22aeba9e…0d3a`) |
| AGES (`ages`) | Gudjonsson A, et al. *Nat Commun* (2022), [doi:10.1038/s41467-021-27850-z](https://doi.org/10.1038/s41467-021-27850-z), [Supplementary Data 3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-27850-z/MediaObjects/41467_2021_27850_MOESM5_ESM.xlsx), sheet `Data 3` (CC BY 4.0). Serum SomaScan 4.9k, marginal cis beta/SE/P and EAF, GRCh37 lifted to GRCh38; deCODE SeqId supplies UniProt, minus-strand EA complemented (EAF kept as published) | `41467_2021_27850_MOESM5_ESM.xlsx` (`f2f4eafe…2dca`) |
| INTERVAL (`interval`) | Sun BB, et al. *Nature* (2018), [doi:10.1038/s41586-018-0175-2](https://doi.org/10.1038/s41586-018-0175-2), published supplementary sheet `ST4 - pQTL summary`. SomaScan legacy panel, cis-only meta-analysis beta/SE/P (not subcohorts), GRCh37 lifted to GRCh38, EA and OA both reverse-complemented on minus strand | `41586_2018_175_MOESM4_ESM.xlsx` (`d7bfada3…b9d5`) |
| deCODE (`decode`) | Ferkingstad E, et al. *Nat Genet* 53, 1712–1721 (2021), [doi:10.1038/s41588-021-00978-w](https://doi.org/10.1038/s41588-021-00978-w). ST02, cis rows (conditional betas) | `41588_2021_978_MOESM4_ESM.xlsx` (`e3f907a1…0ef6`) |
| ARIC (`aric`) | Zhang J, et al. *Nat Genet* 54, 593–602 (2022), [doi:10.1038/s41588-022-01051-w](https://doi.org/10.1038/s41588-022-01051-w). ST3.1 (EA) and ST3.2 (AA) | `41588_2022_1051_MOESM4_ESM.xlsx` (`473e0cb7…b02c`) |
| Fenland (`fenland_hg38`) | Pietzner M, et al. *Science* 374, eabj1541 (2021), [doi:10.1126/science.abj1541](https://doi.org/10.1126/science.abj1541). ST2, lead cis signals (hg19), lifted to hg38 | `science.abj1541_tables_s1_to_s9.xlsx` (`4d242a50…5c51`), downloaded by hand, see below |
| (MESA Olink UniProt only) | Olink, Explore 3072 Fixed LOD file of December 2024 (public download from olink.com; `OlinkID` and `UniProt` columns, one row per assay per reagent lot) | `olink_explore_3072_fixed_lod_2024-12-19.csv` (`0c9543ac…840c`) |

Notes on the sources:

- **Fenland.** The source is the article's supplementary workbook,
  `science.abj1541_tables_s1_to_s9.xlsx` (Supplementary Tables S1 to S9; SHA-256
  `4d242a5057510de09ff645469fd83414048a7e3fc7d52a6c94818c2dac2c5c51`). science.org refuses
  scripted requests, so download it in a browser from the article's Supplementary Materials
  (https://doi.org/10.1126/science.abj1541), create `data/vendor/` (it does not ship) and either place it at
  `data/vendor/science.abj1541_tables_s1_to_s9.xlsx` or pass `--fenland-xlsx <path>` to
  `analysis/external/qc_reference.py` (the stage passes `paths.yaml` `fenland_science_xlsx`); the stage checks the digest before anything is downloaded.
  The positions are lifted to hg38 (below), rows that do not lift are dropped, and each lifted row
  is compared with the trial PVAR at its hg38 position (`REF`, `ALT`, `allele_check`); the allele check
  reads the PVAR family that `paths.yaml` `surmount5_fenland_pvar_pattern` names (`--pvar-pattern`). The
  QC scores with the catalog columns, not the PVAR columns. The catalog carries `eaf`, the published
  frequency of the effect allele `a1`. The liftover moves `chrom` and `pos` only and never swaps or
  complements `a1`/`a2`, so `eaf` stays the frequency of `a1`. The build checks the frame without `eaf`
  against `CATALOG_FRAME_SHA256` and the full frame against `FULL_FRAME_SHA256`; either mismatch raises
  before a file is written. The QC reads `eaf` only when run with `pqtl_palindromic = "rescue"`.
- **Lifting hg19 to hg38.** Fenland, INTERVAL and AGES report GRCh37 positions.
  `analysis/external/liftover.py` lifts each one with Ensembl's GRCh37-to-GRCh38 assembly mapping
  ([GRCh37_to_GRCh38.chain.gz](https://ftp.ensembl.org/pub/assembly_mapping/homo_sapiens/GRCh37_to_GRCh38.chain.gz),
  SHA-256 `351de3cd4a01d9fcffd38881981767b697090d2eba876740891b96d5c546b100`). A variant the chain does not
  cover is placed by its rsID in Ensembl's release 116 GRCh38 variation files
  (`https://ftp.ensembl.org/pub/release-116/variation/vcf/homo_sapiens/homo_sapiens-chr<N>.vcf.gz`), read over
  HTTPS through their indexes (`.csi`, SHA-256 pinned in `VARIATION_INDEX_SHA256`); the search starts at the
  position the nearest aligned block implies and widens up to 5 Mb, and the strand is that block's. A
  variant placed by neither is dropped (Fenland, INTERVAL) or kept with a null `pos` that the QC drops
  (AGES). The chain and the indexes are downloaded into `data/external/ensembl_mapping/` (not tracked);
  the variation files themselves are read in place, and only rsID positions are read from them. Ensembl's terms:
  "Ensembl imposes no restrictions on access to, or use of, the data provided".
- **MESA Olink** takes `uniprot` by OlinkID from UKB-PPP, Olink's Explore 3072 Fixed LOD file and
  JHS, in that order; the LOD file is used for that alone, and every reagent lot of an OlinkID must
  name the same UniProt (the build raises otherwise).
- **CKB Olink** names each assay by gene symbol only and takes UniProt and OlinkID from
  UKB-PPP, MESA Olink and Genes & Health Olink.
- **CKB SomaScan 2025** takes gene and UniProt by SeqId from deCODE and ARIC; **MESA SomaScan**
  takes UniProt from deCODE, ARIC and CKB SomaScan 2025. The CKB Atlas catalog (`ckb_soma`) carries
  the ST2 annotation of its own workbook.
