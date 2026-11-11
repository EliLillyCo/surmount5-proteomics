# Pathway wrapper

`analysis/pathway/` regenerates the manuscript `pathwayRes` directories from the MMRM
`rawOutput` files of a platform:

- marker-to-gene loading for Olink and SomaScan assay manifests
- gene-level ranking `z = estimate / SE` on the log2 scale (the t-like ratio of the emmeans
  estimate), the TZP - SEMA `diff` from `rawOutput/*_chg_diff.csv`
  for the between-treatment contrasts. Multi-gene assays are exploded, and a gene's z is the
  equal-weight mean over its assays (NPPB from BNP and NT-proBNP; IL12B from p40 and the IL-12 p70 assay).
  The ranking uses the analysis cohort, with no extra QC filter
- within each gene set, genes whose z comes from the same assays (identical source-assay sets: the paralog genes
  of one accession such as DEFB4A|DEFB4B, the genes of one multi-target or complex assay) count once, in cameraPR's
  index and size bounds and in ORA's overlap, set size, query and background (one measurement is one draw) (`preparation.source_groups`,
  `methods.collapse_same_source`); the set's first such member stands for them in `Genes`/`Lead_genes`
- The assays in `analysis.platform_inputs.OLINK_EXCLUDED_ASSAYS` (by OlinkID) are
  excluded from the ranked statistic and from the ORA query and background. The number of
  excluded assays is logged per contrast.
- `limma::cameraPR` via rpy2 with `inter.gene.cor = 0.01` passed explicitly and recorded
  (`inter_gene_cor` column), sets of 5-500 measured genes, set members deduplicated
- the FDR family is one gene-set library in one contrast for one method: cameraPR's `FDR q-val` runs over the
  library's tested sets and ORA's `Adjusted P-value` over the library's terms
- ORA (Enrichr statistics) on a gene-level query, on both platforms: a gene's P is the Simes combination of its
  assays' P values, Benjamini-Hochberg runs over genes (genes measured by identical assays are one test), and the
  query is the genes at FDR < 0.05 (`preparation.simes_gene_fdr`); no size filter on ORA sets
- library members are translated to the annotation's symbols through the HGNC alias table built with the annotation
  (`data/external/vendor_assay_tables/hgnc_aliases.tsv`, `--aliases`): approved, then previous, then unambiguous
  alias symbols, matched case-insensitively (Enrichr upper-cases symbols: C1ORF56 for C1orf56)
- CellMarker's mouse sets go through HGNC HCOP human-mouse orthologs (stage `hcop`, `--hcop`): a member is kept when
  it is the symbol of one of its human orthologs, replaced by its ortholog when it has exactly one, and dropped
  otherwise
- annotation rows marked `Excluded_From_Gene_Level` are not ranked
- seven gene-set libraries for both methods: Reactome_Pathways_2024, KEGG_2026,
  WikiPathways_2024_Human, GO_Biological_Process_2025, GO_Molecular_Function_2025,
  GO_Cellular_Component_2025, CellMarker_2024

`batch` finds every `<study>_<platform>/covar_*/rawOutput` under `--base-dir` and runs the
same-visit TZP − SEMA contrast at each post-baseline week (two per platform); any
contrast failure stops the run, and `--dry-run` lists the contrasts and output directories
without running them. Output per contrast:
`<covar>/pathwayRes/acTrt/<label>/{camera_combined.csv,ora_combined.csv}` with `<label>` such
as `TZP15mgorMTDVSSEMA2.4mgorMTD@24`; readers build these paths through
`analysis.mmrm_outputs.pathway_csv`. A method that runs but returns no results replaces its
combined CSV with a header-only file; a method disabled with `--skip-ora` or `--skip-camera`
leaves its existing CSV untouched.

Gene-set libraries are read as JSON from `data/external/gene_sets/` (`--cache-dir`), which the
`gene_sets` stage fills (`pixi run python analysis/external/gene_sets.py`); `--download-missing`
downloads a library that is absent there.

Member corrections: `analysis/reference/gene_set_corrections.tsv` replaces `TF` with `F3` (tissue factor) in two
Reactome clotting terms (Extrinsic Pathway of Fibrin Clot Formation; Formation of Fibrin Clot (Clotting Cascade))
when `GeneSetCache` loads the library (`cache.apply_corrections`, before symbol translation). The pinned JSON
files and digests are unchanged, and a correction whose library term is missing, whose `from_symbol` is not in the
term, or whose `to_symbol` is already there raises. No other set is changed.

### Pinned inputs

Every gene-set library is SHA-256 verified before use, and a mismatch raises `ValueError` naming
the file, the expected digest and the actual digest (no fallback). The digests are in `analysis/pathway/constants.py` (`GENE_SET_SHA256`, the seven Enrichr libraries). The
`gene_sets` stage skips a library already present with its pinned digest and raises on a present
file with another digest. A fresh download passes only if Enrichr still serves byte-identical
content; Enrichr revises libraries in place, so expect a mismatch error if it does not.

### Assay -> gene mapping

Both platforms map assays to genes through the assay annotation, the output of stage
`vendor_assay_tables` (`data/external/vendor_assay_tables/assay_annotation.tsv`, `--annotation`):
the vendor assay lists (Olink's public Explore HT list, the SomaLogic annotated menu; both
SHA-256 pinned) with the UniProt gene-update layer
(`analysis/reference/assay_gene_updates.tsv`) on top. A multi-target assay maps to the genes of
all its accessions, and an accession can carry several genes (identical proteins from paralogous
loci); `loaders.load_assay_genes` reads it and refuses an annotation built from another update
layer. `analysis/pipelines/run_disease_reversal.py` reads the same Olink mapping.

Pathway results are written to the configured output directory (`pathwayRes/acTrt`).

The `pathway` stage needs the MMRM stage's output (`mmrm`) and the three inputs below, each run
through the manifest runner:

```bash
pixi run stage run vendor_assay_tables   # fills --menu from paths.yaml
pixi run stage run hcop
pixi run stage run gene_sets
pixi run stage run pathway               # fills <paths.yaml:mmrm_root>
```
