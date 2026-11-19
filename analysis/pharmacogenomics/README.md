# GIPR E354Q pharmacogenomic analysis

This directory tests whether GIPR rs1800437 (E354Q, p.Glu354Gln) modifies the TZP - SEMA difference in PPY
at Week 24 and Week 72 (Supplementary Fig. 11,
`figures/supp_fig_e354q.py`). The model estimates genotype-specific
tirzepatide-semaglutide contrasts.

## Included modules

- `run.py` — consolidated runner (genotype extraction, frame assembly, MMRM)
- `build_pruned_genome.py` — PLINK2 linkage disequilibrium (LD) pruning for PCA input
- `compute_wgs_pcs.R` — SNPRelate PCA (PC1-10)

## Inputs

| Input | Source | Description |
|---|---|---|
| pVCF (chr19) | `paths.yaml:surmount5_chr19_pvcf` | rs1800437 genotype at chr19:45678134 G>C (GRCh38) |
| Pheno-geno linking | `paths.yaml:surmount5_pheno_geno_linking` | DNA sample ID to subject ID mapping, corrected by the QC's `dna_relink` entries |
| QC corrections | `paths.yaml:surmount5_{olink,soma}_qc_corrections` | The `dna_relink` entries (must agree between the two TOMLs) |
| QC genotype link | `paths.yaml:qc_root`, through `analysis.qc_contract.dna_link_path` | `USUBJID` and `link_status` of each platform's genotype link |
| ADSL | `paths.yaml:pheno_root` `adsl.parquet` | Arm, age, sex |
| Olink long | `paths.yaml:qc_root`, through `analysis.qc_contract.long_table_path` | `NPX`, with the vendor's `AssayQC == "FAIL"` points dropped (the MMRM layer) |
| SomaScan long | `paths.yaml:qc_root`, same reader | non-ANML `RFU`, log2 in the model |
| Analysis cohort | `paths.yaml:qc_root`, through `analysis.qc_contract.cohort_samples` | Sample to USUBJID + analysis visit, `Status != "FAIL"` |
| Genetic PCs | `paths.yaml:qa_root` `3000_genomic_data_summary/3100_pca/3101_*_data*_qced_maf0.01_pruned_genome_pc.csv` | PC1-10 from SNPRelate on the LD-pruned autosomal genome; exactly one file may match (the run raises with the candidate list otherwise) |

The run also raises when a subject, sample or PC row is duplicated where a join needs it
to be unique.

**Genotype links.** Before the joins, each `dna_relink` correction of the QC
(`analysis/qc/dna_relink.py`) files its genome under its owner in the linking table; the
participant it was filed under is left with no genome. The run then checks the corrected
table against each platform's QC genotype link: it stops if the QC link names a participant
the corrected table gives no genome (the QC ran without the same corrections) or holds
any link whose `link_status` is not `ok` (`suspect`, `dropped`, `hold` or null).

Probes used:

| Protein | Olink | SomaScan |
|---|---|---|
| PPY | `OID44867` | `4588-1` |

## Model specification

A separate MMRM is fit for each (gene × platform) cell.

```
value ~ 0 + visit_f:arm_f:geno_f
      + base + age + sex + PC1 + PC2 + PC3
      + us(visit_f | USUBJID)
```

- `visit_f` ∈ {W24, W72}, with `us(visit_f | USUBJID)` capturing
  within-subject correlation
- `geno_f` is a 3-level genotype factor (GG / GC / CC); the model carries a
  full cell mean per visit × arm × genotype
- Per-(gene × platform) TZP-SEMA contrasts via `emmeans` pairwise
  comparisons conditioned on visit × genotype; Kenward-Roger degrees of
  freedom (linear Kenward-Roger coefficient covariance, `MMRM_CONTROL_ARGS`) and t-distribution confidence intervals
- No multiple-testing correction: P values are nominal and two-sided (`adjust="none"`); bold at P < 0.05 in the figure

## Running

```bash
# Build the genetic principal components from the configured inputs
pixi run python analysis/pipelines/run_wgs_pcs.py

# Run the pharmacogenomics analysis (it takes no options; `--help` prints usage and exits)
pixi run python analysis/pharmacogenomics/run.py

# Render Supplementary Fig. 11 (PPY genotype row)
pixi run python figures/supp_fig_e354q.py
```

## Outputs

- `analysis/outputs/genotype_E354Q_frame.parquet` — analysis-ready long frame
- `analysis/outputs/genotype_E354Q_contrasts.parquet` — per-(gene × visit × genotype × platform) TZP-SEMA contrasts

`figures/supp_fig_e354q.py` reads the frame and contrasts parquets (through
`figures/_internal/_ppy_figure.draw_genotype_row`).
