# GIPR E354Q pharmacogenomic analysis

This directory tests whether GIPR rs1800437 (E354Q, p.Glu354Gln) modifies the tirzepatide-induced change in PPY and REG4
across Week 24 and Week 72 in the SURMOUNT-5 trial. Semaglutide serves as the
GLP-1-only comparator arm (no GIPR agonism), where no genotype-by-treatment
interaction is expected.

## Included modules

- `run.py` — consolidated runner (genotype extraction, frame assembly, MMRM)
- `build_pruned_genome.py` — PLINK2 linkage disequilibrium (LD) pruning for PCA input
- `compute_wgs_pcs.R` — SNPRelate PCA (PC1-10)

## Inputs

| Input | Source | Description |
|---|---|---|
| pVCF (chr19) | `paths.yaml:geno_root` | rs1800437 genotype at chr19:45678134 G>C (GRCh38) |
| Pheno-geno linking | `paths.yaml:geno_root` | DNA sample ID to subject ID mapping |
| ADSL | `paths.yaml:pheno_root` | Arm, age, sex, baseline covariates |
| Olink long | `paths.yaml:qc_root` | PCNormalizedNPX values |
| Olink sample QC | `paths.yaml:qc_root` | SampleID to USUBJID + VISITNUM + Status |
| SomaScan long | `paths.yaml:qc_root` | RFU values (log2 in analysis) |
| SomaScan sample QC | `paths.yaml:qc_root` | Same as Olink |
| Genetic PCs | `paths.yaml:qa_root` | PC1-10 from SNPRelate, LD-pruned autosomal genome |

Probes used:

| Protein | Olink | SomaScan |
|---|---|---|
| PPY | `OID44867` | `4588-1` |
| REG4 | `OID44891` | `11102-22` |

## Model specification

A separate MMRM is fit for each (gene × platform) cell. Olink and SomaScan are
not pooled into one model because they measure different physical quantities
(NPX log2 vs log2 RFU) with different error structures.

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
  comparisons conditioned on visit × genotype; Satterthwaite degrees of
  freedom and t-distribution confidence intervals
- Multiple testing correction: Bonferroni × 2 across the 2 proteins

## Recessive interaction test

In addition to per-genotype contrasts, a recessive interaction test is
computed from the same emmeans pairwise object. The contrast weights
[-0.5, -0.5, 1] are applied to the [GG, GC, CC] TZP-SEMA differences,
testing whether the treatment effect is lower in C/C homozygotes than in
G-allele carriers (G/G and G/C combined):

```
estimate = (TZP-SEMA in CC) - 0.5*(TZP-SEMA in GG) - 0.5*(TZP-SEMA in GC)
```

## Running

```bash
# Build the genetic PCs (requires mounted geno_root and qa_root)
uv run python analysis/pipelines/run_wgs_pcs.py

# Run the pharmacogenomics analysis
uv run python analysis/pharmacogenomics/run.py

# Render the downstream manuscript figures
uv run python figures/fig5_ppy.py
uv run python figures/supp_fig16_reg4.py
```

## Outputs

- `analysis/outputs/genotype_E354Q_frame.parquet` — analysis-ready long frame
- `analysis/outputs/genotype_E354Q_contrasts.parquet` — per-(gene × visit × genotype × platform) TZP-SEMA contrasts (24 rows)
- `analysis/outputs/genotype_E354Q_recessive.parquet` — per-(gene × visit × platform) recessive interaction test (16 rows)
