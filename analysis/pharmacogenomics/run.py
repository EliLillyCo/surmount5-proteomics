"""GIPR rs1800437 (E354Q) × PPY/REG4 pharmacogenomic pipeline.

Extracts rs1800437 genotypes, assembles a QC-filtered PPY/REG4 long frame
across Olink and SomaScan, and fits per-(gene × platform) MMRMs to extract
per-genotype TZP-SEMA contrasts. Produces the two parquets used by the
pancreas trajectories figure:

  ``analysis/outputs/genotype_E354Q_frame.parquet``     — analysis-ready
      long frame (USUBJID × platform × probe × visit × value/base/dosage
      /arm/age/sex/PC1..10) for PPY + REG4, TZP/SEMA arms only,
      genotyped subjects only, QC-passing samples only.

  ``analysis/outputs/genotype_E354Q_contrasts.parquet`` — per-(gene × visit
      × genotype × platform) TZP-SEMA contrasts (est, SE, df, t, p, ci_lo,
      ci_hi, n_tzp, n_sema). A separate MMRM is fit for each (gene ×
      platform) cell, each with ``us(visit_f | USUBJID)`` within-subject
      covariance; Olink and SomaScan are NOT pooled into a joint model.

  ``analysis/outputs/genotype_E354Q_recessive.parquet`` — per-(gene × visit
      × platform) recessive interaction test: (TZP-SEMA in CC) vs mean
      (TZP-SEMA in GG, TZP-SEMA in GC), emmeans custom contrast weights
      [-0.5, -0.5, 1] for [GG, GC, CC] applied to the pairwise arm
      differences from the same saturated cell-means MMRM. Columns: gene,
      visit, platform, est, se, df, t, p, ci_lo, ci_hi.

Pre-specification & methods are documented in
``analysis/pharmacogenomics/README.md``. Each per-(gene × platform)
MMRM uses the formula:

    value ~ 0 + visit_f:arm_f:geno_f
          + base + age + sex + PC1 + PC2 + PC3
          + us(visit_f | USUBJID)

The per-genotype TZP-SEMA contrast at each visit is the (TZP − SEMA)
coefficient difference within that platform's own fit.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pysam
import rpy2.robjects as ro
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.packages import importr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import GENO_ROOT, PHENO_DIR, QA_ROOT, QC_ROOT, RAW_PROTEOMICS_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = REPO_ROOT / "analysis" / "outputs"

# Primary inputs (project-wide mount; configured in paths.yaml).
if GENO_ROOT is None or QA_ROOT is None or RAW_PROTEOMICS_ROOT is None:
    raise RuntimeError(
        "paths.yaml must define geno_root, qa_root, and raw_proteomics_root "
        "to run analysis/pharmacogenomics/run.py"
    )

VCF_PATH = GENO_ROOT / "pvcf" / "surmount5.chr19_1-58617616.vcf.gz"
LINKING_CSV = GENO_ROOT / "pheno_geno_linking" / "dx_pheno_geno_linking_surmount5.csv"
ADSL_PARQUET = PHENO_DIR / "adsl.parquet"
OLINK_LONG = QC_ROOT / "surmount5_olink" / "surmount5_olink.npx_long.parquet"
OLINK_SAMPLE_QC = QC_ROOT / "surmount5_olink" / "surmount5_olink.sample_qc_summary.tsv"
SOMA_LONG = QC_ROOT / "surmount5_soma" / "surmount5_soma.rfu_long.parquet"
SOMA_SAMPLE_QC = QC_ROOT / "surmount5_soma" / "surmount5_soma.sample_qc_summary.tsv"
# Raw Olink NPX export carries the per-assay PlateID — the project-wide
# QC long parquet drops it. Stream-only access (large file).
RAW_OLINK_NPX = (
    RAW_PROTEOMICS_ROOT / "olink" / "surmount5_olink_npx_08Dec2025.parquet"
)

# ---------------------------------------------------------------------------
# rs1800437 + PPY/REG4 constants (README.md §3)
# ---------------------------------------------------------------------------
SNP_CHROM = "chr19"
SNP_POS = 45678134
SNP_REF = "G"
SNP_ALT = "C"

PROBES = {
    "olink": {"OID44867": "PPY", "OID44891": "REG4"},
    "soma": {"4588-1": "PPY", "11102-22": "REG4"},
}

PC_COLS = [f"PC{i}" for i in range(1, 11)]


def _resolve_pca_csv() -> str:
    pca_dir = QA_ROOT / "3000_genomic_data_summary/3100_pca"
    matches = sorted(
        list(pca_dir.glob("3101_surmount5_data*_qced_maf0.01_pruned_genome_pc.csv"))
        + list(pca_dir.glob("3101_surmount5_data*_qced_maf0.01_pruned_genome_pc.csv"))
    )
    if not matches:
        raise FileNotFoundError(
            "Missing genetic PCs. Run: uv run python analysis/pipelines/run_wgs_pcs.py"
        )
    if len(matches) == 1:
        return str(matches[0])
    return str(max(matches, key=lambda path: path.stat().st_mtime))


WEEK_MAP = {2: 0, 8: 24, 20: 72}
ARMS = ["TZP 15mg or MTD", "SEMA 2.4mg or MTD"]


# ---------------------------------------------------------------------------
# Step 1: rs1800437 dosage from chr19 pVCF
# ---------------------------------------------------------------------------
def extract_rs1800437_genotypes() -> pl.DataFrame:
    """Return one row per VCF sample with GT + dosage (0/1/2 of ALT)."""
    vcf = pysam.VariantFile(str(VCF_PATH))
    samples = list(vcf.header.samples)
    rec = next(
        (
            r
            for r in vcf.fetch(SNP_CHROM, SNP_POS - 1, SNP_POS)
            if r.pos == SNP_POS and r.ref == SNP_REF and SNP_ALT in r.alts
        ),
        None,
    )
    if rec is None:
        raise RuntimeError(f"rs1800437 not found at {SNP_CHROM}:{SNP_POS}")
    alt_idx = list(rec.alts).index(SNP_ALT) + 1

    rows = []
    for sid in samples:
        gt = rec.samples[sid].get("GT")
        if gt is None or any(a is None for a in gt):
            dosage, gt_str = None, "./."
        else:
            dosage = sum(1 for a in gt if a == alt_idx)
            gt_str = "/".join(str(a) for a in gt)
        rows.append({"dx_dna_sm_id": sid, "GT": gt_str, "dosage": dosage})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: subject-level link table (genotype + ADSL + PCs)
# ---------------------------------------------------------------------------
def build_sample_link(geno: pl.DataFrame) -> pl.DataFrame:
    link = pl.read_csv(LINKING_CSV)
    adsl = pl.read_parquet(
        ADSL_PARQUET,
        columns=["USUBJID", "AGE", "SEX", "TRT01A"],
    )
    pcs = (
        pl.read_csv(_resolve_pca_csv())
        .rename({"IID": "dx_dna_sm_id"})
        .select(["dx_dna_sm_id"] + PC_COLS)
    )

    sample_link = (
        link.join(adsl, left_on="dx_usubjid", right_on="USUBJID", how="inner")
        .join(geno, on="dx_dna_sm_id", how="inner")
        .join(pcs, on="dx_dna_sm_id", how="inner")
        .filter(pl.col("TRT01A").is_in(ARMS))
        .rename(
            {"dx_usubjid": "USUBJID", "TRT01A": "arm", "AGE": "age", "SEX": "sex_str"}
        )
    )
    sample_link = sample_link.with_columns(
        pl.when(pl.col("sex_str") == "M").then(1.0).otherwise(0.0).alias("sex"),
        pl.col("age").cast(pl.Float64),
        pl.col("dosage").cast(pl.Float64),
    )
    return sample_link.select(
        ["USUBJID", "arm", "dosage", "age", "sex"] + PC_COLS
    )


def _build_sample_map(qc_path: Path, sample_col: str) -> pl.DataFrame:
    """SampleID → USUBJID + VISITNUM + week, restricted to PASS/WARN."""
    qc = pl.read_csv(qc_path, separator="\t", schema_overrides={"VISITNUM": pl.Float64})
    qc = qc.rename({sample_col: "SampleID"}).with_columns(
        pl.col("VISITNUM")
        .cast(pl.Int64, strict=False)
        .replace_strict(WEEK_MAP, default=None)
        .alias("week")
    )
    return (
        qc.filter(pl.col("Status") != "FAIL")
        .filter(pl.col("week").is_not_null())
        .select(["SampleID", "USUBJID", "week", "Status"])
    )


# ---------------------------------------------------------------------------
# Step 3: assemble QC-filtered PPY+REG4 long frame for both platforms
# ---------------------------------------------------------------------------
def build_focused_frame(sample_link: pl.DataFrame) -> pl.DataFrame:
    """One row per (USUBJID × probe × visit ∈ {W24, W72}) with baseline +
    covariates. Only PPY + REG4 probes; only genotyped TZP/SEMA subjects.
    """
    olink_map = _build_sample_map(OLINK_SAMPLE_QC, "SampleID")
    soma_map = _build_sample_map(SOMA_SAMPLE_QC, "SampleId")

    eval_subj = sample_link.select("USUBJID")
    olink_map = olink_map.join(eval_subj, on="USUBJID", how="inner")
    soma_map = soma_map.join(eval_subj, on="USUBJID", how="inner")

    # Stream the raw NPX parquet for the two target Olink assays only to
    # pull PlateID per (SampleID × OlinkID). The QC long parquet drops
    # PlateID, so this is the only source. Streaming-collect keeps memory
    # bounded — file is large.
    olink_plates = (
        pl.scan_parquet(RAW_OLINK_NPX)
        .filter(pl.col("OlinkID").is_in(list(PROBES["olink"])))
        .select(["SampleID", "OlinkID", "PlateID"])
        .unique()
        .collect(engine="streaming")
    )

    olink = (
        pl.scan_parquet(OLINK_LONG)
        .filter(pl.col("OlinkID").is_in(list(PROBES["olink"])))
        .join(olink_map.lazy(), on="SampleID", how="inner")
        .join(olink_plates.lazy(), on=["SampleID", "OlinkID"], how="left")
        .select(
            [
                "USUBJID",
                pl.col("OlinkID").alias("probe"),
                "week",
                pl.col("PCNormalizedNPX").alias("value"),
                pl.col("PlateID").alias("plate"),
            ]
        )
        .with_columns(
            pl.lit("olink").alias("platform"),
            pl.col("probe").replace_strict(PROBES["olink"]).alias("gene"),
        )
        .collect(engine="streaming")
    )

    soma = (
        pl.scan_parquet(SOMA_LONG)
        .filter(pl.col("SeqId").is_in(list(PROBES["soma"])))
        .join(soma_map.lazy(), left_on="SampleId", right_on="SampleID", how="inner")
        .filter(pl.col("RFU") > 0)
        .with_columns((pl.col("RFU").log() / pl.lit(2.0).log()).alias("value"))
        .select(["USUBJID", pl.col("SeqId").alias("probe"), "week", "value"])
        .with_columns(
            pl.lit("olink_NA").alias("plate"),  # SomaScan has no plate factor; constant placeholder
            pl.lit("soma").alias("platform"),
            pl.col("probe").replace_strict(PROBES["soma"]).alias("gene"),
        )
        .collect(engine="streaming")
    )

    long_df = pl.concat([olink, soma])
    visit_map = {0: "Week0", 24: "Week24", 72: "Week72"}
    long_df = long_df.with_columns(
        pl.col("week").replace_strict(visit_map).alias("visit")
    )

    base = long_df.filter(pl.col("week") == 0).select(
        ["USUBJID", "platform", "probe", "gene", pl.col("value").alias("base")]
    )
    frame = (
        long_df.filter(pl.col("week").is_in([24, 72]))
        .join(base, on=["USUBJID", "platform", "probe", "gene"], how="inner")
        .join(sample_link, on="USUBJID", how="inner")
    )
    return frame.select(
        [
            "USUBJID",
            "platform",
            "probe",
            "gene",
            "visit",
            "week",
            "value",
            "base",
            "dosage",
            "arm",
            "age",
            "sex",
            "plate",
        ]
        + PC_COLS
    )


# ---------------------------------------------------------------------------
# Step 4: MMRM (rpy2) — per-genotype TZP-SEMA contrasts per visit
# ---------------------------------------------------------------------------
_E354Q_R_BINDINGS: dict[str, Any] | None = None

FORMULA_STR = (
    "value ~ 0 + visit_f:arm_f:geno_f"
    " + base + age + sex + PC1 + PC2 + PC3"
    " + us(visit_f | USUBJID)"
)


def _get_r_bindings() -> dict[str, Any]:
    """Lazily import R packages needed for the E354Q MMRM."""
    global _E354Q_R_BINDINGS
    if _E354Q_R_BINDINGS is None:
        r_base = importr("base")
        _E354Q_R_BINDINGS = {
            "mmrm": importr("mmrm"),
            "emmeans": importr("emmeans"),
            "factor": ro.r["factor"],
            "as_character": ro.r["as.character"],
            "transform": ro.r["transform"],
            "as_data_frame": ro.r["as.data.frame"],
            "summary": ro.r["summary"],
            "pairs": ro.r["pairs"],
            "gc": r_base.gc,
        }
    return _E354Q_R_BINDINGS


def _prepare_cell_frame(frame: pl.DataFrame, gene: str, platform: str) -> pl.DataFrame:
    """Filter to (gene, platform) and add factor-ready string columns."""
    cell = frame.filter(
        (pl.col("gene") == gene) & (pl.col("platform") == platform)
    )
    cell = cell.with_columns(
        pl.when(pl.col("arm") == "TZP 15mg or MTD")
        .then(pl.lit("TZP"))
        .otherwise(pl.lit("SEMA"))
        .alias("arm_f"),
        pl.when(pl.col("dosage") == 0.0)
        .then(pl.lit("GG"))
        .when(pl.col("dosage") == 1.0)
        .then(pl.lit("GC"))
        .when(pl.col("dosage") == 2.0)
        .then(pl.lit("CC"))
        .otherwise(None)
        .alias("geno_f"),
        pl.col("visit").str.replace("Week", "W").alias("visit_f"),
    )
    needed = [
        "value", "base", "age", "sex", "PC1", "PC2", "PC3",
        "geno_f", "visit_f", "arm_f", "USUBJID",
    ]
    return cell.drop_nulls(subset=needed)


def _to_r_data(cell: pl.DataFrame) -> ro.vectors.DataFrame:
    """Convert Polars cell frame to R data.frame with proper factors."""
    r_factor = _get_r_bindings()["factor"]

    cell_pd = cell.select(
        ["USUBJID", "visit_f", "arm_f", "geno_f",
         "value", "base", "age", "sex", "PC1", "PC2", "PC3"]
    ).to_pandas()

    with pandas2ri.converter.context():
        r_data = pandas2ri.py2rpy(cell_pd)

    colnames = list(r_data.names)
    factor_specs = {
        "arm_f": ["SEMA", "TZP"],
        "geno_f": ["GG", "GC", "CC"],
        "visit_f": ["W24", "W72"],
        "USUBJID": None,
    }
    for col, levels in factor_specs.items():
        idx = colnames.index(col)
        if levels is not None:
            r_data[idx] = r_factor(r_data.rx2(col), levels=ro.StrVector(levels))
        else:
            r_data[idx] = r_factor(r_data.rx2(col))

    return r_data


def _compute_cell_counts(cell: pl.DataFrame) -> pl.DataFrame:
    """Unique subject counts per (visit_f, geno_f, arm_f) → n_tzp + n_sema."""
    counts = cell.group_by(["visit_f", "geno_f", "arm_f"]).agg(
        pl.col("USUBJID").n_unique().alias("n")
    )
    n_tzp = (
        counts.filter(pl.col("arm_f") == "TZP")
        .rename({"n": "n_tzp"})
        .select(["visit_f", "geno_f", "n_tzp"])
    )
    n_sema = (
        counts.filter(pl.col("arm_f") == "SEMA")
        .rename({"n": "n_sema"})
        .select(["visit_f", "geno_f", "n_sema"])
    )
    return n_tzp.join(n_sema, on=["visit_f", "geno_f"], how="outer_coalesce").rename(
        {"visit_f": "visit", "geno_f": "geno"}
    )


def _extract_recessive_contrast(
    contrast_obj: Any,
    emmeans_r: Any,
    gene: str,
    platform: str,
) -> pl.DataFrame:
    """Apply the recessive interaction contrast to per-genotype TZP-SEMA diffs.

    Tests whether the TZP-SEMA treatment effect is lower in C/C homozygotes
    than in G-allele carriers (G/G and G/C combined). Applied to the emmeans
    pairwise arm-contrast object already computed in run_mmrm_contrasts, so
    no additional model fit is required.

    Contrast weights [-0.5, -0.5, 1] on [GG, GC, CC] give:
        estimate = (TZP-SEMA in CC) - 0.5*(TZP-SEMA in GG) - 0.5*(TZP-SEMA in GC)

    This framing is motivated by GPCR haplosufficiency: loss-of-function GPCR
    mutations are generally recessive because one functional allele produces
    enough receptor for normal signalling. Consistent with this, G/C
    heterozygotes in the data retain the full treatment effect (clustering
    with G/G), while C/C homozygotes do not.
    """
    output = ro.r(
        """
        function(con) {
            rec <- contrast(
                con,
                list(CC_vs_Gcarriers = c(-0.5, -0.5, 1)),
                by = "visit_f"
            )
            sm <- summary(rec, infer = c(TRUE, TRUE))
            list(
                visit    = as.character(sm[["visit_f"]]),
                estimate = sm[["estimate"]],
                SE       = sm[["SE"]],
                df       = sm[["df"]],
                t_ratio  = sm[["t.ratio"]],
                p_value  = sm[["p.value"]],
                lower_CL = sm[["lower.CL"]],
                upper_CL = sm[["upper.CL"]]
            )
        }
        """
    )(contrast_obj)

    return pl.DataFrame({
        "gene":     [gene]     * len(output.rx2("visit")),
        "platform": [platform] * len(output.rx2("visit")),
        "visit":    list(output.rx2("visit")),
        "est":      list(output.rx2("estimate")),
        "se":       list(output.rx2("SE")),
        "df":       list(output.rx2("df")),
        "t":        list(output.rx2("t_ratio")),
        "p":        list(output.rx2("p_value")),
        "ci_lo":    list(output.rx2("lower_CL")),
        "ci_hi":    list(output.rx2("upper_CL")),
    })


def run_mmrm_contrasts(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit per-(gene × platform) MMRMs; return per-genotype contrasts and
    recessive interaction tests.

    Returns
    -------
    contrasts : pl.DataFrame
        Per-(gene × visit × genotype × platform) TZP-SEMA contrasts (24 rows).
    recessive : pl.DataFrame
        Per-(gene × visit × platform) recessive interaction test (16 rows):
        (TZP-SEMA in CC) vs mean(TZP-SEMA in GG, TZP-SEMA in GC).

    A separate MMRM is fit for each (gene × platform) cell because Olink
    and SomaScan measure different physical quantities with different error
    structures; a joint model would impose artificial cross-platform
    correlation. Each uses us(visit_f | USUBJID) for within-subject W24↔W72
    covariance.
    """
    bindings = _get_r_bindings()
    mmrm_r = bindings["mmrm"]
    emmeans_r = bindings["emmeans"]
    r_pairs = bindings["pairs"]
    r_summary = bindings["summary"]
    r_as_df = bindings["as_data_frame"]
    r_as_character = bindings["as_character"]
    r_transform = bindings["transform"]
    r_gc = bindings["gc"]

    all_results: list[pl.DataFrame] = []
    all_recessive: list[pl.DataFrame] = []

    for gene in ("PPY", "REG4"):
        for platform in ("olink", "soma"):
            cell = _prepare_cell_frame(frame, gene, platform)
            if cell.height == 0:
                continue

            n_subj = cell["USUBJID"].n_unique()
            print(
                f"\n==== {gene} × {platform} — "
                f"n={cell.height} rows, {n_subj} subjects ===="
            )

            cell_counts = _compute_cell_counts(cell)
            r_data = _to_r_data(cell)

            try:
                fit = mmrm_r.mmrm(
                    formula=Formula(FORMULA_STR),
                    data=r_data,
                    method="Satterthwaite",
                    reml=True,
                )
            except Exception as e:
                print(f"  mmrm error: {e}")
                continue

            # emmeans: marginal means of arm_f at each (visit_f × geno_f)
            # cov.keep=character(0) prevents emmeans from treating binary
            # covariates (sex coded 0/1) as categorical and averaging over
            # their levels; all continuous covariates are held at their means.
            emm = emmeans_r.emmeans(
                fit,
                specs=ro.Formula("~ arm_f | visit_f + geno_f"),
                weights="cells",
                **{"cov.keep": ro.StrVector([])},
            )

            # Pairwise contrasts: TZP - SEMA within each (visit_f, geno_f) cell
            contrast_obj = r_pairs(emm, reverse=True, adjust="none")
            contrast_summary = r_as_df(
                r_summary(contrast_obj, infer=ro.BoolVector([True, True]))
            )
            # Convert factor columns to character for clean pandas conversion
            visit_char = r_as_character(contrast_summary.rx2("visit_f"))
            geno_char = r_as_character(contrast_summary.rx2("geno_f"))
            contrast_summary = r_transform(
                contrast_summary, visit_f=visit_char, geno_f=geno_char
            )

            with pandas2ri.converter.context():
                contrasts_pd = pandas2ri.rpy2py(contrast_summary)
                contrasts_pl = pl.from_pandas(contrasts_pd)

            contrasts_pl = (
                contrasts_pl.rename({
                    "estimate": "est",
                    "SE": "se",
                    "t.ratio": "t",
                    "p.value": "p",
                    "lower.CL": "ci_lo",
                    "upper.CL": "ci_hi",
                    "visit_f": "visit",
                    "geno_f": "geno",
                })
                .select(["visit", "geno", "est", "se", "df", "t", "p", "ci_lo", "ci_hi"])
                .with_columns(
                    pl.lit(gene).alias("gene"),
                    pl.lit(platform).alias("platform"),
                )
            )

            contrasts_pl = contrasts_pl.join(cell_counts, on=["visit", "geno"], how="left")
            all_results.append(contrasts_pl)

            # Recessive interaction test: C/C vs G-allele carriers
            all_recessive.append(
                _extract_recessive_contrast(contrast_obj, emmeans_r, gene, platform)
            )

            del fit, emm, contrast_obj, contrast_summary, r_data
            gc.collect()
            r_gc(full=ro.BoolVector([True]))

    if not all_results:
        raise RuntimeError("No models converged — check input data")

    contrasts = pl.concat(all_results).select([
        "gene", "visit", "geno", "platform",
        "n_tzp", "n_sema",
        "est", "se", "df", "t", "p", "ci_lo", "ci_hi",
    ])
    recessive = pl.concat(all_recessive).select([
        "gene", "visit", "platform",
        "est", "se", "df", "t", "p", "ci_lo", "ci_hi",
    ])
    return contrasts, recessive


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def build(verbose: bool = True) -> tuple[Path, Path]:
    """Run the full pipeline; returns paths to the two output parquets."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[1/4] Extracting rs1800437 genotypes from {VCF_PATH.name}")
    geno = extract_rs1800437_genotypes()
    if verbose:
        print(f"      → {geno.height} samples")

    if verbose:
        print("[2/4] Building subject-level sample link (genotype × ADSL × PCs)")
    sample_link = build_sample_link(geno)
    if verbose:
        print(f"      → {sample_link.height} TZP/SEMA subjects genotyped")
        print(sample_link.group_by(["arm", "dosage"]).len().sort(["arm", "dosage"]))

    if verbose:
        print("[3/4] Assembling QC-filtered PPY+REG4 long frame")
    frame = build_focused_frame(sample_link)
    if verbose:
        print(f"      → {frame.height} rows")

    frame_path = OUT_DIR / "genotype_E354Q_frame.parquet"
    frame.write_parquet(frame_path)

    if verbose:
        print("[4/4] Fitting MMRM and extracting per-genotype TZP-SEMA contrasts")
    contrasts, recessive = run_mmrm_contrasts(frame)
    contrasts_path = OUT_DIR / "genotype_E354Q_contrasts.parquet"
    contrasts.write_parquet(contrasts_path)
    recessive_path = OUT_DIR / "genotype_E354Q_recessive.parquet"
    recessive.write_parquet(recessive_path)
    if verbose:
        print(f"      → wrote {contrasts_path.name} ({contrasts.height} rows)")
        with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=140):
            print(
                contrasts.select(
                    [
                        "gene",
                        "visit",
                        "geno",
                        "n_tzp",
                        "n_sema",
                        pl.col("est").round(3),
                        pl.col("se").round(3),
                        pl.col("p")
                        .map_elements(
                            lambda v: round(v, 4) if v is not None else None,
                            return_dtype=pl.Float64,
                        )
                        .alias("p"),
                    ]
                )
            )
        print(f"      → wrote {recessive_path.name} ({recessive.height} rows)")
        with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=140):
            print(
                recessive.select(
                    [
                        "gene",
                        "visit",
                        "platform",
                        pl.col("est").round(3),
                        pl.col("se").round(3),
                        pl.col("p")
                        .map_elements(
                            lambda v: round(v, 4) if v is not None else None,
                            return_dtype=pl.Float64,
                        )
                        .alias("p"),
                    ]
                )
            )

    return frame_path, contrasts_path, recessive_path


if __name__ == "__main__":
    build()
