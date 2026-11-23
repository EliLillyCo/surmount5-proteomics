"""GIPR rs1800437 (E354Q) × PPY pharmacogenomic pipeline.

Extracts rs1800437 genotypes, assembles a QC-filtered PPY long frame
across Olink and SomaScan, and fits per-(gene × platform) MMRMs to extract
per-genotype TZP-SEMA contrasts. Writes two parquets, both read by
``figures/supp_fig_e354q.py``:

  ``analysis/outputs/genotype_E354Q_frame.parquet``     — analysis-ready
      long frame (USUBJID × platform × probe × visit × value/base/dosage
      /arm/age/sex/PC1..10) for PPY, TZP/SEMA arms only,
      genotyped subjects only, QC-passing samples only.

  ``analysis/outputs/genotype_E354Q_contrasts.parquet`` — per-(gene × visit
      × genotype × platform) TZP-SEMA contrasts (est, SE, df, t, p, ci_lo,
      ci_hi, n_tzp, n_sema). A separate MMRM is fit for each (gene ×
      platform) cell, each with ``us(visit_f | USUBJID)`` within-subject
      covariance.

Genotypes reach subjects through the pheno-geno linking table, corrected by the QC's
``dna_relink`` corrections (``analysis/qc/dna_relink.py``) and checked against the QC's
genotype links: the stage stops if a QC link names a participant the corrected table
gives no genome, or marks a link suspect.

The model is described in ``analysis/pharmacogenomics/README.md``.
Each per-(gene × platform) MMRM uses the formula:

    value ~ 0 + visit_f:arm_f:geno_f
          + base + age + sex + PC1 + PC2 + PC3
          + us(visit_f | USUBJID)

The per-genotype TZP-SEMA contrast at each visit is the (TZP − SEMA)
coefficient difference within that platform's own fit.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import polars as pl
import pysam
import rpy2.robjects as ro
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.packages import importr

from analysis.design import (
    BASELINE_WEEK,
    POST_BASELINE_WEEKS,
    RAW_CONTROL_VALUE,
    RAW_TREAT_VALUE,
    WEEKS,
)
from analysis.mmrm.constants import MMRM_CONTROL_ARGS
from analysis.output_layout import ANALYSIS_OUTPUTS, E354Q_CONTRASTS, E354Q_FRAME
from analysis.qc.dna_relink import DNA_RELINK, read_corrections, relink_map
from analysis.qc_contract import (
    PLATFORM_SPECS,
    PLATFORM_STUDIES,
    PLATFORMS,
    WEEK_COL,
    cohort_samples,
    dna_link_path,
    long_table_path,
)
from analysis.qc_paths import required_configured_path
from paths import PHENO_DIR, QA_ROOT, required_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Read input paths from paths.yaml: the chr19 pVCF and the pheno-geno linking table have their
# own keys; the genetic PCs are found under qa_root. The QC corrections TOMLs
# (<study>_qc_corrections) supply the dna_relink corrections applied to the linking table.
if QA_ROOT is None:
    raise RuntimeError("paths.yaml must define qa_root to run analysis/pharmacogenomics/run.py")
VCF_PATH = required_path("surmount5_chr19_pvcf")
LINKING_CSV = required_path("surmount5_pheno_geno_linking")
ADSL_PARQUET = PHENO_DIR / "adsl.parquet"
# Read QC outputs through analysis.qc_contract.
OLINK_LONG = long_table_path("olink")
SOMA_LONG = long_table_path("soma")

# ---------------------------------------------------------------------------
# rs1800437 + PPY constants
# ---------------------------------------------------------------------------
SNP_CHROM = "chr19"
SNP_POS = 45678134
SNP_REF = "G"
SNP_ALT = "C"

PROBES = {
    "olink": {"OID44867": "PPY"},
    "soma": {"4588-1": "PPY"},
}

PC_COLS = [f"PC{i}" for i in range(1, 11)]


PCA_GLOB = "3101_*_data*_qced_maf0.01_pruned_genome_pc.csv"


def _resolve_pca_csv(qa_root: Path = QA_ROOT) -> str:
    """The one genetic-PC CSV under ``qa_root``; raises when there is none or several."""
    matches = sorted((qa_root / "3000_genomic_data_summary/3100_pca").glob(PCA_GLOB))
    if not matches:
        raise FileNotFoundError(
            "Missing genetic PCs. Run: pixi run python analysis/pipelines/run_wgs_pcs.py"
        )
    if len(matches) > 1:
        listing = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"Expected exactly one genetic-PC CSV matching {PCA_GLOB}, found {len(matches)}:"
            f"\n  {listing}"
        )
    return str(matches[0])


def _require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    """Raise when ``frame`` has more than one row for a key, naming the first offenders."""
    duplicated = frame.filter(frame.select(keys).is_duplicated())
    if duplicated.height:
        raise ValueError(
            f"{label}: {duplicated.height} rows share a ({', '.join(keys)}) key; first "
            f"{duplicated.select(keys).unique().head(5).rows()}"
        )


def _platform_corrections() -> dict[str, tuple]:
    """Each platform run's QC corrections, from its ``<study>_qc_corrections`` paths key."""
    return {
        platform: read_corrections(required_configured_path(f"{study}_qc_corrections"))
        for platform, study in PLATFORM_STUDIES.items()
    }


def _qc_dna_links() -> dict[str, pl.DataFrame]:
    """Each platform run's QC genotype link (``dna_link.parquet``): USUBJID and link_status."""
    return {
        platform: pl.read_parquet(dna_link_path(platform), columns=["USUBJID", "link_status"])
        for platform in PLATFORM_STUDIES
    }


def dna_relinks(corrections: dict[str, tuple]) -> tuple:
    """The ``dna_relink`` corrections, which every platform run must carry alike.

    A relink is subject-level, so it is written in the corrections TOML of both platform
    runs; this stage reads both and refuses to choose between them.

    Raises:
        ValueError: If the platforms' ``dna_relink`` entries differ.
    """
    entries = {
        platform: tuple(c for c in entries if c.type == DNA_RELINK)
        for platform, entries in corrections.items()
    }
    keys = {
        frozenset((c.from_usubjid_sha256, c.usubjid_sha256) for c in found)
        for found in entries.values()
    }
    if len(keys) > 1:
        raise ValueError(
            "dna_relink entries differ between the platforms' QC corrections; a relink is "
            "subject-level and is written in both"
        )
    return next(iter(entries.values()))


def corrected_link(link: pl.DataFrame, relinks: tuple, subjects: list[str]) -> pl.DataFrame:
    """The pheno-geno linking table with each relinked genome filed under its owner.

    Args:
        link: The linking table (``dx_usubjid``, ``dx_dna_sm_id``, ...).
        relinks: The ``dna_relink`` corrections (``dna_relinks``).
        subjects: ADSL USUBJIDs, which an owner is resolved against.

    Returns:
        ``link`` with ``dx_usubjid`` replaced by the owner on each relinked row; the
        participant it was filed under is left with no row.
    """
    mapping = relink_map(relinks, link["dx_usubjid"].to_list(), subjects)
    return link.with_columns(pl.col("dx_usubjid").replace(mapping))


def check_qc_links(link: pl.DataFrame, qc_links: dict[str, pl.DataFrame]) -> None:
    """Checks the corrected linking table against each platform's QC genotype link.

    The QC links proteomics to genomes by the same corrections, so every participant
    its link names must hold a genome in ``link``; and no link the QC's identity check
    is anything but ``ok`` (``suspect``, ``dropped``, ``hold`` or a null status) may be used.

    Raises:
        ValueError: If a QC link names a participant ``link`` gives no genome (the QC
            ran without this stage's corrections), or holds a link whose ``link_status`` is not ``ok``. Errors give
            counts, never an ID.
    """
    linked = set(link["dx_usubjid"].to_list())
    for platform, qc in qc_links.items():
        stale = set(qc["USUBJID"].to_list()) - linked
        if stale:
            raise ValueError(
                f"{platform} QC genotype link names {len(stale)} participant(s) the corrected "
                "linking table gives no genome; rerun the QC with the same corrections"
            )
        not_ok = qc.filter(pl.col("link_status").ne_missing("ok"))
        if not_ok.height:
            counts = dict(not_ok.group_by("link_status").len().sort("link_status").rows())
            raise ValueError(
                f"{platform} QC genotype link is not ok after the corrections ({counts} by "
                "status); correct it (dna_relink) or resolve it before the genotype analysis"
            )


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
    """One row per genotyped TZP/SEMA subject: arm, rs1800437 dosage, age, sex and PCs.

    The linking table is corrected by the QC's ``dna_relink`` corrections first and
    checked against the QC's genotype links (``check_qc_links``).
    """
    adsl = pl.read_parquet(
        ADSL_PARQUET,
        columns=["USUBJID", "AGE", "SEX", "TRT01A"],
    )
    link = corrected_link(
        pl.read_csv(LINKING_CSV),
        dna_relinks(_platform_corrections()),
        adsl["USUBJID"].to_list(),
    )
    pcs = (
        pl.read_csv(_resolve_pca_csv())
        .rename({"IID": "dx_dna_sm_id"})
        .select(["dx_dna_sm_id"] + PC_COLS)
    )
    # Each table is one row per key, so a join can only drop subjects, never repeat them.
    _require_unique(adsl, ["USUBJID"], "ADSL")
    _require_unique(geno, ["dx_dna_sm_id"], "genotypes")
    _require_unique(pcs, ["dx_dna_sm_id"], "genetic PCs")

    sample_link = (
        link.join(adsl, left_on="dx_usubjid", right_on="USUBJID", how="inner", validate="m:1")
        .join(geno, on="dx_dna_sm_id", how="inner", validate="m:1")
        .join(pcs, on="dx_dna_sm_id", how="inner", validate="m:1")
        .filter(pl.col("TRT01A").is_in([RAW_TREAT_VALUE, RAW_CONTROL_VALUE]))
        .rename(
            {"dx_usubjid": "USUBJID", "TRT01A": "arm", "AGE": "age", "SEX": "sex_str"}
        )
    )
    # One DNA sample per subject: the proteomics join below would otherwise repeat the subject.
    _require_unique(sample_link, ["USUBJID"], "subject-level sample link")
    check_qc_links(link, _qc_dna_links())
    sample_link = sample_link.with_columns(
        pl.when(pl.col("sex_str") == "M").then(1.0).otherwise(0.0).alias("sex"),
        pl.col("age").cast(pl.Float64),
        pl.col("dosage").cast(pl.Float64),
    )
    return sample_link.select(
        ["USUBJID", "arm", "dosage", "age", "sex"] + PC_COLS
    )


def _build_sample_map(platform: str) -> pl.DataFrame:
    """SampleID → USUBJID + analysis week, the QC's analysis cohort (Status != FAIL)."""
    sid = PLATFORM_SPECS[platform].sample_id
    return (
        cohort_samples(platform)
        .select(pl.col(sid).alias("SampleID"), "USUBJID", pl.col(WEEK_COL).alias("week"))
        .collect()
    )


# ---------------------------------------------------------------------------
# Step 3: assemble QC-filtered PPY long frame for both platforms
# ---------------------------------------------------------------------------
def build_focused_frame(sample_link: pl.DataFrame) -> pl.DataFrame:
    """One row per (USUBJID × probe × visit ∈ {W24, W72}) with baseline +
    covariates. Only the PPY probes; only genotyped TZP/SEMA subjects.
    """
    olink_map = _build_sample_map("olink")
    soma_map = _build_sample_map("soma")

    eval_subj = sample_link.select("USUBJID")
    olink_map = olink_map.join(eval_subj, on="USUBJID", how="inner")
    soma_map = soma_map.join(eval_subj, on="USUBJID", how="inner")

    # NPX with the vendor's AssayQC FAIL datapoints missing, as in the MMRM inputs
    # (analysis.platform_inputs).
    olink = (
        pl.scan_parquet(OLINK_LONG)
        .filter(pl.col("OlinkID").is_in(list(PROBES["olink"])), pl.col("AssayQC") != "FAIL")
        .join(olink_map.lazy(), on="SampleID", how="inner")
        .select(
            [
                "USUBJID",
                pl.col("OlinkID").alias("probe"),
                "week",
                pl.col("NPX").alias("value"),
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
            pl.lit("soma").alias("platform"),
            pl.col("probe").replace_strict(PROBES["soma"]).alias("gene"),
        )
        .collect(engine="streaming")
    )

    long_df = pl.concat([olink, soma])
    # One value per subject x platform x probe x week: the baseline and sample joins below
    # are many-to-one only if this holds.
    _require_unique(long_df, ["USUBJID", "platform", "probe", "week"], "PPY long frame")
    visit_map = {week: f"Week{week}" for week in WEEKS}
    long_df = long_df.with_columns(
        pl.col("week").replace_strict(visit_map).alias("visit")
    )

    base = long_df.filter(pl.col("week") == BASELINE_WEEK).select(
        ["USUBJID", "platform", "probe", "gene", pl.col("value").alias("base")]
    )
    frame = (
        long_df.filter(pl.col("week").is_in(list(POST_BASELINE_WEEKS)))
        .join(base, on=["USUBJID", "platform", "probe", "gene"], how="inner", validate="m:1")
        .join(sample_link, on="USUBJID", how="inner", validate="m:1")
        # Fixed row order for the Supplementary Fig. 11 jitter.
        .sort(["USUBJID", "platform", "probe", "week"])
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
        pl.when(pl.col("arm") == RAW_TREAT_VALUE)
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
        "visit_f": [f"W{week}" for week in POST_BASELINE_WEEKS],
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


def run_mmrm_contrasts(
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Fit per-(gene × platform) MMRMs; return per-genotype contrasts.

    Returns
    -------
    contrasts : pl.DataFrame
        Per-(gene × visit × genotype × platform) TZP-SEMA contrasts, one row per gene, visit, genotype and platform.
    A separate MMRM is fit for each (gene × platform) cell. Each uses
    us(visit_f | USUBJID) for within-subject W24↔W72 covariance.
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

    for gene in ("PPY",):
        for platform in PLATFORMS:
            cell = _prepare_cell_frame(frame, gene, platform)
            if cell.height == 0:
                raise ValueError(f"No complete rows for {gene} x {platform}; refusing to skip the cell")

            n_subj = cell["USUBJID"].n_unique()
            print(
                f"\n==== {gene} × {platform} — "
                f"n={cell.height} rows, {n_subj} subjects ===="
            )

            cell_counts = _compute_cell_counts(cell)
            r_data = _to_r_data(cell)

            fit = mmrm_r.mmrm(
                formula=Formula(FORMULA_STR),
                data=r_data,
                control=mmrm_r.mmrm_control(**MMRM_CONTROL_ARGS),
            )

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
    return contrasts


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def build(verbose: bool = True) -> tuple[Path, Path]:
    """Run the full pipeline; returns paths to the two output parquets."""
    ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)

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
        print("[3/4] Assembling QC-filtered PPY long frame")
    frame = build_focused_frame(sample_link)
    if verbose:
        print(f"      → {frame.height} rows")

    frame_path = E354Q_FRAME
    frame.write_parquet(frame_path)

    if verbose:
        print("[4/4] Fitting MMRM and extracting per-genotype TZP-SEMA contrasts")
    contrasts = run_mmrm_contrasts(frame)
    contrasts_path = E354Q_CONTRASTS
    contrasts.write_parquet(contrasts_path)
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

    return frame_path, contrasts_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line, which takes no options: ``--help`` prints usage and exits.

    Args:
        argv: Arguments after the program name; None reads ``sys.argv``.

    Returns:
        An empty namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Takes no options; inputs come from paths.yaml and the QC outputs, outputs go to analysis/outputs/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse the (empty) command line, then run the pipeline."""
    parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
