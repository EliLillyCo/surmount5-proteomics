"""Per-protein treatment x visit x sex MMRM and the sex interaction on TZP - SEMA.

Model: ``{resp}_DIFF ~ {resp}BL + AGE + TRT01A*VISITNUM*SEX + us(VISITNUM | SEX / USUBJID)``,
i.e. a separate unstructured covariance for each sex. Every contrast is an explicit
coefficient vector over an emmeans grid whose row order is asserted, so no sign depends on
label parsing or factor-level sorting.
"""

from __future__ import annotations

import gc

import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from tqdm import tqdm

from analysis.parallel import default_chunksize, forkserver_pool, resolve_n_jobs
from analysis.platform_inputs import CONTROL_VALUE, TREAT_VALUE, VISIT_LABELS
from analysis.rinterface import r_gc, to_r_factors
from analysis.statistics import bh_fdr

from .constants import mmrm_control_r_call
from .preparation import prepare_data

# Factor-level order of SEX in the fit. The interaction is defined as female minus male, and
# the grid-order assertion below fails if this order does not give (F, F, M, M).
SEX_LEVELS = ("F", "M")

_FORMULA = "us(VISITNUM | SEX / USUBJID)"

# R side of one fit. Returns plain data.frames (no factors) so the Python side only converts.
# ``mmrm_control_r_call`` renders MMRM_CONTROL_ARGS, the primary MMRM's settings.
_R_FIT = f"""
function(d, formula_str) {{
  plain <- function(x) {{
    x <- as.data.frame(x, stringsAsFactors = FALSE)
    x[] <- lapply(x, function(col) if (is.factor(col)) as.character(col) else col)
    x
  }}
  fit <- mmrm::mmrm(formula = stats::as.formula(formula_str), data = d,
                    control = mmrm::{mmrm_control_r_call()})
  summ <- function(emm, method, ...) {{
    plain(summary(emmeans::contrast(emm, method = method, adjust = "none"),
                  infer = c(TRUE, TRUE)))
  }}
  within_emm <- emmeans::emmeans(fit, ~ TRT01A | VISITNUM + SEX)
  avg_emm <- emmeans::emmeans(fit, ~ TRT01A | VISITNUM, weights = "proportional")
  int_emm <- emmeans::emmeans(fit, ~ TRT01A * SEX | VISITNUM)
  int_con <- emmeans::contrast(int_emm, method = list(int = c(-1, 1, 1, -1)), adjust = "none")
  vc <- mmrm::VarCorr(fit)
  list(
    within = summ(within_emm, list(d = c(-1, 1))),
    avg = summ(avg_emm, list(d = c(-1, 1))),
    int = plain(summary(int_con, infer = c(TRUE, TRUE))),
    grid = plain(int_emm@grid),
    var = do.call(rbind, lapply(names(vc), function(g) {{
      data.frame(SEX = g, VISITNUM = colnames(vc[[g]]), var = diag(vc[[g]]),
                 stringsAsFactors = FALSE)
    }})),
    converged = isTRUE(attr(fit, "converged"))
  )
}}
"""


def _to_pl(r_df) -> pl.DataFrame:
    with pandas2ri.converter.context():
        return pl.from_pandas(pandas2ri.rpy2py(r_df))


def _cell_counts(data: pl.DataFrame, treat_trt: str, control_trt: str, visit: int) -> dict[str, int]:
    out = {}
    for sex in SEX_LEVELS:
        for label, arm in (("TZP", treat_trt), ("SEMA", control_trt)):
            out[f"n_{sex}_{label}"] = data.filter(
                (pl.col("SEX") == sex) & (pl.col("TRT01A") == arm) & (pl.col("VISITNUM") == visit)
            ).height
    return out


def _fit_once(
    data: pl.DataFrame,
    *,
    response_var: str,
    response_var_bl: str,
    control_trt: str,
    treat_trt: str,
    visit_labels: dict[int, int],
) -> pl.DataFrame:
    """One per-sex-covariance fit and its contrasts: one row per visit."""
    visits = sorted(data["VISITNUM"].unique().to_list())
    cols = ["USUBJID", "TRT01A", "VISITNUM", "SEX", "AGE", response_var_bl, f"{response_var}_DIFF"]
    pdf = data.select(cols).rechunk().to_pandas()
    r_data = to_r_factors(
        pdf,
        {"TRT01A": control_trt, "VISITNUM": str(visits[0]), "USUBJID": None, "SEX": SEX_LEVELS[0]},
    )
    formula = f"{response_var}_DIFF ~ {response_var_bl} + AGE + TRT01A*VISITNUM*SEX + {_FORMULA}"
    res = ro.r(_R_FIT)(r_data, formula)
    named = dict(zip(res.names, res))
    converged = bool(named["converged"][0])
    within, avg, inter, grid = (_to_pl(named[k]) for k in ("within", "avg", "int", "grid"))
    var = _to_pl(named["var"])

    # The interaction contrast c(-1, 1, 1, -1) is (TZP - SEMA)_F - (TZP - SEMA)_M only for
    # this grid order within each visit.
    for visit in visits:
        g = grid.filter(pl.col("VISITNUM") == str(visit))
        if g["TRT01A"].to_list() != [control_trt, treat_trt] * 2 or g["SEX"].to_list() != [
            "F",
            "F",
            "M",
            "M",
        ]:
            raise RuntimeError(
                f"interaction grid order is not (SEMA,F),(TZP,F),(SEMA,M),(TZP,M) at visit {visit}: "
                f"{g.select('TRT01A', 'SEX').rows()}"
            )

    rows = []
    for visit in visits:
        v = str(visit)
        row = {"time": visit_labels[visit], **_cell_counts(data, treat_trt, control_trt, visit)}
        for sex in SEX_LEVELS:
            w = within.filter((pl.col("VISITNUM") == v) & (pl.col("SEX") == sex)).row(0, named=True)
            row |= {
                f"est_{sex}": w["estimate"], f"se_{sex}": w["SE"], f"df_{sex}": w["df"],
                f"ci_lo_{sex}": w["lower.CL"], f"ci_hi_{sex}": w["upper.CL"], f"p_{sex}": w["p.value"],
            }
        a = avg.filter(pl.col("VISITNUM") == v).row(0, named=True)
        i = inter.filter(pl.col("VISITNUM") == v).row(0, named=True)
        row |= {
            "est_avg": a["estimate"], "se_avg": a["SE"],
            "est_int": i["estimate"], "se_int": i["SE"], "df_int": i["df"], "t_int": i["t.ratio"],
            "ci_lo_int": i["lower.CL"], "ci_hi_int": i["upper.CL"], "p_int": i["p.value"],
        }
        vf = var.filter((pl.col("SEX") == "F") & (pl.col("VISITNUM") == v))["var"].item()
        vm = var.filter((pl.col("SEX") == "M") & (pl.col("VISITNUM") == v))["var"].item()
        row["var_ratio_FM"] = vf / vm
        rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.lit(converged).alias("converged"))


def fit_sex_interaction(
    protein: str,
    data: pl.DataFrame,
    *,
    response_var: str,
    response_var_bl: str,
    control_trt: str,
    treat_trt: str,
    visit_labels: dict[int, int],
) -> pl.DataFrame:
    """Fit the sex-interaction MMRM for one protein.

    Args:
        protein: Protein identifier, used in error messages and as ``marker``.
        data: This protein's post-baseline rows with ``{response_var}_DIFF`` present and
            TRT01A already space-stripped.
        response_var: Response name (e.g. ``NPX``).
        response_var_bl: Baseline column name (e.g. ``NPXBL``).
        control_trt: Comparator (SEMA) label, the TRT01A reference level.
        treat_trt: Treatment (TZP) label.
        visit_labels: VISITNUM -> week.

    Returns:
        One row per visit with the within-sex, sex-averaged and interaction contrasts,
        cell counts and diagnostics.

    Raises:
        RuntimeError: Naming ``protein`` when the fit errors, does not converge, has a
            non-finite reported contrast, or fails the grid-order check.
    """
    try:
        rows = _fit_once(
            data, response_var=response_var, response_var_bl=response_var_bl,
            control_trt=control_trt, treat_trt=treat_trt, visit_labels=visit_labels,
        )
        if not bool(rows["converged"].all()):
            raise RuntimeError("did not converge")
        if not _usable(rows):
            raise RuntimeError("non-finite reported contrast")
        return _finish(protein, rows)
    except Exception as exc:
        raise RuntimeError(f"Sex-interaction fit failed for {protein}: {exc}") from exc
    finally:
        gc.collect()
        r_gc()


def _usable(contrast_rows: pl.DataFrame) -> bool:
    """True when every reported estimate, SE, df and p is finite (an R NA arrives as null)."""
    cols = [c for c in contrast_rows.columns if c.split("_")[0] in {"est", "se", "df", "p"}]
    finite = (pl.col(c).is_finite().fill_null(False) for c in cols)
    return bool(contrast_rows.select(pl.all_horizontal(finite).all()).item())


def _finish(protein: str, rows: pl.DataFrame) -> pl.DataFrame:
    return rows.drop("converged").with_columns(pl.lit(protein).alias("marker"))


# ============================================================================
# ALL PROTEINS: FDR FAMILIES, DIRECTION, PARALLEL RUNNER
# ============================================================================

_P_COLS = ("p_int", "p_F", "p_M")


def _bh_within(df: pl.DataFrame, p_col: str, by: list[str]) -> pl.Series:
    """BH q-values of ``p_col`` computed separately within each ``by`` group, in row order."""
    idx = df.with_row_index("_i")
    parts = [
        g.select("_i", pl.Series("_q", bh_fdr(g[p_col].to_numpy())))
        for g in idx.partition_by(by, maintain_order=True)
    ]
    return pl.concat(parts).sort("_i")["_q"]


def add_fdr_and_direction(rows: pl.DataFrame) -> pl.DataFrame:
    """Add the BH families and ``int_direction`` to one platform's protein x visit rows.

    Families: ``fdr_int`` per visit; ``fdr_F_desc`` / ``fdr_M_desc`` per visit and sex.

    Args:
        rows: All proteins of one platform, with marker, time, est_F, est_M and the
            p-values ``p_int``, ``p_F``, ``p_M``.

    Returns:
        ``rows`` plus ``fdr_int``, ``fdr_F_desc``, ``fdr_M_desc``, ``int_direction``.
    """
    for c in _P_COLS:
        assert rows[c].is_finite().all(), f"non-finite {c} reached the BH family"
    out = rows.with_columns(
        _bh_within(rows, "p_int", ["time"]).alias("fdr_int"),
        _bh_within(rows, "p_F", ["time"]).alias("fdr_F_desc"),
        _bh_within(rows, "p_M", ["time"]).alias("fdr_M_desc"),
    )
    f, m = pl.col("est_F"), pl.col("est_M")
    return out.with_columns(
        pl.when((f.sign() != m.sign()) & (f != 0) & (m != 0))
        .then(pl.lit("opposite_sign"))
        .when(f.abs() > m.abs())
        .then(pl.lit("larger_in_F"))
        .when(f.abs() < m.abs())
        .then(pl.lit("smaller_in_F"))
        .otherwise(pl.lit("equal"))
        .alias("int_direction")
    )


def _fit_protein_task(args) -> pl.DataFrame:
    """Module-level pool worker (picklable)."""
    protein, data, kw = args
    return fit_sex_interaction(protein, data, **kw)


def run_sex_interaction(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    annotation: pl.DataFrame,
    response_var: str,
    response_var_bl: str,
    protein_id_col: str,
    baseline_visitnum: int,
    n_jobs: int = -1,
    chunksize: int | None = None,
) -> pl.DataFrame:
    """Fit the sex-interaction MMRM for every protein of one platform.

    The data are prepared as for the primary MMRM (``prepare_data`` drops baseline rows and
    adds the change score). Rows outside the SEMA and TZP arms are dropped, since the model has
    exactly those two TRT01A levels.

    Args:
        data: Long frame with all visits.
        annotation: Table keyed by ``protein_id_col`` carrying ``Assay``.
        response_var: Response name.
        response_var_bl: Baseline column name.
        protein_id_col: Protein identifier column.
        baseline_visitnum: Baseline VISITNUM, dropped by the prep.
        n_jobs: Worker processes (-1 = all CPUs - 1).
        chunksize: Proteins per pool chunk.

    Returns:
        One row per protein x visit, sorted by marker and time, with FDR columns, direction
        and ``Assay``.

    Raises:
        RuntimeError: Naming the protein whose fit failed.
    """
    prepared = (
        prepare_data(data, baseline_visitnum, response_var, response_var_bl, protein_id_col)
        .with_columns(pl.col("TRT01A").str.replace_all(" ", ""))
        .filter(pl.col("TRT01A").is_in([CONTROL_VALUE, TREAT_VALUE]))
        .collect()
    )
    proteins = prepared[protein_id_col].unique().sort().to_list()
    partitions = prepared.partition_by(protein_id_col, as_dict=True, maintain_order=True)
    kw = dict(
        response_var=response_var, response_var_bl=response_var_bl,
        control_trt=CONTROL_VALUE, treat_trt=TREAT_VALUE, visit_labels=VISIT_LABELS,
    )
    n_jobs = resolve_n_jobs(n_jobs)
    if chunksize is None:
        chunksize = default_chunksize(len(proteins), n_jobs)
    print(f"Fitting the sex-interaction model for {len(proteins)} proteins using {n_jobs} cores...")

    results = []
    with forkserver_pool(n_jobs, maxtasksperchild=50) as pool:
        tasks = ((p, partitions[(p,)], kw) for p in proteins)
        with tqdm(total=len(proteins), desc="Sex interaction", unit="protein") as pbar:
            for res in pool.imap_unordered(_fit_protein_task, tasks, chunksize=chunksize):
                results.append(res)
                pbar.update(1)

    rows = add_fdr_and_direction(pl.concat(results).sort("marker", "time"))
    return rows.join(
        annotation.select(pl.col(protein_id_col).alias("marker"), "Assay").unique("marker"),
        on="marker",
        how="left",
    ).sort("marker", "time")
