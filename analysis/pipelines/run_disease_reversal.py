"""Disease-signature reversal for Fig 3b: HPA disease signatures against the Olink TZP − SEMA z.

    nice -n 10 pixi run python analysis/pipelines/run_disease_reversal.py [--week <LATE_WEEK>]

Reads the TZP − SEMA contrast of the ``covar_AGE_SEX`` Olink MMRM at ``--week``,
the week-0 Olink NPX of the MMRM input (both arms) and the HPA pan-disease blood
atlas (fetched on first use, ``analysis/external/hpa_disease_atlas.py``), and writes one
aggregate row per adult disease to ``analysis/outputs/disease_reversal_olink_wk<week>.parquet``:
``Disease, Class, n_genes, n_baseline_subjects, neg_rho, null_sd, p, mc_se, fdr`` (``n_genes``: the
signature's size in assays, each assay once). The method
is in ``analysis/disease_reversal.py``. Olink only: the HPA atlas is Olink Explore
1536. The 1,000,000 null draws per disease (``ROTATION_DRAWS``) run in one process.

The Olink assay -> gene mapping is the assay annotation the pathway CLI reads (stage
``vendor_assay_tables``). An HPA gene symbol is matched to an assay gene by symbol first; a symbol we do not measure
falls back to its Ensembl gene ID, resolved through the pinned release 116 cross-references (``analysis/external/ensembl_uniprot.py``).
"""

from __future__ import annotations

from analysis.parallel import pin_blas_threads

# Pin the BLAS/OpenMP pools to one thread before numpy is imported, so the rotation null's
# matrix products run single-threaded whatever the host's core count.
pin_blas_threads()

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from analysis.disease_reversal import (  # noqa: E402
    PLATFORM,
    assay_level_z,
    baseline_assay_matrix,
    disease_signatures,
    ensg_alias,
    output_path,
    reversal_table,
)
from analysis.external.hpa_disease_atlas import read_disease_de  # noqa: E402
from analysis.mmrm_outputs import ACROSS_CONTRAST  # noqa: E402
from analysis.external.ensembl_uniprot import read_uniprot_xref  # noqa: E402
from analysis.external.vendor_assay_tables import ANNOTATION_PATH, read_annotation  # noqa: E402
from analysis.pathway.loaders import load_assay_genes  # noqa: E402
from analysis.pathway.preparation import contrast_statistics  # noqa: E402
from analysis.design import BASELINE_VISITNUM, LATE_WEEK, VISIT_LABELS  # noqa: E402
from analysis.platform_inputs import EXCLUDED_ASSAYS, prepare_mmrm_inputs  # noqa: E402
from analysis.statistics import FDR_LABEL, FDR_THRESHOLD  # noqa: E402
from paths import mmrm_dir  # noqa: E402



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def baseline_npx(assays: list[str]) -> np.ndarray:
    """Participants x ``assays`` week-0 NPX of the MMRM input.

    Args:
        assays: Olink assay ids, the columns of the result.

    Returns:
        The NPX matrix, participants sorted by ``USUBJID``, NaN where missing.

    Raises:
        ValueError: When a participant has two baseline values for one assay.
    """
    inputs = prepare_mmrm_inputs(PLATFORM)
    baseline = (
        inputs.data.lazy()
        .filter(
            pl.col("VISITNUM") == BASELINE_VISITNUM, pl.col(inputs.protein_id_col).is_in(assays)
        )
        .select(
            "USUBJID",
            pl.col(inputs.protein_id_col).alias("assay"),
            pl.col(inputs.response_var).alias("value"),
        )
        .collect()
    )
    if baseline.select("USUBJID", "assay").is_duplicated().any():
        raise ValueError("a participant has two baseline values for one assay")
    wide = baseline.pivot(on="assay", index="USUBJID", values="value").sort("USUBJID")
    return wide.select(assays).to_numpy().astype(float)


def main(argv: list[str] | None = None) -> int:
    """Write the reversal table of one week.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        0.

    Raises:
        ValueError: From ``baseline_npx``, ``disease_signatures`` or ``reversal_table`` on
            inconsistent inputs.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--week", type=int, default=LATE_WEEK, choices=sorted(VISIT_LABELS.values()))
    args = ap.parse_args(argv)

    stats = contrast_statistics(mmrm_dir(PLATFORM))[f"{ACROSS_CONTRAST}@{args.week}"]
    mapping = load_assay_genes(PLATFORM, ANNOTATION_PATH)
    assays, assay_genes, z = assay_level_z(stats, mapping, EXCLUDED_ASSAYS[PLATFORM])
    alias = ensg_alias(read_annotation(ANNOTATION_PATH), read_uniprot_xref())
    signatures = disease_signatures(read_disease_de(), alias, assays, assay_genes)
    logger.info(f"{len(assays)} assays; {len(signatures)} adult disease signatures")
    baseline = baseline_assay_matrix(baseline_npx(assays))
    logger.info("baseline assay matrix ready")
    table = reversal_table(signatures, z, baseline)
    out = output_path(args.week)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    significant = table.filter(pl.col("fdr") < FDR_THRESHOLD).sort("fdr")
    logger.info(
        f"wrote {out} ({table.height} diseases); FDR < {FDR_LABEL}: "
        f"{significant.select('Disease', 'fdr').rows()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
