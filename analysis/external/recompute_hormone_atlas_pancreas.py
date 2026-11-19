"""Recompute per-cell-type GIPR / GLP1R stats from the Hormone Cell Atlas.

Per-cell-type GIPR and GLP1R expression recomputed from the Hormone Cell Atlas pancreas
h5ad; every cell type is kept, zeros included.

Source: Pancreas_annotated.h5ad (~911 MB)
  https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/Pancreas_annotated.h5ad
  Fei, Huang-Doran et al., Science 2026 (doi:10.1126/science.aeb2672)

If the h5ad isn't already on disk under ``data/external/hormone_atlas/``,
this script fetches it. The file is SHA-256 pinned (``H5AD_SHA256``) and verified on
every use, a cached copy included. The download is gitignored.

Output: data/external/hormone_atlas_pancreas_gpcr.parquet
  one row per (celltype_level1, gene) with mean log1p-normalized expression,
  fraction of cells with non-zero counts, and total cell count.

X in this h5ad is log1p-normalized counts (see uns['log1p']) — per-cell UMI
counts scaled to a constant cell-total then natural-log transformed via
log(1 + x). "% expressing" = (X > 0).mean() * 100 per cell-type.

Run
---
Stage ``hormone_atlas_pancreas_gpcr``; ``figures/supp_fig_gipr_glp1r_celltype.py`` reads the
output parquet.

``anndata`` is part of the manuscript pixi environment (conda-forge), so:

    pixi run python analysis/external/recompute_hormone_atlas_pancreas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sps

from analysis.external.pinned import fetch_pinned

ROOT = Path(__file__).resolve().parents[2]


H5AD_URL = (
    "https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/"
    "Pancreas_annotated.h5ad"
)
# SHA-256 of Pancreas_annotated.h5ad, verified before it is read.
H5AD_SHA256 = "a1e638e8a1ba7ce2373d43f94c8e5d0ae89f083de9e6780d6f6146a2cfd278e3"
H5AD_PATH = ROOT / "data" / "external" / "hormone_atlas" / "Pancreas_annotated.h5ad"
OUTPUT_PATH = ROOT / "data" / "external" / "hormone_atlas_pancreas_gpcr.parquet"

GENES = ("GIPR", "GLP1R")
CELLTYPE_COL = "celltype_level1"  # broad endocrine + exocrine + stromal labels


def fetch_h5ad() -> Path:
    """The checksum-verified pancreas h5ad, downloaded first when absent (~911 MB).

    Raises:
        RuntimeError: When the cached or downloaded file does not match ``H5AD_SHA256``.
    """
    return fetch_pinned(H5AD_URL, H5AD_PATH, H5AD_SHA256, timeout=600)


def main() -> None:
    h5ad = fetch_h5ad()
    print(f"Reading {h5ad.name}…", file=sys.stderr)
    a = ad.read_h5ad(h5ad)  # in-memory; ~1 GB
    print(f"  shape: {a.shape}", file=sys.stderr)
    assert "log1p" in a.uns, "expected log1p-normalized X (uns['log1p'])"

    gene_idx = {g: a.var.index.get_loc(g) for g in GENES}

    # Extract the selected gene columns into dense arrays
    X = a.X
    if sps.issparse(X):
        sub = X[:, [gene_idx[g] for g in GENES]].toarray()
    else:
        sub = np.asarray(X[:, [gene_idx[g] for g in GENES]])

    obs = a.obs[[CELLTYPE_COL]].copy()
    obs = obs.reset_index(drop=True)  # groupby below needs integer positions
    rows = []
    for cell_type, idx in obs.groupby(CELLTYPE_COL, observed=True).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        block = sub[idx_arr]  # n × 2
        n = len(idx_arr)
        for j, gene in enumerate(GENES):
            v = block[:, j]
            rows.append(
                dict(
                    cell_type=str(cell_type),
                    gene=gene,
                    mean_expr=float(v.mean()),
                    pct_expressing=float((v > 0).mean() * 100.0),
                    n_cells=int(n),
                )
            )
    out = pd.DataFrame(rows)
    pl.from_pandas(out).write_parquet(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}  rows={len(out)}", file=sys.stderr)
    print()
    print(out.pivot(index=["cell_type", "n_cells"], columns="gene", values=["pct_expressing", "mean_expr"]).to_string())


if __name__ == "__main__":
    main()
