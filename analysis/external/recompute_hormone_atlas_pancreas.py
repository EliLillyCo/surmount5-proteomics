"""Recompute per-cell-type GIPR / GLP1R stats from the Hormone Cell Atlas.

The Sanger pre-published Table S6D (extracted in
``data/external/sanger_s6d_gipr_glp1r.parquet``) only exposes hormone-receiving
cell types where a gene passed their inclusion threshold, so GLP1R rows are
mostly absent from cell types where GLP1R is genuinely low rather than truly
missing. For our gamma-vs-beta GIPR-vs-GLP1R figure we want true zeros, so we
recompute from the published h5ad.

Source: Pancreas_annotated.h5ad (~911 MB)
  https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/Pancreas_annotated.h5ad
  Fei, Huang-Doran et al., Science 2026 (doi:10.1126/science.aeb2672)

If the h5ad isn't already on disk under ``data/external/hormone_atlas/``,
this script fetches it via ``requests`` (streamed GET, ~30s on a fast link). The
download is gitignored; the resulting parquet is small enough to commit.

Output: data/external/hormone_atlas_pancreas_gpcr.parquet
  one row per (celltype_level1, gene) with mean log1p-normalized expression,
  fraction of cells with non-zero counts, and total cell count.

X in this h5ad is log1p-normalized counts (see uns['log1p']) — per-cell UMI
counts scaled to a constant cell-total then natural-log transformed via
log(1 + x). "% expressing" = (X > 0).mean() * 100 per cell-type.

Run
---
This is a one-off prep step, not part of the standard figure pipeline. The
output parquet is gitignored (regenerable); supp_fig15 reads it directly. Re-run
this script only if the source h5ad changes.

Requires ``anndata`` (not in the manuscript lockfile to avoid pulling pandas
back to <3.0):

    pip install anndata
    pixi run python analysis/external/recompute_hormone_atlas_pancreas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import requests
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parents[2]
H5AD_URL = (
    "https://cellgeni.cog.sanger.ac.uk/hormonecellatlas/download/"
    "Pancreas_annotated.h5ad"
)
H5AD_PATH = ROOT / "data" / "external" / "hormone_atlas" / "Pancreas_annotated.h5ad"
OUTPUT_PATH = ROOT / "data" / "external" / "hormone_atlas_pancreas_gpcr.parquet"

GENES = ("GIPR", "GLP1R")
CELLTYPE_COL = "celltype_level1"  # broad endocrine + exocrine + stromal labels


def _ensure_h5ad() -> None:
    """Download the pancreas h5ad if it isn't already on disk."""
    if H5AD_PATH.exists():
        return
    H5AD_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {H5AD_URL} → {H5AD_PATH} (~911 MB)…", file=sys.stderr)
    with requests.get(H5AD_URL, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        with open(H5AD_PATH, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)
    print(f"  done: {H5AD_PATH.stat().st_size / 1e6:,.0f} MB", file=sys.stderr)


def main() -> None:
    _ensure_h5ad()
    print(f"Reading {H5AD_PATH.name}…", file=sys.stderr)
    a = ad.read_h5ad(H5AD_PATH)  # in-memory; ~1 GB
    print(f"  shape: {a.shape}", file=sys.stderr)
    assert "log1p" in a.uns, "expected log1p-normalized X (uns['log1p'])"

    gene_idx = {g: a.var.index.get_loc(g) for g in GENES}

    # Pull just the two gene columns into dense arrays — 122k × 2 = trivial
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
