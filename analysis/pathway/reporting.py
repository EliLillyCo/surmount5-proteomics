"""Output helpers: save combined results and print summary tables."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import polars as pl

from analysis.mmrm_outputs import pathway_combined_name
from analysis.statistics import FDR_THRESHOLD

from .methods import CAMERA_COLUMNS

logger = logging.getLogger(__name__)

# The offline gseapy.enrichr result columns returned by methods.run_ora.
ORA_COLUMNS = [
    "Gene_set",
    "Term",
    "Overlap",
    "P-value",
    "Adjusted P-value",
    "Odds Ratio",
    "Combined Score",
    "Genes",
]


def save_combined(
    results_list: list[pl.DataFrame],
    method_name: str,
    sig_col: str,
    output_dir: Path,
) -> None:
    """Write per-library results into one CSV, including a header when there are no rows."""
    path = output_dir / pathway_combined_name(method_name)
    if not results_list:
        method_columns = {"ora": ORA_COLUMNS, "camera": CAMERA_COLUMNS}[method_name]
        pl.DataFrame(schema=["Gene_Set_Database", *method_columns]).write_csv(path)
        logger.info(f"{method_name.upper()}: 0 pathways (0 sig) -> {path}")
        return
    combined = pl.concat(results_list, how="diagonal_relaxed")
    cols = ["Gene_Set_Database"] + [c for c in combined.columns if c != "Gene_Set_Database"]
    combined = combined.select(cols).sort([sig_col, "Gene_Set_Database", "Term"])
    path = output_dir / pathway_combined_name(method_name)
    combined.write_csv(path)
    n_sig = (combined[sig_col] < FDR_THRESHOLD).sum()
    logger.info(f"{method_name.upper()}: {combined.height} pathways ({n_sig} sig) -> {path}")


def print_summary_table(ora_res: list[pl.DataFrame], cam_res: list[pl.DataFrame]) -> None:
    """Print a per-library summary of tested/significant pathways for each method."""
    stats: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    method_info = [("ORA", ora_res, "Adjusted P-value"), ("cameraPR", cam_res, "FDR q-val")]
    active_methods: list[str] = []
    for method_name, res_list, sig_col in method_info:
        if not res_list:
            continue
        active_methods.append(method_name)
        for df in res_list:
            lib = df["Gene_Set_Database"][0]
            stats[lib][method_name] = (df.height, int((df[sig_col].cast(pl.Float64) < FDR_THRESHOLD).sum()))

    if not active_methods:
        return

    libs = list(stats)
    lib_w = max(len("Library"), max(len(lib) for lib in libs))
    col_w = max(12, max(len(m) for m in active_methods) + 2)
    total_w = max(col_w, len("TOTAL") + 2)

    header = f"{'Library':<{lib_w}}"
    for m in active_methods:
        header += f"  {m:>{col_w}}"
    header += f"  {'TOTAL':>{total_w}}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("PATHWAY ANALYSIS SUMMARY (sig / tested)")
    print(sep)
    print(header)
    print(sep)
    for lib in libs:
        row = f"{lib:<{lib_w}}"
        lib_sig_total = 0
        lib_tested_total = 0
        for m in active_methods:
            if m in stats[lib]:
                tested, sig = stats[lib][m]
                cell = f"{sig}/{tested}"
                lib_sig_total += sig
                lib_tested_total += tested
            else:
                cell = "-"
            row += f"  {cell:>{col_w}}"
        row += f"  {f'{lib_sig_total}/{lib_tested_total}':>{total_w}}"
        print(row)
    print(f"{sep}\n")
