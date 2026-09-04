"""Output helpers: save combined results and print summary tables."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_combined(
    results_list: list[pd.DataFrame],
    method_name: str,
    sig_col: str,
    output_dir: Path,
) -> None:
    """Concatenate per-library results and write a single CSV."""
    if not results_list:
        return
    combined = pd.concat(results_list, ignore_index=True)
    cols = ["Gene_Set_Database"] + [
        c for c in combined.columns if c != "Gene_Set_Database"
    ]
    combined = combined[cols].sort_values(sig_col)
    path = output_dir / f"{method_name}_combined.csv"
    combined.to_csv(path, index=False)
    n_sig = (combined[sig_col] < 0.05).sum()
    logger.info(
        f"{method_name.upper()}: {len(combined)} pathways ({n_sig} sig) -> {path}"
    )


def print_summary_table(
    gsea_res: list[pd.DataFrame],
    ora_res: list[pd.DataFrame],
    cam_res: list[pd.DataFrame],
) -> None:
    """Print a per-library summary of tested/significant pathways for each method."""
    stats: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    method_info = [
        ("GSEA", gsea_res, "FDR q-val"),
        ("ORA", ora_res, "Adjusted P-value"),
        ("cameraPR", cam_res, "FDR q-val"),
    ]
    active_methods: list[str] = []
    for method_name, res_list, sig_col in method_info:
        if not res_list:
            continue
        active_methods.append(method_name)
        for df in res_list:
            lib = df["Gene_Set_Database"].iloc[0]
            tested = len(df)
            sig = int((df[sig_col].astype(float) < 0.05).sum())
            stats[lib][method_name] = (tested, sig)

    if not active_methods:
        return

    libs = list(stats.keys())
    lib_w = max(len("Library"), max(len(l) for l in libs))
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
        total_cell = f"{lib_sig_total}/{lib_tested_total}"
        row += f"  {total_cell:>{total_w}}"
        print(row)
    print(f"{sep}\n")
