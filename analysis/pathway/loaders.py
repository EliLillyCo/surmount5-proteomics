"""Marker-to-gene mapping loaders for Olink and Somascan platforms."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, mapping_file: str, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{mapping_file} is missing required columns: {missing}")


def load_olink_mapping(mapping_file: str) -> dict[str, list[str]]:
    """Load OlinkID -> gene symbol mapping (1-to-many)."""
    df = pd.read_csv(mapping_file, sep="\t")
    _require_columns(df, mapping_file, ["OlinkID", "gene_symbol"])
    mapping: dict[str, list[str]] = {}
    for olink_id, group in df.groupby("OlinkID"):
        genes = [
            gene
            for gene in group["gene_symbol"].dropna().astype(str).unique().tolist()
            if gene
        ]
        mapping[olink_id] = genes
    multi = sum(1 for v in mapping.values() if len(v) > 1)
    logger.info(f"Olink mapping: {len(mapping)} IDs ({multi} multi-map)")
    return mapping


def load_somascan_mapping(mapping_file: str) -> dict[str, list[str]]:
    """Load SeqId -> gene symbol mapping (split by |)."""
    df = pd.read_csv(mapping_file, sep="\t")
    _require_columns(df, mapping_file, ["SeqId", "EntrezGeneSymbol"])
    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        seq_id = row["SeqId"]
        symbols = row.get("EntrezGeneSymbol", np.nan)
        if pd.notna(symbols):
            mapping[seq_id] = [g.strip() for g in str(symbols).split("|") if g.strip()]
        else:
            mapping[seq_id] = []
    multi = sum(1 for v in mapping.values() if len(v) > 1)
    logger.info(f"Somascan mapping: {len(mapping)} IDs ({multi} multi-map)")
    return mapping
