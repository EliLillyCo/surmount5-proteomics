# Panel reference files

This directory holds vendor panel manifests used by
`analysis/tools/uniprot_map.py` to build the cross-platform protein ID mapping.

These files are not distributed with the repository. Download them from the
vendor websites and place them here before running
`uv run python analysis/scripts/build_uniprot_map.py`.

## Olink Explore HT

- File: `olink_explore_ht.tsv`
- Source: https://olink.com/products/olink-explore-ht
- Required columns: `OlinkID`, `Gene_Symbol`, `UniProt_ID`

## SomaScan 11K

- File: `somascan_11k.tsv`
- Source: https://menu.somalogic.com
- Required columns: `SeqId`, `Gene_Symbol`, `UniProt_ID`
