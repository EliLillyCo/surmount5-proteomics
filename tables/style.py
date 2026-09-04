"""Shared Excel styling for Nature Medicine supplementary tables.

Layout per sheet:
  Row 1: Table title (bold, e.g. "Table S1: TZP vs SEMA...")
  Row 2: Blank separator
  Row 3: Column headers (bold, light gray fill, top+bottom border)
  Row 4+: Data
  Last data row: bottom border

Abbreviations are listed centrally on the Contents sheet (TOC).

Styling:
- Default font/size
- No vertical gridlines, no cell shading, no autofilter
- Freeze row 3
- Number formats: scientific for P/FDR, 3 decimal for FC/PCBL
- Default column widths
"""

from __future__ import annotations

from openpyxl.styles import Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

FONT_HEADER = Font(bold=True)
FONT_TITLE = Font(bold=True)
FONT_FOOTNOTE = Font(italic=True)

THIN_BORDER = Side(style="thin", color="000000")
BORDER_HEADER = Border(top=THIN_BORDER, bottom=THIN_BORDER)
BORDER_LAST_ROW = Border(bottom=THIN_BORDER)

HEADER_FILL = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

DECIMAL3_EXACT = {
    "Spearman_Rho",
    "Prob_Concordant",
    "Prop_Mediated",
    "ACME",
    "ADE",
    "Total",
    "SE_Wk24vsWk72",
}


def _get_number_format(col_name: str) -> str | None:
    """Determine Excel number format based on column name patterns."""
    # Scientific notation: p-values and FDR columns
    if (
        "P-value" in col_name
        or "FDR" in col_name
        or col_name == "fdr_q"
        or col_name == "Spearman_P"
    ):
        return "0.00E+00"
    # 3 decimals: effect sizes, correlations, CIs, proportions
    if (
        col_name.startswith("log2FC")
        or col_name.startswith("Detection_Rate_")
        or "_CI_" in col_name
        or col_name in DECIMAL3_EXACT
    ):
        return "0.000"
    # 2 decimals: SE of percent-change estimates
    if col_name.startswith("SE_"):
        return "0.00"
    # 1 decimal: percent change from baseline, CV
    if col_name.startswith("PCBL") or col_name == "Intra_CV":
        return "0.0"
    return None


def style_sheet(ws: Worksheet, title: str, footnote: str | None = None) -> None:
    """Apply Nature Medicine styling to a worksheet.

    If ``footnote`` is provided, it is rendered as an italicized note two rows
    below the data block (used for per-sheet conventions like Table-1 notation).
    """
    n_cols = ws.max_column
    n_rows = ws.max_row

    # Insert 2 rows at top: title + blank separator
    ws.insert_rows(1, 2)
    n_rows += 2

    # Row 1: title (no merge — avoids border artifacts)
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = title
    title_cell.font = FONT_TITLE

    # Row 3: headers
    header_row = 3
    data_start = 4

    ws.freeze_panes = f"A{data_start}"

    col_names = [ws.cell(row=header_row, column=i).value for i in range(1, n_cols + 1)]

    for col_idx in range(1, n_cols + 1):
        col_name = col_names[col_idx - 1]

        header_cell = ws.cell(row=header_row, column=col_idx)
        header_cell.font = FONT_HEADER
        header_cell.border = BORDER_HEADER
        header_cell.fill = HEADER_FILL

        num_fmt = _get_number_format(col_name) if col_name else None

        if num_fmt:
            for row_idx in range(data_start, n_rows + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None:
                    cell.number_format = num_fmt

    # Bottom border on last data row
    for col_idx in range(1, n_cols + 1):
        ws.cell(row=n_rows, column=col_idx).border = BORDER_LAST_ROW

    # Bold "section header" rows: first column populated, all other columns blank/None.
    # Used by the baseline-characteristics table to mark sub-section dividers.
    for row_idx in range(data_start, n_rows + 1):
        first = ws.cell(row=row_idx, column=1).value
        if not first:
            continue
        others = [ws.cell(row=row_idx, column=c).value for c in range(2, n_cols + 1)]
        if all(v in (None, "") for v in others):
            ws.cell(row=row_idx, column=1).font = FONT_HEADER

    if footnote:
        ws.cell(row=n_rows + 2, column=1, value=footnote).font = FONT_FOOTNOTE

    # Size column A to fit row labels (rows 3..n_rows). Skip the title row (which
    # spans the whole table) and the footnote (which can also be wide).
    max_a = max(
        len(str(ws.cell(r, 1).value or "")) for r in range(header_row, n_rows + 1)
    )
    ws.column_dimensions[get_column_letter(1)].width = max_a + 2


def create_toc(
    ws: Worksheet,
    entries: list[tuple[str, str]],
    abbreviations: list[tuple[str, str]] | None = None,
) -> None:
    """Create a Table of Contents sheet with hyperlinks and an abbreviations block.

    entries: list of (sheet_tab_name, table_title)
    abbreviations: optional list of (abbreviation, definition) rendered as a
    styled mini-table below the TOC (same header styling).
    """
    ws.cell(row=1, column=1, value="Table of Contents")
    ws.cell(row=1, column=1).font = FONT_TITLE

    ws.cell(row=3, column=1, value="Table")
    ws.cell(row=3, column=2, value="Description")
    for col_idx in range(1, 3):
        cell = ws.cell(row=3, column=col_idx)
        cell.font = FONT_HEADER
        cell.border = BORDER_HEADER
        cell.fill = HEADER_FILL

    for i, (tab, title) in enumerate(entries, start=4):
        cell_tab = ws.cell(row=i, column=1, value=tab)
        cell_tab.hyperlink = f"#'{tab}'!A1"
        cell_tab.font = Font(underline="single", color="0563C1")
        ws.cell(row=i, column=2, value=title)

    last_row = 3 + len(entries)
    for col_idx in range(1, 3):
        ws.cell(row=last_row, column=col_idx).border = BORDER_LAST_ROW

    if abbreviations:
        title_row = last_row + 2
        ws.cell(row=title_row, column=1, value="Abbreviations").font = FONT_TITLE

        header_row = title_row + 2
        ws.cell(row=header_row, column=1, value="Abbreviation")
        ws.cell(row=header_row, column=2, value="Definition")
        for col_idx in range(1, 3):
            cell = ws.cell(row=header_row, column=col_idx)
            cell.font = FONT_HEADER
            cell.border = BORDER_HEADER
            cell.fill = HEADER_FILL

        for i, (abbr, defn) in enumerate(abbreviations, start=header_row + 1):
            ws.cell(row=i, column=1, value=abbr)
            ws.cell(row=i, column=2, value=defn)

        abbr_last_row = header_row + len(abbreviations)
        for col_idx in range(1, 3):
            ws.cell(row=abbr_last_row, column=col_idx).border = BORDER_LAST_ROW

    max_a = max(len(str(ws.cell(r, 1).value or "")) for r in range(1, ws.max_row + 1))
    max_b = max(len(str(ws.cell(r, 2).value or "")) for r in range(1, ws.max_row + 1))
    ws.column_dimensions["A"].width = max_a + 2
    ws.column_dimensions["B"].width = max_b + 2
