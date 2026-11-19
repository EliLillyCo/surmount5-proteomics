"""Journal-box checks, A4 fitting and page stacking for the figure bundles.

A bundle page is a figure's own page with a label strip stacked above it. A main figure must
fit the 183 x 247 mm box and an Extended Data figure the 180 x 170 mm box as drawn; one that
does not is reported and never scaled, because scaling would shrink its text below the 5 pt
floor. A supplementary figure larger than the A4 content box is scaled down to it
(``fit_to_a4_content``), never up.
"""

from __future__ import annotations

from collections.abc import Sequence

from pypdf import PageObject, Transformation

from ._theme import ED_MAX_H_IN, ED_MAX_W_IN, MAIN_MAX_H_IN, MAIN_MAX_W_IN, MM_PER_INCH

A4_WIDTH_PT = 210.0 / 25.4 * 72.0
A4_HEIGHT_PT = 297.0 / 25.4 * 72.0

# A4 page margins (inches). The side margin is 0.38 so the 7.49 in trajectory grids fit the
# 7.51 in content width at true size.
PAGE_SIDE_MARGIN_IN = 0.38
PAGE_TOP_MARGIN_IN = 0.50
PAGE_BOTTOM_MARGIN_IN = 0.50

# Slack when comparing a rendered PDF page to a journal box: the page is the figure's tight
# bounding box, so a layout drawn exactly to the limit can round over it.
FIT_TOLERANCE_PT = 0.5


def content_width_pt() -> float:
    """Return the A4 content width between the side margins.

    Returns:
        Width in points.
    """
    return A4_WIDTH_PT - 2 * PAGE_SIDE_MARGIN_IN * 72.0


def content_height_pt() -> float:
    """Return the A4 content height between the top and bottom margins.

    Returns:
        Height in points.
    """
    return A4_HEIGHT_PT - (PAGE_TOP_MARGIN_IN + PAGE_BOTTOM_MARGIN_IN) * 72.0


def _box_problem(width_pt: float, height_pt: float, max_w_in: float, max_h_in: float) -> str | None:
    """Say which dimensions of a figure page exceed a box, if any do.

    Args:
        width_pt: Figure page width in points.
        height_pt: Figure page height in points.
        max_w_in: Box width in inches.
        max_h_in: Box height in inches.

    Returns:
        A message naming each exceeded dimension in millimeters, or None when the figure
        fits within the width and height limits.
    """
    mm_per_pt = MM_PER_INCH / 72.0
    problems = []
    for name, size_pt, limit_in in (("width", width_pt, max_w_in), ("height", height_pt, max_h_in)):
        if size_pt > limit_in * 72.0 + FIT_TOLERANCE_PT:
            problems.append(
                f"{name} {size_pt * mm_per_pt:.1f} mm exceeds {limit_in * MM_PER_INCH:.0f} mm"
            )
    return "; ".join(problems) or None


def main_fit_problem(width_pt: float, height_pt: float) -> str | None:
    """Say why a main-figure page exceeds the journal box, if it does.

    Args:
        width_pt: Figure page width in points.
        height_pt: Figure page height in points.

    Returns:
        A message naming each exceeded dimension in millimeters, or None when it fits.
    """
    return _box_problem(width_pt, height_pt, MAIN_MAX_W_IN, MAIN_MAX_H_IN)


def extended_fit_problem(width_pt: float, height_pt: float) -> str | None:
    """Say why a figure page does not fit the Extended Data box, if it does not.

    Args:
        width_pt: Figure page width in points.
        height_pt: Figure page height in points.

    Returns:
        A message naming each exceeded dimension in millimeters, or None when it fits.
    """
    return _box_problem(width_pt, height_pt, ED_MAX_W_IN, ED_MAX_H_IN)


def fit_box(
    width_pt: float,
    height_pt: float,
    *,
    max_width_pt: float,
    max_height_pt: float,
) -> tuple[float, float, float]:
    """Scale a box down (never up) to fit within a maximum size.

    Args:
        width_pt: Box width in points.
        height_pt: Box height in points.
        max_width_pt: Largest allowed width in points.
        max_height_pt: Largest allowed height in points.

    Returns:
        (scaled width, scaled height, scale factor <= 1.0).

    Raises:
        ValueError: When either box dimension is not positive.
    """
    if width_pt <= 0 or height_pt <= 0:
        raise ValueError("Box dimensions must be positive.")
    scale = min(max_width_pt / width_pt, max_height_pt / height_pt, 1.0)
    return width_pt * scale, height_pt * scale, scale


def fit_to_a4_content(page: PageObject, name: str) -> PageObject:
    """Scale a figure page down (never up) to fit the A4 content box.

    Args:
        page: The figure's page; its mediabox must start at the origin.
        name: Figure name for the note printed when the page is scaled.

    Returns:
        ``page`` itself when it fits, else a new page of the scaled size holding it.
    """
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    scaled_w, scaled_h, scale = fit_box(
        width, height, max_width_pt=content_width_pt(), max_height_pt=content_height_pt()
    )
    if scale == 1.0:
        return page
    print(f"  [note] {name} scaled to {scale:.1%} to fit the A4 content box")
    out = PageObject.create_blank_page(width=scaled_w, height=scaled_h)
    out.merge_transformed_page(page, Transformation().scale(scale, scale))
    return out


def stack_pages_vertically(pages: Sequence[PageObject], bottom_first: bool = False) -> PageObject:
    """A new page with ``pages`` stacked top to bottom, each centered on the widest.

    Args:
        pages: Source pages in top-to-bottom order.
        bottom_first: Merge the lowest page first. The drawing is the same either way; the
            order only sets the page's content-stream order, which the callers keep fixed so
            their PDFs stay byte-identical.

    Returns:
        A page as wide as the widest source and as tall as all of them together.
    """
    widths = [float(p.mediabox.width) for p in pages]
    heights = [float(p.mediabox.height) for p in pages]
    page_w = max(widths)
    page = PageObject.create_blank_page(width=page_w, height=sum(heights))
    order = reversed(range(len(pages))) if bottom_first else range(len(pages))
    for i in order:
        shift = Transformation().translate(tx=(page_w - widths[i]) / 2, ty=sum(heights[i + 1 :]))
        page.merge_transformed_page(pages[i], shift)
    return page
