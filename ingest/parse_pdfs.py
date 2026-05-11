"""Parse arXiv PDFs into sections using PyMuPDF."""

import logging
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _median_font_size(page: fitz.Page) -> float:
    """Return the median font size of all text spans on the page."""
    sizes = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0) > 0:
                    sizes.append(span["size"])
    if not sizes:
        return 12.0
    return statistics.median(sizes)


def _is_section_header(text: str, font_size: float, median_size: float) -> bool:
    """Heuristic: short, no trailing period, font >= 1.2× median."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) >= 80:
        return False
    if stripped.endswith("."):
        return False
    return font_size >= median_size * 1.2


def parse_pdf(pdf_path: str) -> list[dict]:
    """Parse a PDF into a list of sections with metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys: section, text, page_start, page_end.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Could not open %s: %s", pdf_path, exc)
        return []

    if len(doc) == 0:
        return []

    sections: list[dict] = []
    current_section = "Introduction"
    current_lines: list[str] = []
    current_page_start = 1

    for page_num, page in enumerate(doc, start=1):
        median_size = _median_font_size(page)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                max_span_size = max(s.get("size", 0) for s in spans)
                if _is_section_header(line_text, max_span_size, median_size):
                    if current_lines:
                        raw = "\n".join(current_lines)
                        cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()
                        if cleaned:
                            sections.append({
                                "section": current_section,
                                "text": cleaned,
                                "page_start": current_page_start,
                                "page_end": page_num,
                            })
                    current_section = line_text
                    current_lines = []
                    current_page_start = page_num
                else:
                    current_lines.append(line_text)

    # Flush last section
    if current_lines:
        raw = "\n".join(current_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()
        if cleaned:
            sections.append({
                "section": current_section,
                "text": cleaned,
                "page_start": current_page_start,
                "page_end": len(doc),
            })

    return sections


def parse_all_pdfs(pdf_dir: str) -> dict[str, list[dict]]:
    """Parse all PDFs in a directory, keyed by arXiv ID.

    Args:
        pdf_dir: Directory containing PDF files named {arxiv_id}.pdf.

    Returns:
        Dict mapping arxiv_id to list of section dicts.
    """
    result: dict[str, list[dict]] = {}
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        arxiv_id = pdf_path.stem
        sections = parse_pdf(str(pdf_path))
        result[arxiv_id] = sections
        logger.info("Parsed %s: %d sections", arxiv_id, len(sections))
    return result


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Parse PDFs into sections")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--output", default="data/parsed_sections.json")
    args = parser.parse_args()

    parsed = parse_all_pdfs(args.pdf_dir)
    total_sections = sum(len(s) for s in parsed.values())
    print(f"Parsed {len(parsed)} papers, {total_sections} total sections")
    with open(args.output, "w") as f:
        json.dump(parsed, f, indent=2)
