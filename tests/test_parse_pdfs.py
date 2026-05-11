"""Tests for ingest/parse_pdfs.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from ingest.parse_pdfs import _is_section_header, parse_all_pdfs, parse_pdf


def _make_span(text: str, size: float) -> dict:
    return {"text": text, "size": size}


def _make_line(spans: list[dict]) -> dict:
    return {"spans": spans}


def _make_block(lines: list[dict]) -> dict:
    return {"type": 0, "lines": lines}


def _make_page_dict(blocks: list[dict]) -> dict:
    return {"blocks": blocks}


# --- Unit tests for the heuristic ---

def test_is_section_header_true():
    assert _is_section_header("Related Work", 14.0, 10.0) is True


def test_is_section_header_too_long():
    long_text = "A" * 80
    assert _is_section_header(long_text, 14.0, 10.0) is False


def test_is_section_header_ends_with_period():
    assert _is_section_header("Introduction.", 14.0, 10.0) is False


def test_is_section_header_small_font():
    assert _is_section_header("Short Line", 9.0, 10.0) is False


def test_is_section_header_empty():
    assert _is_section_header("", 14.0, 10.0) is False


# --- Tests for parse_pdf ---

def test_parse_returns_sections(tmp_path):
    """A minimal real PDF with at least one section should return a non-empty list."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Introduction", fontsize=16)
    page.insert_text((72, 130), "This is the introduction text that is long enough.")
    doc.save(str(pdf_path))
    doc.close()

    result = parse_pdf(str(pdf_path))
    assert isinstance(result, list)
    for section in result:
        assert "section" in section
        assert "text" in section
        assert "page_start" in section
        assert "page_end" in section


def test_parse_handles_corrupted_pdf(tmp_path):
    bad_file = tmp_path / "bad.pdf"
    bad_file.write_bytes(b"not a pdf at all")

    result = parse_pdf(str(bad_file))
    assert result == []


def test_parse_empty_pdf():
    """A PDF with zero pages should return an empty list."""
    with patch("ingest.parse_pdfs.fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_open.return_value = mock_doc
        result = parse_pdf("fake_empty.pdf")
    assert result == []


def test_parse_strips_excess_whitespace(tmp_path):
    """Sections with 3+ consecutive newlines should be collapsed to at most 2."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Insert multiple lines that will be gathered into one section
    for y in range(100, 500, 20):
        page.insert_text((72, y), "Some content line here.")
    doc.save(str(pdf_path))
    doc.close()

    result = parse_pdf(str(pdf_path))
    for section in result:
        assert "\n\n\n" not in section["text"]


def test_parse_all_pdfs_aggregates(tmp_path):
    for arxiv_id in ["2401.00001", "2401.00002"]:
        pdf_path = tmp_path / f"{arxiv_id}.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), f"Paper {arxiv_id} content here.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

    result = parse_all_pdfs(str(tmp_path))
    assert "2401.00001" in result
    assert "2401.00002" in result
    assert isinstance(result["2401.00001"], list)
    assert isinstance(result["2401.00002"], list)
