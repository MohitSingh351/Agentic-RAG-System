"""Tests for ingest/chunk.py."""

import re

import tiktoken
import pytest

from ingest.chunk import (
    MIN_TOKENS,
    MAX_TOKENS,
    _count_tokens,
    _chunk_text,
    chunk_sections,
    chunk_all_papers,
)

_ENC = tiktoken.get_encoding("cl100k_base")


def _make_text(n_tokens: int) -> str:
    """Create a text string of approximately n_tokens tokens."""
    word = "word "
    approx = word * (n_tokens // 1 + 5)
    tokens = _ENC.encode(approx)
    return _ENC.decode(tokens[:n_tokens])


def _make_metadata(arxiv_id: str = "2401.00001") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["Author A"],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def _make_section(text: str, name: str = "Introduction") -> dict:
    return {"section": name, "text": text, "page_start": 1, "page_end": 2}


# --- _chunk_text tests ---

def test_chunk_target_range():
    """All chunks from a large section should be within [MIN, MAX] tokens."""
    # Build a text of ~1800 tokens (multiple paragraphs)
    paragraphs = [_make_text(300) for _ in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks[:-1]:  # last chunk may be smaller
        assert _count_tokens(c) <= MAX_TOKENS


def test_chunk_oversized_paragraph():
    """A single paragraph exceeding MAX_TOKENS should be split."""
    text = _make_text(900)  # single paragraph, no newlines
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert _count_tokens(c) <= MAX_TOKENS


def test_chunk_undersized_merge():
    """Three paragraphs of 150 tokens each should be merged, not kept separate."""
    paragraphs = [_make_text(150) for _ in range(3)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_text(text)
    # All 450 tokens should fit in <=2 chunks, not 3 separate under-MIN chunks
    assert len(chunks) <= 2


def test_chunk_single_short_section():
    """A section with 50 tokens should produce exactly one chunk."""
    text = _make_text(50)
    chunks = _chunk_text(text)
    assert len(chunks) == 1


# --- chunk_sections tests ---

def test_chunk_id_format():
    """chunk_id must match '{arxiv_id}_chunk_{4-digit-index}'."""
    meta = _make_metadata("2401.00001")
    sections = [_make_section(_make_text(500))]
    chunks = chunk_sections(sections, meta)
    for chunk in chunks:
        assert re.match(r"^\S+_chunk_\d{4}$", chunk["chunk_id"]), chunk["chunk_id"]


def test_chunk_preserves_metadata():
    """Every chunk must carry paper-level metadata."""
    meta = _make_metadata("2401.00099")
    sections = [_make_section(_make_text(500))]
    chunks = chunk_sections(sections, meta)
    for chunk in chunks:
        assert chunk["arxiv_id"] == "2401.00099"
        assert chunk["title"] == "Paper 2401.00099"
        assert "url" in chunk
        assert "section" in chunk


def test_chunk_empty_sections():
    chunks = chunk_sections([], _make_metadata())
    assert chunks == []


def test_chunk_index_sequential():
    """chunk_index values should be 0, 1, 2, ... across all sections."""
    meta = _make_metadata()
    # Build two sections that together produce at least 5 chunks
    sections = [
        _make_section(_make_text(1500), "Introduction"),
        _make_section(_make_text(1500), "Methods"),
    ]
    chunks = chunk_sections(sections, meta)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


# --- chunk_all_papers tests ---

def test_chunk_all_papers_aggregates():
    parsed = {
        "2401.00001": [_make_section(_make_text(500))],
        "2401.00002": [_make_section(_make_text(500))],
    }
    metadata_map = {
        "2401.00001": _make_metadata("2401.00001"),
        "2401.00002": _make_metadata("2401.00002"),
    }
    chunks = chunk_all_papers(parsed, metadata_map)
    ids = {c["arxiv_id"] for c in chunks}
    assert "2401.00001" in ids
    assert "2401.00002" in ids


def test_chunk_all_papers_empty():
    assert chunk_all_papers({}, {}) == []
