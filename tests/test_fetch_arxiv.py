"""Tests for ingest/fetch_arxiv.py."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingest.fetch_arxiv import fetch_papers


def _make_author(name: str) -> MagicMock:
    a = MagicMock()
    a.name = name
    return a


def _make_entry(arxiv_id: str, days_ago: int = 10) -> MagicMock:
    entry = MagicMock()
    entry.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
    entry.title = f"Paper {arxiv_id}"
    entry.authors = [_make_author("Author A")]
    entry.summary = "Abstract text."
    entry.published = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    entry.download_pdf = MagicMock()
    return entry


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_returns_metadata_list(mock_sleep, mock_client_cls, tmp_path):
    entries = [_make_entry("2401.00001"), _make_entry("2401.00002"), _make_entry("2401.00003")]
    mock_client_cls.return_value.results.return_value = iter(entries)

    results = fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    assert len(results) == 3
    for r in results:
        assert "arxiv_id" in r
        assert "title" in r
        assert "authors" in r
        assert "published" in r


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_filters_old_papers(mock_sleep, mock_client_cls, tmp_path):
    entries = [
        _make_entry("2401.00001", days_ago=10),
        _make_entry("2401.00002", days_ago=100),  # older than 90 days
        _make_entry("2401.00003", days_ago=5),
        _make_entry("2401.00004", days_ago=95),   # older than 90 days
        _make_entry("2401.00005", days_ago=30),
    ]
    mock_client_cls.return_value.results.return_value = iter(entries)

    results = fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    assert len(results) == 3


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_respects_rate_limit(mock_sleep, mock_client_cls, tmp_path):
    entries = [_make_entry("2401.00001"), _make_entry("2401.00002")]
    mock_client_cls.return_value.results.return_value = iter(entries)

    fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    calls = [c for c in mock_sleep.call_args_list if c.args and c.args[0] == 3]
    assert len(calls) == 2


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_handles_download_failure(mock_sleep, mock_client_cls, tmp_path):
    ok_entry = _make_entry("2401.00001")
    bad_entry = _make_entry("2401.00002")
    bad_entry.download_pdf.side_effect = Exception("network error")
    ok_entry2 = _make_entry("2401.00003")

    mock_client_cls.return_value.results.return_value = iter([ok_entry, bad_entry, ok_entry2])

    results = fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    arxiv_ids = [r["arxiv_id"] for r in results]
    assert "2401.00001" in arxiv_ids
    assert "2401.00003" in arxiv_ids
    assert "2401.00002" not in arxiv_ids


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_saves_metadata_json(mock_sleep, mock_client_cls, tmp_path):
    entries = [_make_entry("2401.00001"), _make_entry("2401.00002")]
    mock_client_cls.return_value.results.return_value = iter(entries)

    fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    meta_file = tmp_path / "metadata.json"
    assert meta_file.exists()
    with open(meta_file) as f:
        meta = json.load(f)
    assert "2401.00001" in meta
    assert "title" in meta["2401.00001"]


@patch("ingest.fetch_arxiv.arxiv.Client")
@patch("ingest.fetch_arxiv.time.sleep")
def test_fetch_zero_results(mock_sleep, mock_client_cls, tmp_path):
    mock_client_cls.return_value.results.return_value = iter([])

    results = fetch_papers(max_results=10, days_back=90, output_dir=str(tmp_path))

    assert results == []
