"""Tests for agent/tools.py."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.tools import arxiv_search


def _make_author(name: str) -> MagicMock:
    a = MagicMock()
    a.name = name
    return a


def _make_paper(title: str, abstract: str = "Short abstract.") -> MagicMock:
    paper = MagicMock()
    paper.title = title
    paper.authors = [_make_author("Author A"), _make_author("Author B")]
    paper.published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    paper.summary = abstract
    paper.entry_id = f"https://arxiv.org/abs/2401.00001"
    return paper


@patch("agent.tools.arxiv.Client")
def test_arxiv_search_returns_formatted_string(mock_client_cls):
    papers = [_make_paper("Paper One"), _make_paper("Paper Two")]
    mock_client_cls.return_value.results.return_value = iter(papers)

    result = arxiv_search.invoke({"query": "transformers"})
    assert "[1]" in result
    assert "[2]" in result
    assert "Paper One" in result
    assert "Paper Two" in result


@patch("agent.tools.arxiv.Client")
def test_arxiv_search_empty_results(mock_client_cls):
    mock_client_cls.return_value.results.return_value = iter([])

    result = arxiv_search.invoke({"query": "transformers"})
    assert "No results found" in result


@patch("agent.tools.arxiv.Client")
def test_arxiv_search_api_error(mock_client_cls):
    mock_client_cls.side_effect = Exception("connection refused")

    result = arxiv_search.invoke({"query": "transformers"})
    assert result.startswith("[ERROR]")


@patch("agent.tools.arxiv.Client")
def test_arxiv_search_max_results_respected(mock_client_cls):
    papers = [_make_paper(f"Paper {i}") for i in range(10)]
    # arxiv.Search handles max_results — mock returns only 3
    mock_client_cls.return_value.results.return_value = iter(papers[:3])

    result = arxiv_search.invoke({"query": "transformers", "max_results": 3})
    assert "[4]" not in result
    assert "[3]" in result


@patch("agent.tools.arxiv.Client")
def test_arxiv_search_abstract_truncated(mock_client_cls):
    long_abstract = "A" * 500
    papers = [_make_paper("Long Abstract Paper", abstract=long_abstract)]
    mock_client_cls.return_value.results.return_value = iter(papers)

    result = arxiv_search.invoke({"query": "test"})
    assert "..." in result
    # Verify the abstract in the result is not the full 500 chars
    assert long_abstract not in result
