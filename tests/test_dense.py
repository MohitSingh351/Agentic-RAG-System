"""Tests for retrieval/dense.py."""

from unittest.mock import MagicMock, patch

import pytest

import retrieval.dense as dense_mod
from retrieval.dense import dense_retrieve


def _mock_collection(ids, documents, metadatas, distances):
    col = MagicMock()
    col.count.return_value = len(ids)
    col.query.return_value = {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    return col


def _mock_embed(embedding: list[float] | None = None):
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=embedding or [0.1] * 1536)]
    )
    return client


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_retrieve_returns_list(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    mock_col.return_value = _mock_collection(
        ["c1", "c2"],
        ["text1", "text2"],
        [{"title": "A"}, {"title": "B"}],
        [0.1, 0.3],
    )
    results = dense_retrieve("test query", top_k=5)
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "metadata" in r
        assert "score" in r


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_score_range(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    mock_col.return_value = _mock_collection(
        ["c1", "c2", "c3"],
        ["t1", "t2", "t3"],
        [{}, {}, {}],
        [0.0, 0.5, 1.0],
    )
    results = dense_retrieve("query")
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_sorted_by_score(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    mock_col.return_value = _mock_collection(
        ["c1", "c2", "c3"],
        ["t1", "t2", "t3"],
        [{}, {}, {}],
        [0.3, 0.1, 0.5],
    )
    results = dense_retrieve("query")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_empty_collection(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    empty_col = MagicMock()
    empty_col.count.return_value = 0
    mock_col.return_value = empty_col

    results = dense_retrieve("query")
    assert results == []
    empty_col.query.assert_not_called()


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_top_k_respected(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    mock_col.return_value = _mock_collection(
        ["c1", "c2", "c3", "c4", "c5"],
        ["t1", "t2", "t3", "t4", "t5"],
        [{} for _ in range(5)],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    )
    results = dense_retrieve("query", top_k=3)
    assert len(results) == 3


@patch("retrieval.dense._get_collection")
@patch("retrieval.dense._get_mistral")
def test_dense_empty_query(mock_openai, mock_col):
    mock_openai.return_value = _mock_embed()
    mock_col.return_value = _mock_collection(
        ["c1"], ["t1"], [{}], [0.2]
    )
    results = dense_retrieve("")
    assert isinstance(results, list)
