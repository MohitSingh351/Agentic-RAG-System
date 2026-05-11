"""Tests for retrieval/rerank.py."""

from unittest.mock import MagicMock, patch

import numpy as np

import retrieval.rerank as rerank_mod
from retrieval.rerank import rerank


def _make_candidate(i: int, score: float = 0.5) -> dict:
    return {
        "chunk_id": f"chunk_{i:04d}",
        "text": f"Text for chunk {i}",
        "metadata": {"title": f"Paper {i}"},
        "score": score,
    }


def _mock_model(scores: list[float]):
    model = MagicMock()
    model.predict.return_value = np.array(scores)
    return model


@patch("retrieval.rerank._get_model")
def test_rerank_returns_top_n(mock_get_model):
    scores = [float(i) for i in range(10, 0, -1)]
    mock_get_model.return_value = _mock_model(scores)
    candidates = [_make_candidate(i) for i in range(10)]

    results = rerank("query", candidates, top_n=5)
    assert len(results) == 5


@patch("retrieval.rerank._get_model")
def test_rerank_sorted_by_score(mock_get_model):
    scores = [3.0, 1.0, 4.0, 1.5, 2.0]
    mock_get_model.return_value = _mock_model(scores)
    candidates = [_make_candidate(i) for i in range(5)]

    results = rerank("query", candidates)
    result_scores = [r["score"] for r in results]
    assert result_scores == sorted(result_scores, reverse=True)


def test_rerank_empty_candidates():
    with patch("retrieval.rerank._get_model") as mock_get_model:
        results = rerank("query", [], top_n=5)
    assert results == []
    mock_get_model.assert_not_called()


@patch("retrieval.rerank._get_model")
def test_rerank_fewer_than_top_n(mock_get_model):
    scores = [2.0, 1.0, 3.0]
    mock_get_model.return_value = _mock_model(scores)
    candidates = [_make_candidate(i) for i in range(3)]

    results = rerank("query", candidates, top_n=5)
    assert len(results) == 3


@patch("retrieval.rerank._get_model")
def test_rerank_model_called_once(mock_get_model):
    model = _mock_model([1.0, 2.0])
    mock_get_model.return_value = model
    candidates = [_make_candidate(i) for i in range(2)]

    rerank("query", candidates)
    model.predict.assert_called_once()
    pairs = model.predict.call_args[0][0]
    assert all(p[0] == "query" for p in pairs)


@patch("retrieval.rerank._get_model")
def test_rerank_preserves_metadata(mock_get_model):
    scores = [1.0, 2.0, 3.0]
    mock_get_model.return_value = _mock_model(scores)
    candidates = [_make_candidate(i) for i in range(3)]

    results = rerank("query", candidates)
    for r in results:
        assert "metadata" in r
        assert "chunk_id" in r
