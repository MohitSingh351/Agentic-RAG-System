"""Tests for retrieval/hybrid.py."""

from unittest.mock import patch

import pytest

from retrieval.hybrid import hybrid_retrieve


def _make_result(chunk_id: str, score: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "text": f"text for {chunk_id}", "metadata": {}, "score": score}


def _dense(results):
    return patch("retrieval.hybrid.dense_retrieve", return_value=results)


def _sparse(results):
    return patch("retrieval.hybrid.sparse_retrieve", return_value=results)


def test_rrf_formula_correctness():
    """Manual RRF calculation should match hybrid_retrieve output."""
    rrf_k = 60
    # dense rank 1 = chunk_A, rank 2 = chunk_B
    # sparse rank 1 = chunk_B, rank 2 = chunk_A
    dense_results = [_make_result("A"), _make_result("B")]
    sparse_results = [_make_result("B"), _make_result("A")]

    with _dense(dense_results), _sparse(sparse_results):
        results = hybrid_retrieve("q", top_k=2, rrf_k=rrf_k)

    scores = {r["chunk_id"]: r["score"] for r in results}
    # A: 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522
    # B: 1/(60+2) + 1/(60+1) = same
    expected_a = 1 / (rrf_k + 1) + 1 / (rrf_k + 2)
    expected_b = 1 / (rrf_k + 2) + 1 / (rrf_k + 1)
    assert abs(scores["A"] - expected_a) < 1e-9
    assert abs(scores["B"] - expected_b) < 1e-9


def test_hybrid_merges_unique_chunks():
    dense_results = [_make_result("A"), _make_result("B")]
    sparse_results = [_make_result("B"), _make_result("C")]

    with _dense(dense_results), _sparse(sparse_results):
        results = hybrid_retrieve("q", top_k=10)

    chunk_ids = {r["chunk_id"] for r in results}
    assert {"A", "B", "C"} == chunk_ids


def test_hybrid_top_k_output():
    dense_results = [_make_result(f"d{i}") for i in range(20)]
    sparse_results = [_make_result(f"s{i}") for i in range(20)]

    with _dense(dense_results), _sparse(sparse_results):
        results = hybrid_retrieve("q", top_k=20)

    assert len(results) == 20


def test_hybrid_sorted_by_rrf_score():
    dense_results = [_make_result(f"c{i}") for i in range(5)]
    sparse_results = [_make_result(f"c{i}") for i in range(5)]

    with _dense(dense_results), _sparse(sparse_results):
        results = hybrid_retrieve("q")

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_empty_dense():
    sparse_results = [_make_result("A"), _make_result("B")]

    with _dense([]), _sparse(sparse_results):
        results = hybrid_retrieve("q")

    assert len(results) == 2
    chunk_ids = {r["chunk_id"] for r in results}
    assert "A" in chunk_ids and "B" in chunk_ids


def test_hybrid_empty_both():
    with _dense([]), _sparse([]):
        results = hybrid_retrieve("q")
    assert results == []


def test_hybrid_chunk_in_both_has_higher_score():
    """Rank-1 in both lists should score higher than rank-1 in only one."""
    # chunk_A appears rank-1 in both; chunk_B appears rank-1 in only dense
    dense_results = [_make_result("A"), _make_result("B")]
    sparse_results = [_make_result("A"), _make_result("C")]

    with _dense(dense_results), _sparse(sparse_results):
        results = hybrid_retrieve("q", top_k=3)

    scores = {r["chunk_id"]: r["score"] for r in results}
    # A is rank 1 in both → highest score
    assert scores["A"] > scores.get("B", 0)
    assert scores["A"] > scores.get("C", 0)
