"""Tests for eval/ablation.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.ablation import run_ablation, _get_retrieve_questions


def _make_state(**kwargs) -> dict:
    base = {
        "action": "RETRIEVE",
        "answer": "attention transformer",
        "refusal_message": "",
        "clarification_question": "",
        "confidence": 0.8,
        "trace": [],
    }
    return {**base, **kwargs}


def _make_questions(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"q{i:02d}",
            "question": f"Question {i}?",
            "expected_action": "RETRIEVE",
            "expected_topics": ["attention", "transformer"],
            "should_refuse": False,
            "should_clarify": False,
            "notes": "",
        }
        for i in range(1, n + 1)
    ]


def test_three_configs_run(tmp_path, capsys):
    questions = _make_questions(5)

    with patch("eval.ablation.run_agent", return_value=_make_state()):
        with patch("eval.ablation.dense_retrieve", return_value=[]):
            results_d = run_ablation({"name": "dense_only"}, questions)
            results_h = run_ablation({"name": "hybrid_no_rerank"}, questions)
            results_f = run_ablation({"name": "full_pipeline"}, questions)

    assert len(results_d) == 5
    assert len(results_h) == 5
    assert len(results_f) == 5
    # 3 configs × 5 questions = 15 total
    assert len(results_d) + len(results_h) + len(results_f) == 15


def test_dense_only_skips_bm25(tmp_path):
    questions = _make_questions(2)
    with patch("eval.ablation.run_agent", return_value=_make_state()):
        with patch("agent.graph.hybrid_retrieve") as mock_hybrid:
            with patch("retrieval.sparse.sparse_retrieve") as mock_sparse:
                run_ablation({"name": "dense_only"}, questions)
    # sparse_retrieve should not be called from within run_ablation directly
    mock_sparse.assert_not_called()


def test_hybrid_no_rerank_skips_reranker(tmp_path):
    questions = _make_questions(2)
    with patch("eval.ablation.run_agent", return_value=_make_state()):
        with patch("retrieval.rerank.rerank") as mock_rerank:
            run_ablation({"name": "hybrid_no_rerank"}, questions)
    mock_rerank.assert_not_called()


def test_ablation_table_printed(tmp_path, capsys):
    questions = _make_questions(3)
    with patch("eval.ablation.run_agent", return_value=_make_state()):
        with patch("eval.ablation.dense_retrieve", return_value=[]):
            import eval.ablation as abl
            import unittest.mock as mock
            with mock.patch.object(abl, "_get_retrieve_questions", return_value=questions):
                abl._run_all_configs()

    captured = capsys.readouterr()
    assert "dense_only" in captured.out
    assert "hybrid_no_rerank" in captured.out
    assert "full_pipeline" in captured.out


def test_full_pipeline_avg_column():
    questions = _make_questions(3)
    # answer contains both expected topics
    state = _make_state(answer="attention transformer paper")
    with patch("eval.ablation.run_agent", return_value=state):
        results = run_ablation({"name": "full_pipeline"}, questions)

    scores = [r["topic_coverage"] for r in results]
    avg = sum(scores) / len(scores)
    # All answers contain both topics → each coverage = 1.0
    assert abs(avg - 1.0) < 1e-6
