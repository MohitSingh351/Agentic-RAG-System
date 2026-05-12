"""Tests for eval/run_eval.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.run_eval import (
    _score_action,
    _score_topic_coverage,
    _score_refusal,
    _score_clarification,
    run_evaluation,
)


def _make_state(**kwargs) -> dict:
    base = {
        "action": "RETRIEVE",
        "answer": "",
        "refusal_message": "",
        "clarification_question": "",
        "confidence": 0.8,
        "trace": [],
    }
    return {**base, **kwargs}


def test_eval_runs_all_questions(tmp_path):
    questions = [
        {
            "id": f"q{i:02d}",
            "question": f"Question {i}?",
            "expected_action": "RETRIEVE",
            "expected_topics": ["topic"],
            "should_refuse": False,
            "should_clarify": False,
            "notes": "",
        }
        for i in range(1, 11)
    ]
    q_path = tmp_path / "questions.json"
    q_path.write_text(json.dumps(questions))
    r_path = tmp_path / "results.json"

    with patch("eval.run_eval._RESULTS_PATH", r_path):
        with patch("eval.run_eval.run_agent", return_value=_make_state(answer="topic answer")):
            results = run_evaluation(q_path)

    assert len(results) == 10


def test_action_match_scoring():
    state = _make_state(action="RETRIEVE")
    assert _score_action(state, "RETRIEVE") == 1
    assert _score_action(state, "REFUSE") == 0


def test_topic_coverage_scoring():
    state = _make_state(answer="The paper discusses attention and transformers in detail")
    score = _score_topic_coverage(state, ["attention", "transformers", "diffusion"])
    assert abs(score - 2 / 3) < 1e-6


def test_refusal_scoring_correct():
    state = _make_state(refusal_message="I cannot help with that.")
    assert _score_refusal(state, should_refuse=True) == 1


def test_refusal_scoring_incorrect():
    state = _make_state(refusal_message="")
    assert _score_refusal(state, should_refuse=True) == 0


def test_results_json_written(tmp_path):
    questions = [
        {
            "id": "q01",
            "question": "What is RAG?",
            "expected_action": "RETRIEVE",
            "expected_topics": ["RAG"],
            "should_refuse": False,
            "should_clarify": False,
            "notes": "",
        }
    ]
    q_path = tmp_path / "questions.json"
    q_path.write_text(json.dumps(questions))
    r_path = tmp_path / "results.json"

    with patch("eval.run_eval._RESULTS_PATH", r_path):
        with patch("eval.run_eval.run_agent", return_value=_make_state(answer="RAG is great")):
            run_evaluation(q_path)

    assert r_path.exists()
    data = json.loads(r_path.read_text())
    assert "q01" in data


def test_table_printed(tmp_path, capsys):
    questions = [
        {
            "id": "q01",
            "question": "What is RAG?",
            "expected_action": "RETRIEVE",
            "expected_topics": [],
            "should_refuse": False,
            "should_clarify": False,
            "notes": "",
        }
    ]
    q_path = tmp_path / "questions.json"
    q_path.write_text(json.dumps(questions))
    r_path = tmp_path / "results.json"

    with patch("eval.run_eval._RESULTS_PATH", r_path):
        with patch("eval.run_eval.run_agent", return_value=_make_state()):
            run_evaluation(q_path)

    captured = capsys.readouterr()
    assert "Average" in captured.out
