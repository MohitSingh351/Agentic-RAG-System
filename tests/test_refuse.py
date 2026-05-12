"""Tests for agent/refuse.py."""

import pytest

from agent.refuse import CONFIDENCE_THRESHOLD, generate_refusal
from agent.state import AgentState


def _make_state(
    confidence: float = 0.0,
    retrieval_results=None,
    query: str = "What is the best pizza recipe?",
) -> AgentState:
    return {
        "query": query, "rewritten_query": query, "action": "REFUSE",
        "retrieval_results": retrieval_results or [],
        "tool_result": "", "answer": "",
        "clarification_question": "", "refusal_message": "",
        "confidence": confidence, "conversation_context": "",
        "semantic_facts": [], "similar_episodes": [],
        "trace": [], "session_id": "test", "debug": False,
    }


def test_hard_refusal_message():
    state = _make_state()
    result = generate_refusal(state, refusal_type="hard")
    assert "specialized in cs.AI" in result


def test_soft_refusal_includes_confidence():
    state = _make_state(confidence=0.25)
    result = generate_refusal(state, refusal_type="soft")
    assert "25%" in result


def test_soft_refusal_includes_partial_answer():
    results = [{"chunk_id": "c1", "text": "This is the chunk text content.", "metadata": {}, "score": 0.3}]
    state = _make_state(confidence=0.3, retrieval_results=results)
    result = generate_refusal(state, refusal_type="soft")
    assert "This is the chunk text content." in result


def test_soft_refusal_empty_results():
    state = _make_state(confidence=0.2, retrieval_results=[])
    result = generate_refusal(state, refusal_type="soft")
    assert isinstance(result, str)
    assert len(result) > 0


def test_confidence_threshold_constant():
    assert CONFIDENCE_THRESHOLD == 0.4


def test_trace_entry_contains_refusal_type():
    state = _make_state()
    generate_refusal(state, refusal_type="hard")
    trace_entry = next(e for e in state["trace"] if e["node"] == "refuse")
    assert trace_entry["refusal_type"] == "hard"
