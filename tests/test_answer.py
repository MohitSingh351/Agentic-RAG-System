"""Tests for agent/answer.py."""

import json
from unittest.mock import MagicMock, call

import pytest

from agent.answer import generate_answer
from agent.state import AgentState


def _make_state(
    retrieval_results=None,
    tool_result="",
    query="What is attention?",
) -> AgentState:
    return {
        "query": query,
        "rewritten_query": query,
        "action": "RETRIEVE",
        "retrieval_results": retrieval_results or [],
        "tool_result": tool_result,
        "answer": "",
        "clarification_question": "",
        "refusal_message": "",
        "confidence": 0.0,
        "conversation_context": "",
        "semantic_facts": [],
        "similar_episodes": [],
        "trace": [],
        "session_id": "test",
        "debug": False,
    }


def _make_result(chunk_id: str, text: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"title": "Test Paper", "section": "Methods"},
        "score": score,
    }


def _make_llm(pass1_text: str, pass2_json: str | None = None) -> MagicMock:
    client = MagicMock()
    responses = [MagicMock(content=[MagicMock(text=pass1_text)])]
    if pass2_json is not None:
        responses.append(MagicMock(content=[MagicMock(text=pass2_json)]))
    else:
        no_contradiction = json.dumps({"contradictions": [], "revised_answer": pass1_text})
        responses.append(MagicMock(content=[MagicMock(text=no_contradiction)]))
    client.messages.create.side_effect = responses
    return client


def test_generate_answer_with_results():
    results = [_make_result(f"c{i}", f"text {i}", 0.8 - i * 0.1) for i in range(3)]
    llm = _make_llm("Attention is a mechanism that...")
    state = _make_state(retrieval_results=results)
    out = generate_answer(state, llm)
    assert isinstance(out["answer"], str)
    assert len(out["answer"]) > 0


def test_contradiction_detection_triggers_revision():
    results = [_make_result("c1", "text", 0.8)]
    contradiction_json = json.dumps({
        "contradictions": [{"claim": "X is true", "issue": "Source says X is false"}],
        "revised_answer": "This is the revised answer."
    })
    llm = _make_llm("Initial answer.", contradiction_json)
    state = _make_state(retrieval_results=results)
    out = generate_answer(state, llm)
    assert "revised" in out["answer"].lower() or out["answer"] == "This is the revised answer."


def test_contradiction_detection_malformed_json():
    results = [_make_result("c1", "text", 0.8)]
    llm = _make_llm("Initial answer.", "not valid json {{ at all")
    state = _make_state(retrieval_results=results)
    out = generate_answer(state, llm)
    assert out["answer"] == "Initial answer."


def test_confidence_calculation_three_chunks():
    results = [
        _make_result("c1", "t", 0.9),
        _make_result("c2", "t", 0.7),
        _make_result("c3", "t", 0.5),
    ]
    llm = _make_llm("Answer text.")
    state = _make_state(retrieval_results=results)
    out = generate_answer(state, llm)
    assert abs(out["confidence"] - (0.9 + 0.7 + 0.5) / 3) < 1e-6


def test_confidence_calculation_fewer_than_three():
    results = [_make_result("c1", "t", 0.8)]
    llm = _make_llm("Answer text.")
    state = _make_state(retrieval_results=results)
    out = generate_answer(state, llm)
    assert abs(out["confidence"] - 0.8) < 1e-6


def test_uses_tool_result_when_no_retrieval():
    llm = _make_llm("Tool-based answer.")
    state = _make_state(retrieval_results=[], tool_result="[1] Paper Title: ...")
    out = generate_answer(state, llm)
    assert isinstance(out["answer"], str)
    # Ensure LLM was called (not early-returned)
    llm.messages.create.assert_called()


def test_empty_retrieval_and_tool():
    llm = MagicMock()
    state = _make_state(retrieval_results=[], tool_result="")
    out = generate_answer(state, llm)
    assert "could not find" in out["answer"].lower() or "insufficient" in out["answer"].lower()
    llm.messages.create.assert_not_called()


def test_trace_entry_keys():
    results = [_make_result("c1", "t", 0.8)]
    llm = _make_llm("Answer.")
    state = _make_state(retrieval_results=results)
    generate_answer(state, llm)
    trace_entry = next(e for e in state["trace"] if e["node"] == "answer")
    assert "confidence" in trace_entry
    assert "contradiction_found" in trace_entry
    assert "source_chunks" in trace_entry
