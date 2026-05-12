"""Tests for agent/clarify.py."""

from unittest.mock import MagicMock

import pytest

from agent.clarify import _FALLBACK, generate_clarification
from agent.state import AgentState


def _make_state(query: str = "Tell me about transformers.") -> AgentState:
    return {
        "query": query, "rewritten_query": query, "action": "CLARIFY",
        "retrieval_results": [], "tool_result": "", "answer": "",
        "clarification_question": "", "refusal_message": "",
        "confidence": 0.0, "conversation_context": "",
        "semantic_facts": [], "similar_episodes": [],
        "trace": [], "session_id": "test", "debug": False,
    }


def _make_llm(response: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = MagicMock(content=[MagicMock(text=response)])
    return client


def test_clarify_returns_question():
    llm = _make_llm("What specific time period are you asking about?")
    result = generate_clarification(_make_state(), llm)
    assert result.endswith("?")


def test_clarify_extracts_from_preamble():
    llm = _make_llm("I need to know: What specific aspect do you mean? Please elaborate more.")
    result = generate_clarification(_make_state(), llm)
    assert result == "What specific aspect do you mean?"


def test_clarify_appends_question_mark():
    llm = _make_llm("Please clarify the scope of your question")
    result = generate_clarification(_make_state(), llm)
    assert result.endswith("?")


def test_clarify_fallback_on_error():
    llm = MagicMock()
    llm.messages.create.side_effect = Exception("LLM down")
    result = generate_clarification(_make_state(), llm)
    assert result == _FALLBACK


def test_clarify_trace_entry():
    llm = _make_llm("Are you asking about NLP transformers?")
    state = _make_state()
    generate_clarification(state, llm)
    trace_nodes = [e["node"] for e in state["trace"]]
    assert "clarify" in trace_nodes
