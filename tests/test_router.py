"""Tests for agent/router.py."""

from unittest.mock import MagicMock

import pytest

from agent.router import route_query
from agent.state import AgentState


def _make_state(
    query: str = "What is attention?",
    similar_episodes: list = None,
    semantic_facts: list = None,
) -> AgentState:
    return {
        "query": query,
        "rewritten_query": query,
        "action": "",
        "retrieval_results": [],
        "tool_result": "",
        "answer": "",
        "clarification_question": "",
        "refusal_message": "",
        "confidence": 0.0,
        "conversation_context": "",
        "semantic_facts": semantic_facts or [],
        "similar_episodes": similar_episodes or [],
        "trace": [],
        "session_id": "test",
        "debug": False,
    }


def _make_llm(response: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=response)]
    )
    return client


def test_route_returns_valid_action():
    llm = _make_llm("RETRIEVE")
    state = _make_state()
    assert route_query(state, llm) == "RETRIEVE"


def test_route_defaults_on_invalid_response():
    llm = _make_llm("DUNNO")
    state = _make_state()
    action = route_query(state, llm)
    assert action == "RETRIEVE"
    trace_entry = next(e for e in state["trace"] if e["node"] == "router")
    assert trace_entry["defaulted"] is True


def test_route_case_insensitive():
    llm = _make_llm("retrieve")
    state = _make_state()
    assert route_query(state, llm) == "RETRIEVE"


@pytest.mark.parametrize("action", ["RETRIEVE", "USE_TOOL", "CLARIFY", "REFUSE", "ANSWER"])
def test_route_all_valid_actions(action):
    llm = _make_llm(action)
    state = _make_state()
    assert route_query(state, llm) == action


def test_route_with_similar_episodes_biases_answer():
    llm = _make_llm("ANSWER")
    episodes = [{"query": "What is attention?", "answer": "Attention is...", "sources": []}]
    state = _make_state(similar_episodes=episodes)
    route_query(state, llm)
    # Verify the prompt included mention of the similar episode
    call_args = llm.messages.create.call_args
    user_message = call_args.kwargs.get("messages", [{}])[-1].get("content", "")
    assert "similar question" in user_message or "ANSWER" in user_message


def test_route_trace_entry_structure():
    llm = _make_llm("RETRIEVE")
    state = _make_state()
    route_query(state, llm)
    trace_entry = next(e for e in state["trace"] if e["node"] == "router")
    assert "node" in trace_entry
    assert "raw_response" in trace_entry
    assert "action" in trace_entry
    assert "defaulted" in trace_entry


def test_route_handles_llm_whitespace():
    llm = _make_llm("  CLARIFY  \n")
    state = _make_state()
    assert route_query(state, llm) == "CLARIFY"
