"""Tests for agent/query_rewriter.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.query_rewriter import rewrite_query
from agent.state import AgentState


def _make_state(
    query: str = "test",
    context: str = "",
    action: str = "RETRIEVE",
) -> AgentState:
    return {
        "query": query,
        "rewritten_query": "",
        "action": action,
        "retrieval_results": [],
        "tool_result": "",
        "answer": "",
        "clarification_question": "",
        "refusal_message": "",
        "confidence": 0.0,
        "conversation_context": context,
        "semantic_facts": [],
        "similar_episodes": [],
        "trace": [],
        "session_id": "test",
        "debug": False,
    }


def _make_llm(response: str = "rewritten query") -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=response)]
    )
    return client


def test_rewrite_with_context():
    llm = _make_llm("What is the attention mechanism in transformers?")
    state = _make_state(query="What about it?", context="User: Tell me about transformers.\nAssistant: ...")
    result = rewrite_query(state, llm)
    llm.messages.create.assert_called()
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_no_context_passthrough():
    llm = _make_llm()
    state = _make_state(query="machine learning basics", context="", action="RETRIEVE")
    # Non-question query, no context → should not call LLM for context step
    # But HyDE won't apply since no question word
    result = rewrite_query(state, llm)
    assert result == "machine learning basics"
    llm.messages.create.assert_not_called()


def test_hyde_applied_for_question_query():
    llm = _make_llm("Hypothetical answer passage about attention.")
    state = _make_state(query="What is attention?", context="", action="RETRIEVE")
    result = rewrite_query(state, llm)
    assert "[HYPOTHETICAL CONTEXT]" in result


def test_hyde_not_applied_for_non_retrieve_action():
    llm = _make_llm("Hypothetical passage.")
    state = _make_state(query="What is attention?", context="", action="USE_TOOL")
    result = rewrite_query(state, llm)
    assert "[HYPOTHETICAL CONTEXT]" not in result


def test_rewrite_retries_on_llm_failure():
    call_count = [0]
    def flaky_create(**kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("transient error")
        return MagicMock(content=[MagicMock(text="retry success")])

    llm = MagicMock()
    llm.messages.create.side_effect = flaky_create
    state = _make_state(query="What is RAG?", context="", action="RETRIEVE")
    # Should succeed on retry (tenacity retries up to 2 times)
    result = rewrite_query(state, llm)
    assert isinstance(result, str)


def test_trace_entry_added():
    llm = _make_llm("Hypothetical passage about attention mechanisms.")
    state = _make_state(query="What is attention?", context="", action="RETRIEVE")
    rewrite_query(state, llm)
    trace_nodes = [e["node"] for e in state["trace"]]
    assert "query_rewriter" in trace_nodes
    rewriter_entry = next(e for e in state["trace"] if e["node"] == "query_rewriter")
    assert "hyde_applied" in rewriter_entry


def test_rewrite_empty_query():
    llm = _make_llm("some response")
    state = _make_state(query="", context="", action="RETRIEVE")
    result = rewrite_query(state, llm)
    assert result == ""
    llm.messages.create.assert_not_called()
