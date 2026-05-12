"""Tests for agent/graph.py — node routing and end-to-end paths."""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent.graph as graph_mod
from agent.graph import run_agent
from agent.state import AgentState


@pytest.fixture(autouse=True)
def reset_graph_state():
    """Reset module-level singletons between tests to prevent state leakage."""
    graph_mod._conv_memory.clear()
    graph_mod._llm_client = None
    yield
    graph_mod._conv_memory.clear()
    graph_mod._llm_client = None


def _make_retrieval_results(n: int = 3, score: float = 0.8) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "text": f"text {i}", "metadata": {"title": "T"}, "score": score}
        for i in range(n)
    ]


def _make_llm(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text=r)]) for r in responses
    ]
    return client


def test_graph_compiles():
    from agent.graph import app
    assert app is not None


@patch("agent.graph._get_llm")
@patch("agent.graph.hybrid_retrieve")
@patch("agent.graph.rerank")
def test_retrieve_path_end_to_end(mock_rerank, mock_hybrid, mock_get_llm):
    mock_hybrid.return_value = _make_retrieval_results()
    mock_rerank.return_value = _make_retrieval_results(score=0.85)

    no_contradiction = json.dumps({"contradictions": [], "revised_answer": "Test answer."})
    # Calls: rewrite (HyDE), route, answer pass1, answer pass2
    mock_get_llm.return_value = _make_llm([
        "rewritten query",  # rewrite context step (won't call if no context)
        "What is attention mechanism in neural networks?",  # HyDE
        "RETRIEVE",          # router
        "The answer is...",  # answer pass1
        no_contradiction,    # answer pass2
        "[]",                # semantic extraction
    ])

    state = run_agent("What is attention?")
    assert state["action"] == "RETRIEVE"
    assert state["answer"]
    assert len(state["trace"]) >= 2


@patch("agent.graph._get_llm")
@patch("agent.graph.hybrid_retrieve")
@patch("agent.graph.rerank")
def test_low_confidence_triggers_soft_refuse(mock_rerank, mock_hybrid, mock_get_llm):
    # Return low-score results
    mock_hybrid.return_value = _make_retrieval_results(score=0.1)
    mock_rerank.return_value = _make_retrieval_results(n=1, score=0.1)

    no_contradiction = json.dumps({"contradictions": [], "revised_answer": "Answer."})
    mock_get_llm.return_value = _make_llm([
        "rewritten",     # HyDE or context
        "What is attention in detail?",
        "RETRIEVE",      # router
    ])

    state = run_agent("What is attention?")
    assert state["refusal_message"]
    assert "confidence" in state["refusal_message"].lower() or "25%" in state["refusal_message"] or "low" in state["refusal_message"].lower()


@patch("agent.graph._get_llm")
def test_clarify_path(mock_get_llm):
    # "Tell me about transformers." doesn't start with a question word → HyDE skipped
    mock_get_llm.return_value = _make_llm([
        "CLARIFY",   # router
        "Are you asking about transformer architecture or a specific model?",  # clarify
    ])
    state = run_agent("Tell me about transformers.")
    assert state["clarification_question"]
    assert state["answer"] == ""


@patch("agent.graph._get_llm")
def test_refuse_path(mock_get_llm):
    mock_get_llm.return_value = _make_llm([
        "What is the best pizza recipe?",  # passthrough (not a question we rewrite)
        "REFUSE",  # router
    ])
    state = run_agent("What is the best pizza recipe?")
    assert state["refusal_message"]
    assert state["answer"] == ""


@patch("agent.graph._get_llm")
@patch("agent.graph.arxiv_search")
def test_use_tool_path(mock_arxiv, mock_get_llm):
    mock_arxiv.invoke.return_value = "[1] Title: Recent Paper on X"
    no_contradiction = json.dumps({"contradictions": [], "revised_answer": "Tool answer."})
    mock_get_llm.return_value = _make_llm([
        "What is speculative decoding?",  # HyDE
        "USE_TOOL",  # router
        "Based on recent arXiv...",  # answer pass1
        no_contradiction,  # answer pass2
        "[]",  # semantic extraction
    ])
    state = run_agent("What papers on speculative decoding came out this week?")
    assert state["tool_result"]
    assert state["answer"]


@patch("agent.graph._get_llm")
def test_answer_path_skips_retrieval(mock_get_llm):
    no_contradiction = json.dumps({"contradictions": [], "revised_answer": "Direct answer."})
    mock_get_llm.return_value = _make_llm([
        "What is RAG?",  # passthrough
        "ANSWER",  # router → skip retrieval
        "RAG is retrieval augmented generation.",  # answer
        no_contradiction,
        "[]",
    ])
    with patch("agent.graph.hybrid_retrieve") as mock_hybrid:
        state = run_agent("What is RAG?")
        mock_hybrid.assert_not_called()
    assert state["answer"]
    assert state["retrieval_results"] == []


@patch("agent.graph._get_llm")
@patch("agent.graph.hybrid_retrieve")
@patch("agent.graph.rerank")
def test_trace_accumulates_all_nodes(mock_rerank, mock_hybrid, mock_get_llm):
    mock_hybrid.return_value = _make_retrieval_results()
    mock_rerank.return_value = _make_retrieval_results(score=0.85)
    no_contradiction = json.dumps({"contradictions": [], "revised_answer": "Answer."})
    mock_get_llm.return_value = _make_llm([
        "rewritten",
        "What is attention in detail?",
        "RETRIEVE",
        "The attention mechanism...",
        no_contradiction,
        "[]",
    ])
    state = run_agent("What is attention?")
    trace_nodes = [e["node"] for e in state["trace"]]
    assert len(trace_nodes) >= 3
