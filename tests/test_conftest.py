"""Tests for shared conftest fixtures."""

from unittest.mock import MagicMock

from agent.state import AgentState


def test_fixtures_provide_correct_types(
    mock_llm_client,
    sample_chunks,
    sample_state,
    tmp_bm25_index,
):
    assert isinstance(mock_llm_client, MagicMock)
    assert isinstance(sample_chunks, list)
    assert isinstance(sample_state, dict)
    assert tmp_bm25_index.exists()


def test_sample_state_has_all_fields(sample_state):
    expected_fields = {
        "query", "rewritten_query", "action", "retrieval_results",
        "tool_result", "answer", "clarification_question", "refusal_message",
        "confidence", "conversation_context", "semantic_facts",
        "similar_episodes", "trace", "session_id", "debug",
    }
    assert set(sample_state.keys()) == expected_fields


def test_mock_llm_returns_response(mock_llm_client):
    response = mock_llm_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response.content[0].text == "Default mock response."


def test_sample_chunks_have_required_fields(sample_chunks):
    required = {"chunk_id", "text", "arxiv_id", "section", "token_count",
                 "chunk_index", "title", "authors", "url", "score"}
    for chunk in sample_chunks:
        assert required.issubset(chunk.keys())


def test_tmp_bm25_index_loadable(tmp_bm25_index):
    import pickle
    from rank_bm25 import BM25Okapi
    with open(tmp_bm25_index, "rb") as fh:
        bm25, chunk_ids = pickle.load(fh)
    assert isinstance(bm25, BM25Okapi)
    assert len(chunk_ids) == 5
