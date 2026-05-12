"""Shared pytest fixtures for the SkyClad test suite."""

import pickle
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rank_bm25 import BM25Okapi

from agent.state import AgentState


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """MagicMock Anthropic client with a default single response."""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Default mock response.")]
    )
    return client


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Five hardcoded chunk dicts with all required fields."""
    return [
        {
            "chunk_id": f"2401.00001_chunk_{i:04d}",
            "text": f"Sample text for chunk {i}. This discusses transformer attention mechanisms.",
            "arxiv_id": "2401.00001",
            "section": "Methods",
            "token_count": 450 + i * 10,
            "chunk_index": i,
            "page_start": i,
            "page_end": i + 1,
            "title": "Test Paper on Transformers",
            "authors": ["Author A", "Author B"],
            "url": "https://arxiv.org/abs/2401.00001",
            "metadata": {
                "title": "Test Paper on Transformers",
                "section": "Methods",
                "arxiv_id": "2401.00001",
            },
            "score": 0.9 - i * 0.1,
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_state() -> AgentState:
    """Fully populated AgentState for testing."""
    return AgentState(
        query="What is the attention mechanism?",
        rewritten_query="What is the attention mechanism in transformer models?",
        action="RETRIEVE",
        retrieval_results=[],
        tool_result="",
        answer="",
        clarification_question="",
        refusal_message="",
        confidence=0.0,
        conversation_context="",
        semantic_facts=[],
        similar_episodes=[],
        trace=[],
        session_id="test-session-001",
        debug=False,
    )


@pytest.fixture(scope="session")
def test_chroma_collection():
    """In-memory ChromaDB collection with 5 test documents."""
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(5)],
        documents=[
            f"Transformer attention mechanism paper {i}. "
            f"This discusses self-attention and multi-head attention."
            for i in range(5)
        ],
        metadatas=[{"title": f"Test Paper {i}", "section": "Methods"} for i in range(5)],
        embeddings=[[float(j == i) for j in range(5)] for i in range(5)],
    )
    return collection


@pytest.fixture
def tmp_bm25_index(tmp_path) -> Path:
    """Temporary BM25 index file with 5 known texts. Returns path."""
    corpus = [
        "transformer attention mechanism",
        "diffusion model generation",
        "graph neural network",
        "few-shot learning NLP",
        "reinforcement learning policy",
    ]
    tokenized = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [f"chunk_{i:04d}" for i in range(5)]
    index_path = tmp_path / "bm25_index.pkl"
    with open(index_path, "wb") as fh:
        pickle.dump((bm25, chunk_ids), fh)
    return index_path
