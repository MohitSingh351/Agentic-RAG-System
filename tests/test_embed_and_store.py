"""Tests for ingest/embed_and_store.py."""

import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from rank_bm25 import BM25Okapi

import ingest.embed_and_store as eas


def _make_chunk(i: int) -> dict:
    return {
        "chunk_id": f"2401.00001_chunk_{i:04d}",
        "text": f"This is chunk number {i} with some content.",
        "arxiv_id": "2401.00001",
        "title": "Test Paper",
        "section": "Introduction",
        "page_start": 1,
        "url": "https://arxiv.org/abs/2401.00001",
    }


def _make_chunks(n: int) -> list[dict]:
    return [_make_chunk(i) for i in range(n)]


def _mock_openai_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 1536 for _ in texts]


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_embed_batches_correctly(mock_collection, mock_openai_cls, tmp_path, monkeypatch):
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(eas, "CORPUS_METADATA_PATH", str(tmp_path / "meta.json"))

    mock_embed = MagicMock()
    mock_embed.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536) for _ in range(100)]
    )
    mock_openai_cls.return_value = mock_embed
    mock_collection.return_value = MagicMock()

    chunks = _make_chunks(250)
    eas.embed_and_store(chunks)

    assert mock_embed.embeddings.create.call_count == 3  # 100+100+50


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_chroma_upsert_called(mock_collection, mock_openai_cls, tmp_path, monkeypatch):
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(eas, "CORPUS_METADATA_PATH", str(tmp_path / "meta.json"))

    batch_sizes = []

    def fake_upsert(ids, embeddings, documents, metadatas):
        batch_sizes.append(len(ids))

    mock_col = MagicMock()
    mock_col.upsert.side_effect = fake_upsert
    mock_collection.return_value = mock_col

    mock_embed = MagicMock()
    mock_embed.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536) for _ in range(eas.EMBED_BATCH_SIZE)]
    )
    mock_openai_cls.return_value = mock_embed

    chunks = _make_chunks(150)
    eas.embed_and_store(chunks)

    for size in batch_sizes:
        assert size <= eas.EMBED_BATCH_SIZE


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_bm25_pickle_saved(mock_collection, mock_openai_cls, tmp_path, monkeypatch):
    bm25_path = str(tmp_path / "bm25.pkl")
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(eas, "CORPUS_METADATA_PATH", str(tmp_path / "meta.json"))

    mock_embed = MagicMock()
    mock_embed.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536) for _ in range(10)]
    )
    mock_openai_cls.return_value = mock_embed
    mock_collection.return_value = MagicMock()

    eas.embed_and_store(_make_chunks(10))

    assert Path(bm25_path).exists()
    with open(bm25_path, "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, tuple)
    assert isinstance(obj[0], BM25Okapi)
    assert isinstance(obj[1], list)


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_corpus_metadata_json_saved(mock_collection, mock_openai_cls, tmp_path, monkeypatch):
    meta_path = str(tmp_path / "meta.json")
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(eas, "CORPUS_METADATA_PATH", meta_path)

    mock_embed = MagicMock()
    mock_embed.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536) for _ in range(5)]
    )
    mock_openai_cls.return_value = mock_embed
    mock_collection.return_value = MagicMock()

    chunks = _make_chunks(5)
    eas.embed_and_store(chunks)

    assert Path(meta_path).exists()
    with open(meta_path) as f:
        meta = json.load(f)
    for chunk in chunks:
        assert chunk["chunk_id"] in meta


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_embed_retries_on_api_error(mock_collection, mock_openai_cls, tmp_path, monkeypatch):
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
    monkeypatch.setattr(eas, "CORPUS_METADATA_PATH", str(tmp_path / "meta.json"))

    call_count = [0]

    def flaky_create(**kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise APIError("rate limit", request=MagicMock(), body=None)
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in range(5)])

    mock_embed = MagicMock()
    mock_embed.embeddings.create.side_effect = flaky_create
    mock_openai_cls.return_value = mock_embed
    mock_collection.return_value = MagicMock()

    eas.embed_and_store(_make_chunks(5))
    assert call_count[0] == 3


@patch("ingest.embed_and_store.OpenAI")
@patch("ingest.embed_and_store.get_chroma_collection")
def test_embed_empty_chunks(mock_collection, mock_openai_cls):
    mock_embed = MagicMock()
    mock_openai_cls.return_value = mock_embed

    eas.embed_and_store([])

    mock_embed.embeddings.create.assert_not_called()


def test_load_bm25_returns_tuple(tmp_path, monkeypatch):
    bm25_path = str(tmp_path / "bm25.pkl")
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", bm25_path)

    bm25_obj = BM25Okapi([["hello", "world"], ["foo", "bar"]])
    chunk_ids = ["chunk_0000", "chunk_0001"]
    with open(bm25_path, "wb") as f:
        pickle.dump((bm25_obj, chunk_ids), f)

    result = eas.load_bm25()
    assert isinstance(result, tuple)
    assert isinstance(result[0], BM25Okapi)
    assert result[1] == chunk_ids


def test_load_bm25_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(eas, "BM25_INDEX_PATH", str(tmp_path / "nonexistent.pkl"))
    with pytest.raises(FileNotFoundError):
        eas.load_bm25()
