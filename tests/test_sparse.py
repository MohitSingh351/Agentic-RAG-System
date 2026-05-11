"""Tests for retrieval/sparse.py."""

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
from rank_bm25 import BM25Okapi

import retrieval.sparse as sparse_mod
from retrieval.sparse import sparse_retrieve


def _setup_bm25(tmp_path, texts: list[str]) -> tuple[str, str]:
    """Create BM25 index and corpus metadata in tmp_path, return paths."""
    chunk_ids = [f"chunk_{i:04d}" for i in range(len(texts))]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    bm25_path = str(tmp_path / "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump((bm25, chunk_ids), f)

    meta = {
        cid: {"chunk_id": cid, "text": text, "arxiv_id": "2401.00001",
              "title": "T", "section": "S", "page_start": 1, "url": "u"}
        for cid, text in zip(chunk_ids, texts)
    }
    meta_path = str(tmp_path / "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return bm25_path, meta_path


def _patch_paths(monkeypatch, bm25_path, meta_path):
    monkeypatch.setattr(sparse_mod, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(sparse_mod, "CORPUS_METADATA_PATH", meta_path)
    monkeypatch.setattr(sparse_mod, "_bm25", None)
    monkeypatch.setattr(sparse_mod, "_chunk_ids", None)
    monkeypatch.setattr(sparse_mod, "_corpus_meta", None)


def test_sparse_retrieve_returns_list(tmp_path, monkeypatch):
    texts = ["cat sat on mat", "dog ran away", "cat ate fish", "bird flew high", "fish swam deep"]
    bm25_path, meta_path = _setup_bm25(tmp_path, texts)
    _patch_paths(monkeypatch, bm25_path, meta_path)

    results = sparse_retrieve("cat")
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "metadata" in r
        assert "score" in r


def test_sparse_top_k(tmp_path, monkeypatch):
    texts = [f"document number {i}" for i in range(10)]
    bm25_path, meta_path = _setup_bm25(tmp_path, texts)
    _patch_paths(monkeypatch, bm25_path, meta_path)

    results = sparse_retrieve("document", top_k=3)
    assert len(results) == 3


def test_sparse_relevant_result_first(tmp_path, monkeypatch):
    texts = ["cat sat on mat", "dog ran away", "cat ate fish"]
    bm25_path, meta_path = _setup_bm25(tmp_path, texts)
    _patch_paths(monkeypatch, bm25_path, meta_path)

    results = sparse_retrieve("cat", top_k=3)
    top_texts = {results[0]["text"], results[1]["text"]}
    assert "cat sat on mat" in top_texts or "cat ate fish" in top_texts
    assert results[0]["score"] >= results[-1]["score"]


def test_sparse_empty_query(tmp_path, monkeypatch):
    texts = ["hello world", "foo bar"]
    bm25_path, meta_path = _setup_bm25(tmp_path, texts)
    _patch_paths(monkeypatch, bm25_path, meta_path)

    results = sparse_retrieve("")
    assert isinstance(results, list)


def test_sparse_top_k_larger_than_corpus(tmp_path, monkeypatch):
    texts = ["doc one", "doc two", "doc three", "doc four", "doc five"]
    bm25_path, meta_path = _setup_bm25(tmp_path, texts)
    _patch_paths(monkeypatch, bm25_path, meta_path)

    results = sparse_retrieve("doc", top_k=20)
    assert len(results) == 5


def test_sparse_nonexistent_bm25_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse_mod, "BM25_INDEX_PATH", str(tmp_path / "nonexistent.pkl"))
    monkeypatch.setattr(sparse_mod, "CORPUS_METADATA_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(sparse_mod, "_bm25", None)
    monkeypatch.setattr(sparse_mod, "_chunk_ids", None)
    monkeypatch.setattr(sparse_mod, "_corpus_meta", None)

    with pytest.raises(FileNotFoundError, match="BM25 index not found"):
        sparse_retrieve("query")
