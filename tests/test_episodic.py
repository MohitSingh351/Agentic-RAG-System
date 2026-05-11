"""Tests for memory/episodic.py."""

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import memory.episodic as ep_mod
from memory.episodic import SESSION_ID, find_similar, get_recent, log_episode


def _patch_log(monkeypatch, path: str):
    monkeypatch.setattr(ep_mod, "EPISODIC_LOG_PATH", path)
    monkeypatch.setenv("EPISODIC_LOG_PATH", path)


def test_log_episode_writes_jsonl(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    log_episode("What is RAG?", "RAG = Retrieval-Augmented Generation.", ["chunk_0001"])

    lines = Path(log_path).read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "query" in entry
    assert "answer" in entry
    assert "sources" in entry
    assert "timestamp" in entry
    assert "session_id" in entry


def test_log_multiple_episodes(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    for i in range(3):
        log_episode(f"question {i}", f"answer {i}", [])

    lines = Path(log_path).read_text().strip().split("\n")
    assert len(lines) == 3


def test_find_similar_above_threshold(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    log_episode("attention in transformers", "An attention mechanism...", [])
    results = find_similar("attention in transformers", threshold=0.8)
    assert len(results) >= 1
    assert results[0]["query"] == "attention in transformers"


def test_find_similar_below_threshold(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    log_episode("attention in transformers", "An attention mechanism...", [])
    results = find_similar("the price of tea in china", threshold=0.8)
    assert results == []


def test_find_similar_missing_file(tmp_path, monkeypatch):
    _patch_log(monkeypatch, str(tmp_path / "nonexistent.jsonl"))
    results = find_similar("any query")
    assert results == []


def test_find_similar_malformed_line(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    # Write one valid line and one invalid
    valid_entry = json.dumps({"query": "valid query", "answer": "a", "sources": [], "timestamp": "t", "session_id": "s"})
    with open(log_path, "w") as f:
        f.write(valid_entry + "\n")
        f.write("{ not valid json\n")

    results = find_similar("valid query", threshold=0.8)
    assert len(results) >= 1


def test_get_recent_returns_last_n(tmp_path, monkeypatch):
    log_path = str(tmp_path / "episodes.jsonl")
    _patch_log(monkeypatch, log_path)

    for i in range(10):
        log_episode(f"query {i}", f"answer {i}", [])

    recent = get_recent(3)
    assert len(recent) == 3
    # Last 3 queries
    assert recent[-1]["query"] == "query 9"
    assert recent[0]["query"] == "query 7"


def test_get_recent_empty_file(tmp_path, monkeypatch):
    _patch_log(monkeypatch, str(tmp_path / "empty.jsonl"))
    assert get_recent(5) == []


def test_session_id_is_uuid4():
    parsed = uuid.UUID(SESSION_ID)
    assert parsed.version == 4
