"""Tests for memory/semantic.py."""

from unittest.mock import MagicMock

import pytest

import memory.semantic as sem_mod
from memory.semantic import (
    clear,
    extract_and_store,
    get_all_facts,
    get_fact,
    upsert_fact,
)


def _patch_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test_semantic.db")
    monkeypatch.setattr(sem_mod, "SEMANTIC_DB_PATH", db_path)
    monkeypatch.setenv("SEMANTIC_DB_PATH", db_path)


def _make_llm(json_response: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json_response)]
    )
    return client


def test_upsert_and_retrieve(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    upsert_fact("research_interest", "reinforcement learning", 0.9)
    fact = get_fact("research_interest")
    assert fact is not None
    assert fact["value"] == "reinforcement learning"
    assert abs(fact["confidence"] - 0.9) < 1e-6


def test_upsert_overwrites_existing(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    upsert_fact("expertise", "beginner", 0.8)
    upsert_fact("expertise", "expert", 0.95)
    all_facts = get_all_facts()
    expertise_facts = [f for f in all_facts if f["key"] == "expertise"]
    assert len(expertise_facts) == 1
    assert expertise_facts[0]["value"] == "expert"


def test_extract_calls_llm(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    llm = _make_llm('[{"key": "interest", "value": "NLP", "confidence": 0.9}]')
    extract_and_store("I research NLP models", llm)
    fact = get_fact("interest")
    assert fact is not None
    assert fact["value"] == "NLP"


def test_extract_handles_malformed_json(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    llm = _make_llm("not valid json at all")
    extract_and_store("some message", llm)
    assert get_all_facts() == []


def test_extract_empty_response(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    upsert_fact("pre_existing", "value", 1.0)
    llm = _make_llm("[]")
    extract_and_store("boring message", llm)
    assert get_fact("pre_existing") is not None


def test_confidence_below_threshold_stored(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    upsert_fact("low_conf_fact", "maybe", 0.3)
    fact = get_fact("low_conf_fact")
    assert fact is not None
    assert abs(fact["confidence"] - 0.3) < 1e-6


def test_get_nonexistent_key(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    assert get_fact("does_not_exist") is None


def test_clear_removes_all(tmp_path, monkeypatch):
    _patch_db(monkeypatch, tmp_path)
    upsert_fact("k1", "v1", 0.9)
    upsert_fact("k2", "v2", 0.8)
    upsert_fact("k3", "v3", 0.7)
    clear()
    assert get_all_facts() == []


def test_db_path_uses_env_var(tmp_path, monkeypatch):
    custom_path = str(tmp_path / "custom.db")
    monkeypatch.setenv("SEMANTIC_DB_PATH", custom_path)
    monkeypatch.setattr(sem_mod, "SEMANTIC_DB_PATH", custom_path)
    upsert_fact("test_key", "test_val", 1.0)
    from pathlib import Path
    assert Path(custom_path).exists()
