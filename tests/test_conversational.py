"""Tests for memory/conversational.py."""

from datetime import datetime

import pytest

from memory.conversational import ConversationalMemory


def test_add_turn_stores_correctly():
    cm = ConversationalMemory()
    cm.add_turn("user", "Hello!")
    ctx = cm.get_context()
    assert len(ctx) == 1
    assert ctx[0]["role"] == "user"
    assert ctx[0]["content"] == "Hello!"


def test_window_size_limit():
    cm = ConversationalMemory(max_turns=6)
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        cm.add_turn(role, f"msg {i}")
    assert len(cm.get_context()) == 6


def test_clear_resets_memory():
    cm = ConversationalMemory()
    cm.add_turn("user", "a")
    cm.add_turn("assistant", "b")
    cm.add_turn("user", "c")
    cm.clear()
    assert cm.get_context() == []


def test_invalid_role_raises():
    cm = ConversationalMemory()
    with pytest.raises(ValueError, match="role must be one of"):
        cm.add_turn("system", "bad role")


def test_formatted_context_format():
    cm = ConversationalMemory()
    cm.add_turn("user", "What is RAG?")
    cm.add_turn("assistant", "RAG stands for retrieval-augmented generation.")
    text = cm.get_formatted_context()
    assert "User:" in text
    assert "Assistant:" in text


def test_empty_context_format():
    cm = ConversationalMemory()
    assert cm.get_formatted_context() == ""


def test_timestamp_is_iso8601():
    cm = ConversationalMemory()
    cm.add_turn("user", "hello")
    ts = cm.get_context()[0]["timestamp"]
    parsed = datetime.fromisoformat(ts)
    assert parsed is not None
