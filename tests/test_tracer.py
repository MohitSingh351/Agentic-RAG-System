"""Tests for observability/tracer.py."""

import json
import threading
from datetime import datetime
from unittest.mock import patch

import pytest

from observability.tracer import Tracer


def _make_state(**overrides) -> dict:
    base = {"action": "RETRIEVE", "confidence": 0.85, "answer": "Test answer.", "trace": []}
    return {**base, **overrides}


def test_log_appends_entry():
    t = Tracer("sess-001", "What is RAG?")
    t.log("router", {"action": "RETRIEVE"})
    assert len(t._entries) == 1
    entry = t._entries[0]
    assert entry["node"] == "router"
    assert entry["session_id"] == "sess-001"
    assert entry["query"] == "What is RAG?"
    assert entry["action"] == "RETRIEVE"


def test_log_adds_timestamp():
    t = Tracer("sess-002", "test query")
    t.log("retrieve", {"num_results": 5})
    entry = t._entries[0]
    # Should not raise
    datetime.fromisoformat(entry["timestamp"])


def test_finalize_writes_jsonl(tmp_path):
    with patch("observability.tracer._TRACES_DIR", str(tmp_path)):
        t = Tracer("sess-003", "test query")
        t.log("router", {"action": "RETRIEVE"})
        t.finalize(_make_state())
        out_file = tmp_path / "sess-003.jsonl"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2  # router + FINAL


def test_finalize_summary_entry(tmp_path):
    with patch("observability.tracer._TRACES_DIR", str(tmp_path)):
        t = Tracer("sess-004", "test query")
        t.finalize(_make_state(answer="Hello world"))
        data = json.loads((tmp_path / "sess-004.jsonl").read_text())
        final = data[-1]
        assert final["node"] == "FINAL"
        assert final["action"] == "RETRIEVE"
        assert final["confidence"] == 0.85
        assert final["answer_length"] == len("Hello world")


def test_print_trace_in_debug_mode(capsys):
    t = Tracer("sess-005", "test", debug=True)
    t.log("router", {"action": "RETRIEVE"})
    t.print_trace()
    captured = capsys.readouterr()
    assert "router" in captured.out
    assert "---" in captured.out


def test_print_trace_silent_when_not_debug(capsys):
    t = Tracer("sess-006", "test", debug=False)
    t.log("router", {"action": "RETRIEVE"})
    t.print_trace()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_trace_returns_entries(tmp_path):
    with patch("observability.tracer._TRACES_DIR", str(tmp_path)):
        t = Tracer("sess-007", "test")
        t.log("router", {"action": "RETRIEVE"})
        t.finalize(_make_state())
        entries = Tracer.load_trace("sess-007")
    assert isinstance(entries, list)
    assert len(entries) == 2
    assert entries[0]["node"] == "router"


def test_load_trace_missing_file(tmp_path):
    with patch("observability.tracer._TRACES_DIR", str(tmp_path)):
        with pytest.raises(FileNotFoundError):
            Tracer.load_trace("nonexistent-session")


def test_thread_safety():
    t = Tracer("sess-008", "test")
    errors = []

    def log_entry(i: int) -> None:
        try:
            t.log(f"node_{i}", {"index": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=log_entry, args=(i,)) for i in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors
    assert len(t._entries) == 10
