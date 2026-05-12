"""Tests for cli.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

import cli


def _make_state(**kwargs) -> dict:
    base = {
        "action": "RETRIEVE",
        "answer": "",
        "clarification_question": "",
        "refusal_message": "",
        "confidence": 0.8,
        "trace": [],
    }
    return {**base, **kwargs}


@patch("builtins.input", side_effect=["/quit"])
@patch("agent.graph.run_agent")
@patch("agent.graph._conv_memory")
def test_cli_quit_command(mock_mem, mock_run, mock_input):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
    assert exc_info.value.code == 0


@patch("builtins.input", side_effect=["/clear", "/quit"])
@patch("agent.graph.run_agent")
@patch("agent.graph._conv_memory")
def test_cli_clear_command(mock_mem, mock_run, mock_input):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit):
            cli.main()
    mock_mem.clear.assert_called()


@patch("builtins.input", side_effect=["What is RAG?", "/quit"])
@patch("agent.graph.run_agent", return_value=_make_state(answer="RAG is retrieval augmented generation."))
@patch("agent.graph._conv_memory")
def test_cli_displays_answer(mock_mem, mock_run, mock_input, capsys):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit):
            cli.main()
    captured = capsys.readouterr()
    assert "RAG is retrieval augmented generation." in captured.out


@patch("builtins.input", side_effect=["Tell me more.", "/quit"])
@patch("agent.graph.run_agent", return_value=_make_state(
    action="CLARIFY",
    clarification_question="Are you asking about transformer architecture?",
))
@patch("agent.graph._conv_memory")
def test_cli_displays_clarification(mock_mem, mock_run, mock_input, capsys):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit):
            cli.main()
    captured = capsys.readouterr()
    assert "Clarifying" in captured.out or "clarif" in captured.out.lower()


@patch("builtins.input", side_effect=["What is pizza?", "/quit"])
@patch("agent.graph.run_agent", return_value=_make_state(
    action="REFUSE",
    refusal_message="I cannot help with that topic.",
))
@patch("agent.graph._conv_memory")
def test_cli_displays_refusal(mock_mem, mock_run, mock_input, capsys):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit):
            cli.main()
    captured = capsys.readouterr()
    assert "cannot help" in captured.out.lower() or "Cannot help" in captured.out


@patch("builtins.input", side_effect=KeyboardInterrupt)
@patch("agent.graph._conv_memory")
def test_cli_keyboard_interrupt_exits_cleanly(mock_mem, mock_input, capsys):
    with patch.object(sys, "argv", ["cli", "--no-color"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
    captured = capsys.readouterr()
    assert "Goodbye" in captured.out
    assert exc_info.value.code == 0
