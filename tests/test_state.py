"""Tests for agent/state.py."""

import pytest
from agent.state import AgentState


def test_agent_state_is_typeddict():
    assert issubclass(AgentState, dict)


def test_agent_state_all_fields_present():
    state: AgentState = {
        "query": "test",
        "rewritten_query": "test",
        "action": "RETRIEVE",
        "retrieval_results": [],
        "tool_result": "",
        "answer": "",
        "clarification_question": "",
        "refusal_message": "",
        "confidence": 0.0,
        "conversation_context": "",
        "semantic_facts": [],
        "similar_episodes": [],
        "trace": [],
        "session_id": "abc",
        "debug": False,
    }
    assert len(state.keys()) == 15


def test_action_field_accepts_valid_values():
    valid_actions = ["RETRIEVE", "USE_TOOL", "CLARIFY", "REFUSE", "ANSWER"]
    for action in valid_actions:
        state: AgentState = {
            "query": "q", "rewritten_query": "q", "action": action,
            "retrieval_results": [], "tool_result": "", "answer": "",
            "clarification_question": "", "refusal_message": "",
            "confidence": 0.0, "conversation_context": "",
            "semantic_facts": [], "similar_episodes": [],
            "trace": [], "session_id": "s", "debug": False,
        }
        assert state["action"] == action
