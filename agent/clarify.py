"""Clarification question generator."""

import logging
from datetime import datetime, timezone

from agent.state import AgentState

logger = logging.getLogger(__name__)

_FALLBACK = "Could you please clarify what specific aspect you're interested in?"

_CLARIFY_PROMPT = """The user asked: '{query}'

This query is ambiguous. Generate ONE focused clarification question (not multiple questions) that would help determine the user's intent. The question must be under 30 words.

Return only the clarification question, nothing else."""


def generate_clarification(state: AgentState, llm_client) -> str:
    """Generate a single focused clarification question.

    Args:
        state: Current agent state.
        llm_client: Anthropic client.

    Returns:
        A clarification question string ending with '?'.
    """
    query = state.get("rewritten_query") or state.get("query", "")

    try:
        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": _CLARIFY_PROMPT.format(query=query)}],
        )
        raw = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Clarification LLM call failed: %s", exc)
        raw = _FALLBACK

    # Extract the question sentence: find first '?', then look back for sentence start
    if "?" in raw:
        q_pos = raw.index("?")
        before = raw[:q_pos]
        # Find the start of the sentence containing '?' by looking back for '. ' or ': '
        start = 0
        for sep in (". ", ": "):
            idx = before.rfind(sep)
            if idx != -1:
                start = max(start, idx + len(sep))
        question = raw[start: q_pos + 1].strip()
    else:
        question = raw.strip() + "?"

    if "trace" in state:
        state["trace"].append({
            "node": "clarify",
            "query": query,
            "clarification": question,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    return question
