"""LLM-based action router: classifies queries into 5 action labels."""

import logging
from datetime import datetime, timezone

from agent.state import AgentState

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"RETRIEVE", "USE_TOOL", "CLARIFY", "REFUSE", "ANSWER"}

_SYSTEM_PROMPT = """You are a routing agent for an AI/ML research paper Q&A system.
The corpus contains ~70 recent arXiv papers in cs.AI from the last 90 days.

Decide the best action for the user query:

RETRIEVE   - The question is answerable from AI/ML research papers in the corpus.
USE_TOOL   - The question asks about very recent work (today/this week), a specific paper by ID,
             or information the corpus likely does not contain — use arXiv live search.
CLARIFY    - The question is ambiguous and could mean multiple very different things.
             Ask one focused clarifying question.
REFUSE     - The question is outside AI/ML research entirely
             (cooking, finance, personal questions, general knowledge).
ANSWER     - The answer is already known from conversation history. No retrieval needed.

Rules:
- Default to RETRIEVE for most AI/ML questions.
- Only CLARIFY if ambiguity would meaningfully change the answer.
- Only REFUSE for clearly out-of-scope questions.
- Return ONLY one word: RETRIEVE, USE_TOOL, CLARIFY, REFUSE, or ANSWER."""


def route_query(state: AgentState, llm_client) -> str:
    """Classify the query into one of 5 action labels.

    Args:
        state: Current agent state with rewritten_query, semantic_facts, similar_episodes.
        llm_client: Anthropic client.

    Returns:
        One of: RETRIEVE, USE_TOOL, CLARIFY, REFUSE, ANSWER.
    """
    query = state.get("rewritten_query") or state.get("query", "")
    semantic_facts = state.get("semantic_facts", [])
    similar_episodes = state.get("similar_episodes", [])

    user_content_parts = [f"Query: {query}"]
    if semantic_facts:
        facts_str = "; ".join(f"{f['key']}={f['value']}" for f in semantic_facts[:5])
        user_content_parts.append(f"User context: {facts_str}")
    if similar_episodes:
        ep = similar_episodes[0]
        user_content_parts.append(
            f"Note: The user asked a very similar question before: '{ep['query']}'. "
            "Consider returning ANSWER if the context is sufficient."
        )

    user_content_parts.append("Respond with exactly one word.")
    user_message = "\n".join(user_content_parts)

    raw_response = ""
    defaulted = False
    try:
        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_response = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Router LLM call failed: %s", exc)
        raw_response = "RETRIEVE"
        defaulted = True

    action = raw_response.upper().strip(".,;!?")
    if action not in VALID_ACTIONS:
        logger.warning("Router returned invalid action %r, defaulting to RETRIEVE", action)
        action = "RETRIEVE"
        defaulted = True

    if "trace" in state:
        state["trace"].append({
            "node": "router",
            "raw_response": raw_response,
            "action": action,
            "defaulted": defaulted,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    return action
