"""Query rewriting: context-aware disambiguation + HyDE for retrieval."""

import logging
from datetime import datetime, timezone

from tenacity import retry, stop_after_attempt, wait_fixed

from agent.state import AgentState

logger = logging.getLogger(__name__)

_QUESTION_WORDS = {"how", "what", "why", "explain", "describe", "when", "where", "who", "which"}

_CONTEXT_PROMPT = """Given this conversation context:
{context}

Rewrite the following query to be fully self-contained (no pronouns, no ambiguous references).
Return only the rewritten query, nothing else.

Query: {query}"""

_HYDE_PROMPT = """Write a short passage from an academic AI/ML research paper that would directly answer the following question. Be technical and specific. Write 2-3 sentences only.

Question: {query}"""


def _is_question(query: str) -> bool:
    first_word = query.strip().split()[0].lower().rstrip("?,") if query.strip() else ""
    return first_word in _QUESTION_WORDS


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def _llm_call(llm_client, prompt: str, max_tokens: int) -> str:
    response = llm_client.chat.completions.create(
        model="mistral-small-latest",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def rewrite_query(state: AgentState, llm_client) -> str:
    """Rewrite the user query using conversation context and optionally apply HyDE.

    Args:
        state: Current agent state.
        llm_client: Anthropic client with .messages.create().

    Returns:
        The rewritten query string (or the original if no rewriting is needed).
    """
    query = state["query"]
    context = state.get("conversation_context", "")
    action = state.get("action", "RETRIEVE")
    hyde_applied = False

    if not query.strip():
        return query

    # Step 1: context-aware rewriting
    rewritten = query
    if context.strip():
        try:
            rewritten = _llm_call(
                llm_client,
                _CONTEXT_PROMPT.format(context=context, query=query),
                max_tokens=150,
            )
        except Exception as exc:
            logger.warning("Context rewriting failed: %s", exc)

    # Step 2: HyDE for retrieval-bound question queries
    if action == "RETRIEVE" and _is_question(query):
        try:
            hypothetical = _llm_call(
                llm_client,
                _HYDE_PROMPT.format(query=rewritten),
                max_tokens=200,
            )
            rewritten = rewritten + "\n\n[HYPOTHETICAL CONTEXT]\n" + hypothetical
            hyde_applied = True
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)

    if "trace" in state:
        state["trace"].append({
            "node": "query_rewriter",
            "input_query": query,
            "output_query": rewritten[:200],
            "hyde_applied": hyde_applied,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    return rewritten
