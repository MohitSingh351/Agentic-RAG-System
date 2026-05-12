"""Answer generation with 2-pass contradiction detection."""

import json
import logging
from datetime import datetime, timezone

from agent.state import AgentState

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """You are a research assistant specializing in AI/ML research.
Based ONLY on the following source passages, answer the user's question.
If the context is insufficient, explicitly say so.

Sources:
{context}

Question: {question}

Provide a clear, concise answer with references to the source papers where relevant."""

_CONTRADICTION_PROMPT = """Check whether any part of the following answer contradicts the source passages provided.

Answer:
{answer}

Sources:
{context}

Return ONLY valid JSON in this format:
{{"contradictions": [{{"claim": "...", "issue": "..."}}], "revised_answer": "..."}}

If no contradictions, return:
{{"contradictions": [], "revised_answer": "{answer_escaped}"}}"""


def _format_context(retrieval_results: list[dict], max_chunks: int = 5) -> str:
    parts = []
    for i, r in enumerate(retrieval_results[:max_chunks], start=1):
        meta = r.get("metadata", {})
        title = meta.get("title", "Unknown")
        section = meta.get("section", "")
        parts.append(f"[Source {i}] {title} ({section}):\n{r['text'][:600]}")
    return "\n\n".join(parts)


def _compute_confidence(retrieval_results: list[dict]) -> float:
    scores = [r.get("score", 0.0) for r in retrieval_results[:3]]
    if not scores:
        return 0.0
    return min(1.0, max(0.0, sum(scores) / len(scores)))


def generate_answer(state: AgentState, llm_client) -> dict:
    """Generate a grounded answer with 2-pass contradiction checking.

    Args:
        state: Current agent state.
        llm_client: Anthropic client.

    Returns:
        Dict with 'answer' (str) and 'confidence' (float).
    """
    query = state.get("rewritten_query") or state.get("query", "")
    retrieval_results = state.get("retrieval_results", [])
    tool_result = state.get("tool_result", "")

    # Choose context source
    if retrieval_results:
        context = _format_context(retrieval_results)
        source_chunks = [r["chunk_id"] for r in retrieval_results[:5]]
        confidence = _compute_confidence(retrieval_results)
    elif tool_result:
        context = tool_result
        source_chunks = ["arXiv API"]
        confidence = 0.6
    else:
        answer = "I could not find relevant information to answer your question. Please try rephrasing or check arXiv directly."
        if "trace" in state:
            state["trace"].append({
                "node": "answer",
                "confidence": 0.0,
                "contradiction_found": False,
                "source_chunks": [],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        return {"answer": answer, "confidence": 0.0}

    # Pass 1: Generate initial answer
    try:
        pass1 = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": _ANSWER_PROMPT.format(context=context, question=query),
            }],
        )
        initial_answer = pass1.content[0].text.strip()
    except Exception as exc:
        logger.warning("Answer generation failed: %s", exc)
        return {"answer": "An error occurred while generating the answer.", "confidence": 0.0}

    # Pass 2: Contradiction check
    contradiction_found = False
    final_answer = initial_answer
    escaped = initial_answer.replace('"', '\\"').replace('\n', ' ')
    try:
        pass2 = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": _CONTRADICTION_PROMPT.format(
                    answer=initial_answer,
                    context=context,
                    answer_escaped=escaped,
                ),
            }],
        )
        raw = pass2.content[0].text.strip()
        data = json.loads(raw)
        if data.get("contradictions"):
            contradiction_found = True
            final_answer = data.get("revised_answer", initial_answer)
    except (json.JSONDecodeError, Exception) as exc:
        logger.debug("Contradiction check skipped: %s", exc)

    if "trace" in state:
        state["trace"].append({
            "node": "answer",
            "confidence": confidence,
            "contradiction_found": contradiction_found,
            "source_chunks": source_chunks,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    return {"answer": final_answer, "confidence": confidence}
