"""Hard and soft refusal handlers."""

import logging
from datetime import datetime, timezone

from agent.state import AgentState

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.4


def generate_refusal(state: AgentState, refusal_type: str = "hard") -> str:
    """Generate an appropriate refusal message.

    Args:
        state: Current agent state.
        refusal_type: 'hard' (off-topic) or 'soft' (low confidence).

    Returns:
        Refusal message string.
    """
    query = state.get("query", "")
    confidence = state.get("confidence", 0.0)
    retrieval_results = state.get("retrieval_results", [])

    if refusal_type == "soft":
        partial = ""
        if retrieval_results:
            partial = "\n\nHere is what I found (low confidence):\n" + retrieval_results[0]["text"][:300]
        message = (
            f"I found some relevant papers but my confidence is low ({confidence:.0%}). "
            f"This answer may be incomplete — please verify with arXiv directly.{partial}"
        )
    else:
        message = (
            "I'm specialized in cs.AI research papers from the last 90 days. "
            f"I can't help with this type of question. "
            "I can answer questions about AI/ML research, methods, papers, and findings."
        )

    if "trace" in state:
        state["trace"].append({
            "node": "refuse",
            "refusal_type": refusal_type,
            "confidence": confidence,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })

    return message
