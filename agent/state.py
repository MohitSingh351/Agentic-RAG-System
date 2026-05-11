"""AgentState TypedDict — the contract between all LangGraph nodes."""

from typing import TypedDict


class AgentState(TypedDict):
    """State that flows through every node in the agent graph."""

    query: str
    rewritten_query: str
    action: str  # RETRIEVE | USE_TOOL | CLARIFY | REFUSE | ANSWER
    retrieval_results: list[dict]
    tool_result: str
    answer: str
    clarification_question: str
    refusal_message: str
    confidence: float
    conversation_context: str
    semantic_facts: list[dict]
    similar_episodes: list[dict]
    trace: list[dict]
    session_id: str
    debug: bool
