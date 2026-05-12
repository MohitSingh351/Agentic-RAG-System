"""LangGraph StateGraph assembly for the agentic RAG system."""

import logging
import os
import uuid
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agent.answer import generate_answer
from agent.clarify import generate_clarification
from agent.query_rewriter import rewrite_query
from agent.refuse import CONFIDENCE_THRESHOLD, generate_refusal
from agent.router import route_query
from agent.state import AgentState
from agent.tools import arxiv_search
from memory.conversational import ConversationalMemory
from memory.episodic import find_similar, log_episode
from memory.semantic import extract_and_store, get_all_facts
from retrieval.hybrid import hybrid_retrieve
from retrieval.rerank import rerank

load_dotenv()
logger = logging.getLogger(__name__)

_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

_conv_memory = ConversationalMemory(max_turns=6)
_llm_client: OpenAI | None = None


def _get_llm() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url=_MISTRAL_BASE_URL,
        )
    return _llm_client


# --- Node functions ---

def rewrite_node(state: AgentState) -> AgentState:
    """Load memory context and rewrite the query."""
    state["conversation_context"] = _conv_memory.get_formatted_context()
    state["semantic_facts"] = get_all_facts()
    state["similar_episodes"] = find_similar(state["query"], threshold=0.8)
    # Set action to RETRIEVE as default so HyDE can fire in the rewriter
    if not state.get("action"):
        state["action"] = "RETRIEVE"
    state["rewritten_query"] = rewrite_query(state, _get_llm())
    return state


def router_node(state: AgentState) -> AgentState:
    """Classify the query into an action."""
    state["action"] = route_query(state, _get_llm())
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Hybrid retrieval + cross-encoder reranking."""
    query = state.get("rewritten_query") or state["query"]
    candidates = hybrid_retrieve(query, top_k=20)
    ranked = rerank(query, candidates, top_n=5)
    state["retrieval_results"] = ranked

    scores = [r.get("score", 0.0) for r in ranked[:3]]
    state["confidence"] = min(1.0, max(0.0, sum(scores) / len(scores))) if scores else 0.0

    if "trace" in state:
        state["trace"].append({
            "node": "retrieve",
            "num_results": len(ranked),
            "top_score": ranked[0]["score"] if ranked else 0.0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
    return state


def tool_node(state: AgentState) -> AgentState:
    """Call arXiv API and store result."""
    query = state.get("rewritten_query") or state["query"]
    # Strip HyDE suffix for tool call
    clean_query = query.split("\n\n[HYPOTHETICAL CONTEXT]")[0].strip()
    state["tool_result"] = arxiv_search.invoke({"query": clean_query, "max_results": 5})
    if "trace" in state:
        state["trace"].append({
            "node": "use_tool",
            "query": clean_query,
            "result_length": len(state["tool_result"]),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
    return state


def answer_node(state: AgentState) -> AgentState:
    """Generate answer and update memory."""
    result = generate_answer(state, _get_llm())
    state["answer"] = result["answer"]
    state["confidence"] = result["confidence"]

    # Update memory
    _conv_memory.add_turn("user", state["query"])
    _conv_memory.add_turn("assistant", state["answer"])
    log_episode(
        query=state["query"],
        answer=state["answer"],
        sources=[r["chunk_id"] for r in state.get("retrieval_results", [])[:5]],
    )
    try:
        extract_and_store(state["query"], _get_llm())
    except Exception as exc:
        logger.debug("Semantic extraction skipped: %s", exc)
    return state


def clarify_node(state: AgentState) -> AgentState:
    """Generate a clarification question."""
    state["clarification_question"] = generate_clarification(state, _get_llm())
    _conv_memory.add_turn("user", state["query"])
    _conv_memory.add_turn("assistant", state["clarification_question"])
    return state


def refuse_node(state: AgentState) -> AgentState:
    """Generate a refusal message (hard or soft)."""
    refusal_type = "soft" if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD and state.get("retrieval_results") else "hard"
    state["refusal_message"] = generate_refusal(state, refusal_type)
    return state


# --- Routing functions ---

def _route_after_router(state: AgentState) -> str:
    return state["action"]


def _route_after_retrieve(state: AgentState) -> str:
    if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
        return "refuse"
    return "answer"


# --- Build graph ---

def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("router_node", router_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("clarify_node", clarify_node)
    graph.add_node("refuse_node", refuse_node)

    graph.set_entry_point("rewrite_node")
    graph.add_edge("rewrite_node", "router_node")

    graph.add_conditional_edges("router_node", _route_after_router, {
        "RETRIEVE": "retrieve_node",
        "USE_TOOL": "tool_node",
        "CLARIFY":  "clarify_node",
        "REFUSE":   "refuse_node",
        "ANSWER":   "answer_node",
    })

    graph.add_conditional_edges("retrieve_node", _route_after_retrieve, {
        "answer": "answer_node",
        "refuse": "refuse_node",
    })

    graph.add_edge("tool_node",    "answer_node")
    graph.add_edge("answer_node",  END)
    graph.add_edge("clarify_node", END)
    graph.add_edge("refuse_node",  END)

    return graph


_graph = _build_graph()
app = _graph.compile()


def run_agent(query: str, debug: bool = False) -> AgentState:
    """Run the agent on a single query.

    Args:
        query: The user's question.
        debug: Whether to enable debug trace output.

    Returns:
        Final AgentState after the graph has completed.
    """
    initial_state: AgentState = {
        "query": query,
        "rewritten_query": "",
        "action": "",
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
        "session_id": str(uuid.uuid4()),
        "debug": debug,
    }
    return app.invoke(initial_state)
