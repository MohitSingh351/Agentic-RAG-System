# Agentic RAG System — Detailed Implementation Plan
### Skyclad Ventures AI Engineering Intern Assignment

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Architecture](#2-final-architecture)
3. [Tech Stack Decisions](#3-tech-stack-decisions)
4. [Repository Structure](#4-repository-structure)
5. [Day-by-Day Implementation Plan](#5-day-by-day-implementation-plan)
6. [Component Deep Dives](#6-component-deep-dives)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Observability Design](#8-observability-design)
9. [README & Demo Checklist](#9-readme--demo-checklist)
10. [Risk Register](#10-risk-register)

---

## 1. Project Overview

**Goal**: Build an agentic RAG system over ~60–80 arXiv cs.AI papers (last 90 days) that autonomously decides when to retrieve, use a tool, ask for clarification, refuse, or answer — with memory, non-naive retrieval, evaluation, and full observability.

**Guiding principle**: Every non-trivial choice must have a one-sentence written justification. Depth over breadth. One thing done well with evidence beats five things wired together with no proof they help.

---

## 2. Final Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│             Memory Module               │
│  • Conversational (sliding window)      │
│  • Semantic (extracted facts)           │
│  • Episodic (past query log)            │
└───────────────┬─────────────────────────┘
                │ enriched context
                ▼
┌─────────────────────────────────────────┐
│          Query Rewriter                 │
│  • Rewrites query using memory context  │
│  • Generates HyDE hypothetical doc      │
└───────────────┬─────────────────────────┘
                │ rewritten query
                ▼
┌─────────────────────────────────────────┐
│         Agent Router (LLM)              │
│  Decides: RETRIEVE / USE_TOOL /         │
│           CLARIFY / REFUSE / ANSWER     │
└──┬──────┬──────────┬──────┬─────────────┘
   │      │          │      │
   ▼      ▼          ▼      ▼
RETR   TOOL       CLAR   REFUSE
   │      │
   ▼      ▼
┌─────────────────────────────────────────┐
│        Hybrid Retriever                 │
│  BM25 (sparse) + Dense (vector)         │
│  → Reciprocal Rank Fusion               │
│  → Cross-Encoder Reranker (top 5)       │
└───────────────┬─────────────────────────┘
                │ ranked chunks
                ▼
┌─────────────────────────────────────────┐
│         Answer Generator                │
│  LLM + retrieved context + memory       │
│  + contradiction detection              │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         Observability Layer             │
│  Structured trace per query             │
│  {action, reasoning, chunks, answer}    │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│           Eval Harness                  │
│  10 curated Q&A + ablation scoring      │
└─────────────────────────────────────────┘
```

**Agent loop type**: LangGraph `StateGraph` — explicit state machine, not a chain. Justified by the fact that the loop has 5+ distinct decision branches, each with different node logic and conditional edges.

---

## 3. Tech Stack Decisions

| Component | Choice | Why | Considered & Rejected |
|---|---|---|---|
| Agent framework | LangGraph | Explicit state + conditional edges; needed for 5-branch router | LangChain (too implicit), raw loops (no observability) |
| LLM | Claude Haiku 3.5 or GPT-4o-mini | Cheap (~$1–2 total), fast, capable enough for routing + answering | GPT-4o (expensive), local LLaMA (slow without GPU) |
| Embeddings | `text-embedding-3-small` (OpenAI) | 1536-dim, cheap, strong benchmark scores | `all-MiniLM-L6-v2` (free but weaker), ada-002 (older) |
| Vector DB | ChromaDB (local) | Zero infra setup, persistent to disk, Python-native | Qdrant (better for prod but overkill), Pinecone (SaaS) |
| Sparse retrieval | BM25 (rank-bm25 library) | Standard keyword retrieval; complements dense for exact terms | TF-IDF (weaker), Elasticsearch (infra overhead) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local, measurably improves precision in ablation | Cohere Rerank (paid API) |
| Corpus | arXiv cs.AI, last 90 days, ~70 papers | Topically coherent, arXiv API as natural external tool | Wikipedia (too broad), textbooks (PDFs hard to get) |
| External tool | arXiv API | Extends corpus on demand; same domain as corpus; free | Web search (noisy), calculator (irrelevant domain) |
| Interface | CLI with `--debug` flag | Clarity > prettiness; debug flag exposes traces | Streamlit (pretty but hides reasoning), Jupyter (messy) |
| PDF parsing | PyMuPDF (`fitz`) | Preserves layout, fast, handles multi-column | pdfplumber (slow), PyPDF2 (lossy) |

---

## 4. Repository Structure

```
agentic-rag/
│
├── README.md                    # Setup + architecture + decisions log
├── requirements.txt
├── .env.example
│
├── ingest/
│   ├── fetch_arxiv.py           # Download papers via arXiv API
│   ├── parse_pdfs.py            # PyMuPDF → raw text per section
│   ├── chunk.py                 # Semantic chunking with metadata
│   └── embed_and_store.py       # Embed chunks → ChromaDB
│
├── retrieval/
│   ├── dense.py                 # ChromaDB vector search
│   ├── sparse.py                # BM25 index + search
│   ├── hybrid.py                # RRF fusion of dense + sparse
│   └── rerank.py                # Cross-encoder reranking
│
├── memory/
│   ├── conversational.py        # Sliding window of last N turns
│   ├── semantic.py              # Extracted user facts (SQLite)
│   └── episodic.py              # Past query log + fuzzy match
│
├── agent/
│   ├── state.py                 # LangGraph state definition (TypedDict)
│   ├── router.py                # LLM-based action router
│   ├── query_rewriter.py        # Query rewriting + HyDE
│   ├── tools.py                 # arXiv API tool
│   ├── answer.py                # Answer generation with context
│   ├── clarify.py               # Clarification question logic
│   ├── refuse.py                # Refusal + reason generation
│   └── graph.py                 # LangGraph StateGraph assembly
│
├── observability/
│   └── tracer.py                # Structured trace logger (JSONL)
│
├── eval/
│   ├── questions.json           # 10 curated eval questions
│   ├── run_eval.py              # Run all questions, score, log
│   └── ablation.py              # Compare retrieval strategies
│
├── cli.py                       # Entry point: python cli.py [--debug]
│
└── tests/
    ├── test_chunking.py
    ├── test_hybrid_retrieval.py
    └── test_router.py
```

---

## 5. Day-by-Day Implementation Plan

---

### Day 1 — Corpus Ingestion

**Goal**: Have ~70 papers downloaded, parsed, chunked, embedded, and stored in ChromaDB.

#### Step 1.1 — Fetch papers from arXiv API

File: `ingest/fetch_arxiv.py`

```
Logic:
- Query arXiv API with category=cs.AI, date range = last 90 days
- Fetch metadata: paper_id, title, authors, abstract, published date
- Download PDFs to /data/pdfs/
- Save metadata to /data/metadata.json
- Target: 60–80 papers
```

Key details:
- Use the `arxiv` Python library (official wrapper)
- Rate-limit requests (3s sleep between batches)
- Store arXiv ID — you'll use it as document ID in ChromaDB

```python
# Example metadata structure per paper
{
  "paper_id": "2401.12345",
  "title": "...",
  "authors": ["..."],
  "abstract": "...",
  "published": "2024-01-15",
  "pdf_path": "data/pdfs/2401.12345.pdf"
}
```

#### Step 1.2 — Parse PDFs

File: `ingest/parse_pdfs.py`

```
Logic:
- Use PyMuPDF (fitz) to extract text page by page
- Detect section headings (heuristic: short line, bold/larger font, or ALL CAPS)
- Group text into sections: Abstract, Introduction, Method, Experiments, etc.
- Output: list of {paper_id, section_title, section_text, page_number}
```

Why sections matter: they are natural semantic units. A fixed 512-token chunk often splits mid-argument. A section chunk preserves coherent thought.

#### Step 1.3 — Semantic Chunking

File: `ingest/chunk.py`

```
Strategy:
- For each section, if len(text) < 800 tokens → keep as one chunk
- If len(text) > 800 tokens → split at paragraph boundaries (double newline)
- Never split mid-sentence
- Each chunk carries metadata:
    {chunk_id, paper_id, title, section, page_start, char_start, text}
- Target chunk size: 400–700 tokens
```

Why this over fixed-size: paragraph splits preserve argument structure. A chunk that starts "However, our approach differs..." without preceding context is useless. Paragraphs are rarely orphaned.

#### Step 1.4 — Embed and Store

File: `ingest/embed_and_store.py`

```
Logic:
- Batch all chunks (batch size 100 to avoid API rate limits)
- Embed each chunk text with text-embedding-3-small
- Store in ChromaDB collection "arxiv_papers":
    - document = chunk text
    - embedding = chunk vector
    - metadata = {paper_id, title, section, page_start}
- Also build BM25 index over all chunk texts → save to /data/bm25_index.pkl
- Log: total chunks stored, total papers, avg chunks per paper
```

**Day 1 checkpoint**: Run `python ingest/embed_and_store.py`. Verify ChromaDB shows ~500–1000 chunks. BM25 index pickled successfully.

---

### Day 2 — Retrieval Pipeline

**Goal**: Implement hybrid retrieval (BM25 + dense) with RRF fusion and cross-encoder reranking.

#### Step 2.1 — Dense Retrieval

File: `retrieval/dense.py`

```python
def dense_search(query: str, top_k: int = 20) -> list[dict]:
    # Embed query with text-embedding-3-small
    # Query ChromaDB collection
    # Return list of {chunk_id, text, metadata, score}
```

#### Step 2.2 — Sparse Retrieval (BM25)

File: `retrieval/sparse.py`

```python
def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    # Load BM25 index from pickle
    # Tokenize query
    # Score all documents
    # Return top-k with scores
```

#### Step 2.3 — Hybrid Fusion (RRF)

File: `retrieval/hybrid.py`

```
RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k = 60 (standard constant), rank_i = rank of document d in list i

Logic:
- Run dense_search(query, top_k=20)
- Run bm25_search(query, top_k=20)
- Merge both ranked lists using RRF
- Return unified top-20 by RRF score
```

Why RRF over score normalization: scores from BM25 and cosine similarity are on different scales and distributions. RRF is rank-based so it is scale-invariant and empirically robust.

#### Step 2.4 — Cross-Encoder Reranking

File: `retrieval/rerank.py`

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    # Score each (query, chunk_text) pair together
    # Sort by cross-encoder score descending
    # Return top_n
```

Why cross-encoder over bi-encoder for reranking: cross-encoders see the query and document together, giving much better relevance judgments. Too slow for first-stage retrieval but fine for rescoring 20 candidates.

**Day 2 checkpoint**: Write a quick test — query "attention mechanism in transformers" → verify top-5 returned chunks are actually about attention. Run an ablation preview: compare dense-only vs hybrid vs hybrid+rerank on 3 sample queries.

---

### Day 3 — Memory Module

**Goal**: Implement all three memory types with clear separation and distinct roles.

#### Step 3.1 — Conversational Memory

File: `memory/conversational.py`

```python
class ConversationalMemory:
    def __init__(self, max_turns: int = 6):
        self.turns = []  # list of {role, content}

    def add(self, role: str, content: str): ...
    def get_context(self) -> list[dict]: ...  # returns last max_turns entries
    def clear(self): ...
```

Sliding window of last 6 turns (3 user + 3 assistant). Older turns are dropped. This is the baseline — every RAG system should have this minimum.

#### Step 3.2 — Semantic Memory

File: `memory/semantic.py`

```
Purpose: Remember persistent facts about the user and session.
Storage: SQLite table (key, value, confidence, timestamp)

Facts are extracted by an LLM call after each turn:
  Prompt: "Given this conversation turn, extract any persistent facts
           about the user's research interests, expertise level, or
           topic focus. Return JSON list of {key, value} or []."

Example extracted facts:
  {"key": "research_interest", "value": "reinforcement learning"}
  {"key": "expertise_level",   "value": "PhD student"}
  {"key": "topic_focus",       "value": "model alignment"}

Usage: Facts are injected into retrieval queries and answer prompts.
       "Since you are focused on RL, here are the most relevant papers..."
```

Why this matters: if a user said "I work on LLM safety" three turns ago, the current retrieval should be biased toward safety papers even if the new query is generic ("show me recent work").

#### Step 3.3 — Episodic Memory

File: `memory/episodic.py`

```
Purpose: Remember specific past interactions.
Storage: JSONL file — each line is {query, action, answer, timestamp, chunk_ids}

On new query:
  1. Load last 50 episodes
  2. Compute string similarity (difflib or embedding cosine) to current query
  3. If similarity > 0.8 → surface the past answer as context:
     "You asked something similar before: [past answer]. Building on that..."

This prevents re-retrieving the same chunks for repeated questions
and gives the agent an honest episodic sense of "I have seen this before."
```

**Day 3 checkpoint**: Simulate a 5-turn conversation. Verify conversational memory truncates correctly. Verify semantic memory extracts "interest in NLP" after a relevant turn. Verify episodic memory surfaces a match on a near-duplicate query.

---

### Day 4 — Agent Router + Query Rewriter

**Goal**: Build the brain — the LLM router that decides what to do, and the query rewriter that makes queries better before retrieval.

#### Step 4.1 — Query Rewriter

File: `agent/query_rewriter.py`

```
Two rewriting strategies:

1. Context-aware rewriting:
   Uses conversational + semantic memory to disambiguate the query.
   "transformers" → "transformer architecture for NLP (attention, BERT, GPT)"
   because semantic memory says user is an NLP researcher.

2. HyDE (Hypothetical Document Embeddings):
   Prompt LLM: "Write a short passage from an academic paper that would
               answer this question: {query}"
   Embed the hypothetical passage instead of the raw query.
   Better for technical questions where the query phrasing differs from
   how papers phrase the answer.

Config: Use HyDE for questions phrased as questions ("How does X work?")
        Use context-rewriting for follow-up queries ("What about efficiency?")
```

#### Step 4.2 — Agent Router

File: `agent/router.py`

This is the most important prompt in the system. Write it carefully.

```
ROUTER_PROMPT:

You are a routing agent for a research paper Q&A system.
The corpus contains ~70 recent arXiv papers in AI/ML.

Given a user query and conversation context, decide the best action:

RETRIEVE   - The question is answerable from AI/ML research papers in the corpus.
USE_TOOL   - The question asks about very recent work, a specific paper by ID,
             or information the corpus likely does not contain — use arXiv API.
CLARIFY    - The question is ambiguous and could mean multiple very different things.
             Ask one focused clarifying question.
REFUSE     - The question is outside AI/ML research entirely
             (coding help, general knowledge, personal questions).
ANSWER     - The answer is already known from conversation history.
             No retrieval needed.

Rules:
- Default to RETRIEVE for most AI/ML questions.
- Only CLARIFY if ambiguity would meaningfully change the answer.
- Only REFUSE for clearly out-of-scope questions.
- Return ONLY one word: RETRIEVE, USE_TOOL, CLARIFY, REFUSE, or ANSWER.
```

The router output is one word. Parse it, validate it, default to RETRIEVE on invalid output.

#### Step 4.3 — LangGraph State Definition

File: `agent/state.py`

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str                        # original user query
    rewritten_query: str              # after query rewriting
    action: str                       # router decision
    retrieved_chunks: list[dict]      # from hybrid retriever
    tool_output: Optional[str]        # from arXiv API
    clarification_question: Optional[str]
    answer: Optional[str]
    refusal_reason: Optional[str]
    memory_context: dict              # {conversational, semantic, episodic}
    trace: list[dict]                 # observability log
```

#### Step 4.4 — LangGraph Graph Assembly

File: `agent/graph.py`

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("route",         route_node)
graph.add_node("retrieve",      retrieve_node)
graph.add_node("use_tool",      tool_node)
graph.add_node("clarify",       clarify_node)
graph.add_node("refuse",        refuse_node)
graph.add_node("answer",        answer_node)

graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "route")

graph.add_conditional_edges("route", lambda s: s["action"], {
    "RETRIEVE": "retrieve",
    "USE_TOOL": "use_tool",
    "CLARIFY":  "clarify",
    "REFUSE":   "refuse",
    "ANSWER":   "answer",
})

graph.add_edge("retrieve", "answer")
graph.add_edge("use_tool", "answer")
graph.add_edge("clarify",  END)
graph.add_edge("refuse",   END)
graph.add_edge("answer",   END)

app = graph.compile()
```

**Day 4 checkpoint**: Run the agent on 5 manual queries. Verify: an AI/ML question routes to RETRIEVE, "what is the weather" routes to REFUSE, "tell me about transformers" routes to CLARIFY (ambiguous), a recent-paper question routes to USE_TOOL.

---

### Day 5 — External Tool + Failure Handling

**Goal**: Wire the arXiv API tool and implement all four failure modes explicitly.

#### Step 5.1 — arXiv Tool

File: `agent/tools.py`

```python
import arxiv

def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for papers matching the query.
    Returns formatted string of results for LLM consumption.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for paper in client.results(search):
        results.append(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a.name for a in paper.authors)}\n"
            f"Published: {paper.published.date()}\n"
            f"Abstract: {paper.summary[:400]}...\n"
            f"URL: {paper.entry_id}\n"
        )
    return "\n---\n".join(results) if results else "No papers found."
```

The `use_tool` node in the graph calls this and stores the output in `state["tool_output"]`, which is then passed to the answer generator.

#### Step 5.2 — Failure Mode Handlers

**Failure 1: Low retrieval confidence**

In `agent/retrieve.py`:
```
After retrieval, check top chunk similarity score.
If max score < 0.35 (threshold tuned via experimentation):
  → Log "low_confidence_retrieval" in trace
  → Either route to USE_TOOL, or include disclaimer in answer:
    "I found limited relevant material in the corpus.
     This answer may be incomplete. Consider checking arXiv directly."
```

**Failure 2: Contradicting context**

In `agent/answer.py`:
```
Before generating the answer, run a quick LLM check:
  Prompt: "Do these two passages contradict each other? Answer YES or NO and explain."
  If YES:
    → Include in answer: "Note: I found conflicting information.
      Paper A claims X, while Paper B claims Y. Here is my best synthesis..."
  Log contradiction detection result in trace.
```

**Failure 3: Ambiguous query**

In `agent/clarify.py`:
```
Generate ONE focused question. Not multiple.

Bad:  "What do you mean? What topic? What timeframe?"
Good: "Are you asking about the transformer architecture for NLP,
       or a different meaning of the word 'transformer'?"

The clarification question is returned to the user and the turn ends.
Their response is added to conversational memory and the loop restarts.
```

**Failure 4: Out-of-domain query**

In `agent/refuse.py`:
```
Two levels of refusal:

Hard refuse (router directly chooses REFUSE):
  "What is the capital of France?" →
  "This system is designed for AI/ML research questions from arXiv papers.
   I cannot help with general knowledge questions."

Soft refuse (router chose RETRIEVE but retrieval returned no relevant chunks):
  "What is the best Python IDE?" →
  "While I can discuss software engineering in the context of ML research,
   this question is outside the scope of the arXiv corpus."
```

**Day 5 checkpoint**: Test all four failure modes manually. Verify each produces a different, appropriately worded response. Verify trace logs the failure reason with the correct label.

---

### Day 6 — Evaluation Harness + Ablation

**Goal**: Run 10 curated questions, score the system, and prove hybrid+rerank beats dense-only.

#### Step 6.1 — Question Design

File: `eval/questions.json`

```json
[
  {
    "id": "q01",
    "question": "What are the main approaches to improving LLM reasoning without fine-tuning?",
    "expected_action": "RETRIEVE",
    "expected_topics": ["chain-of-thought", "prompting", "in-context learning"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q02",
    "question": "Explain the role of KV cache in transformer inference efficiency.",
    "expected_action": "RETRIEVE",
    "expected_topics": ["KV cache", "attention", "inference"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q03",
    "question": "What is the difference between RLHF and DPO for alignment?",
    "expected_action": "RETRIEVE",
    "expected_topics": ["RLHF", "DPO", "alignment"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q04",
    "question": "Summarize the most recent work on mixture-of-experts architectures.",
    "expected_action": "USE_TOOL",
    "expected_topics": ["MoE", "mixture of experts"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q05",
    "question": "What papers have been published on speculative decoding in the last month?",
    "expected_action": "USE_TOOL",
    "expected_topics": ["speculative decoding"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q06",
    "question": "Tell me about transformers.",
    "expected_action": "CLARIFY",
    "expected_topics": [],
    "should_refuse": false,
    "should_clarify": true,
    "clarify_reason": "Ambiguous — architecture? specific model? specific paper?"
  },
  {
    "id": "q07",
    "question": "What is the best way to prepare for a data science interview?",
    "expected_action": "REFUSE",
    "expected_topics": [],
    "should_refuse": true,
    "refuse_reason": "Outside AI/ML research domain"
  },
  {
    "id": "q08",
    "question": "How does the attention mechanism handle positional information in recent models?",
    "expected_action": "RETRIEVE",
    "expected_topics": ["positional encoding", "RoPE", "attention"],
    "should_refuse": false,
    "should_clarify": false
  },
  {
    "id": "q09",
    "question": "What is the recipe for biryani?",
    "expected_action": "REFUSE",
    "expected_topics": [],
    "should_refuse": true,
    "refuse_reason": "Completely out of domain"
  },
  {
    "id": "q10",
    "question": "How do recent papers address the problem of hallucination in LLMs?",
    "expected_action": "RETRIEVE",
    "expected_topics": ["hallucination", "factuality", "grounding"],
    "should_refuse": false,
    "should_clarify": false
  }
]
```

#### Step 6.2 — Scoring Logic

File: `eval/run_eval.py`

```
For each question, score on three dimensions:

1. Action correctness (0 or 1):
   Did the agent take the expected action (RETRIEVE/USE_TOOL/CLARIFY/REFUSE)?

2. Answer quality (0, 1, or 2) — for RETRIEVE and USE_TOOL actions only:
   0 = Answer is wrong or hallucinates
   1 = Answer is partially correct or vague
   2 = Answer is accurate and cites relevant papers

3. Hallucination flag (0 or 1):
   Does the answer cite a paper not in the corpus and not from the tool?
   Manual check — log as 0 (clean) or 1 (hallucination present)

Report: % action accuracy, avg quality score, % hallucination-free responses
```

#### Step 6.3 — Ablation Study

File: `eval/ablation.py`

Run the same 10 questions (or a 5-question subset for speed) against three retrieval configurations:

```
Config A: Dense only  (ChromaDB top-5, no reranking)
Config B: Hybrid      (BM25 + Dense via RRF, no reranking, top-5)
Config C: Hybrid + Rerank (top-5 from top-20 candidates via cross-encoder)

For each config, record:
- Top-5 chunk precision (manually assessed: are the top 5 relevant?)
- Answer quality score (same 0–2 scale)
- Time to retrieve (milliseconds)

Present results as a table in the README.
```

Expected outcome: Config C ≥ B ≥ A on quality, with a small latency cost for C. If results differ, investigate and report honestly — that is more impressive than fabricated results.

**Day 6 checkpoint**: Full eval run complete. Ablation table generated. README decisions log drafted.

---

### Day 7 — Polish, README, Demo Video

**Goal**: Ship a clean, readable repo with a great README and a confident demo.

#### Step 7.1 — CLI Entry Point

File: `cli.py`

```python
import argparse
from agent.graph import app
from memory.conversational import ConversationalMemory
from memory.semantic import SemanticMemory
from memory.episodic import EpisodicMemory
from observability.tracer import Tracer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Print full trace after each query")
    args = parser.parse_args()

    conv_mem = ConversationalMemory(max_turns=6)
    sem_mem  = SemanticMemory()
    ep_mem   = EpisodicMemory()
    tracer   = Tracer(debug=args.debug)

    print("Agentic RAG — arXiv cs.AI Corpus")
    print("Type your question. Ctrl+C to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue

        state = {
            "query": query,
            "memory_context": {
                "conversational": conv_mem.get_context(),
                "semantic":       sem_mem.get_facts(),
                "episodic":       ep_mem.search(query),
            },
            "trace": []
        }

        result = app.invoke(state)

        response = (
            result.get("answer") or
            result.get("clarification_question") or
            result.get("refusal_reason") or
            "No response generated."
        )

        print(f"\nAgent: {response}\n")

        if args.debug:
            tracer.print_trace(result["trace"])

        conv_mem.add("user", query)
        conv_mem.add("assistant", response)
        sem_mem.extract_and_store(query, response)
        ep_mem.log(query, result)

if __name__ == "__main__":
    main()
```

#### Step 7.2 — Observability Tracer

File: `observability/tracer.py`

```python
import json
from datetime import datetime
from pathlib import Path

class Tracer:
    def __init__(self, debug: bool = False, log_path: str = "traces.jsonl"):
        self.debug = debug
        self.log_path = Path(log_path)

    def log(self, state: dict) -> None:
        entry = {
            "timestamp":         datetime.utcnow().isoformat(),
            "query":             state.get("query"),
            "rewritten_query":   state.get("rewritten_query"),
            "action":            state.get("action"),
            "chunks_retrieved":  len(state.get("retrieved_chunks", [])),
            "top_chunk_preview": state["retrieved_chunks"][0]["text"][:200]
                                 if state.get("retrieved_chunks") else None,
            "tool_used":         bool(state.get("tool_output")),
            "answer_preview":    (state.get("answer") or "")[:300],
            "trace_steps":       state.get("trace", []),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def print_trace(self, trace: list) -> None:
        print("\n── DEBUG TRACE ──────────────────────────")
        for step in trace:
            print(f"  [{step['node']}] {step['detail']}")
        print("─────────────────────────────────────────\n")
```

Every node in the graph appends a step to `state["trace"]`:

```python
# Example inside the router node
state["trace"].append({
    "node":      "router",
    "detail":    f"Action chosen: RETRIEVE. Reason: query is about AI/ML research.",
    "timestamp": datetime.utcnow().isoformat()
})

# Example inside the retrieve node
state["trace"].append({
    "node":      "retrieve",
    "detail":    f"Hybrid search: 20 dense + 20 BM25 → RRF merge → reranked top 5. "
                 f"Top chunk: '{chunk['metadata']['title']}', section: '{chunk['metadata']['section']}'",
    "timestamp": datetime.utcnow().isoformat()
})
```

`--debug` mode prints the trace inline. Every query is also written to `traces.jsonl` for post-hoc inspection.

#### Step 7.3 — README Structure

The README must have these sections in this order:

```
1. Setup
   Clone → pip install -r requirements.txt → set .env → python ingest/embed_and_store.py → python cli.py
   Target: a stranger can run this in under 10 minutes.

2. Architecture overview
   ASCII diagram or image. Brief description of each component.

3. Decisions log
   Format per decision:
     ## Decision: [Topic]
     Considered: X, Y, Z
     Chose: X
     Why: [One clear sentence with a concrete reason]

4. Evaluation results
   Table: question | expected_action | actual_action | quality | hallucination
   Include honest analysis of where the system failed.

5. Ablation table
   Config | Precision@5 | Avg quality | Latency (ms)

6. What I would do with another week
   At least 5 specific, concrete items. Not vague wishes.

7. Known limitations and failure modes observed
   At least 3 honest limitations observed during testing.
```

#### Step 7.4 — Demo Video Script (5–8 minutes)

```
0:00–2:00 — Architecture walkthrough
  Show the repo structure briefly.
  Walk through: ingest → retriever → agent graph → memory → tracer.
  Show the LangGraph graph visualization (graph.get_graph().draw_mermaid()).

2:00–5:00 — Three live queries with --debug flag on

  Query 1 (happy path):
    "How do recent papers approach LLM hallucination reduction?"
    Show: RETRIEVE action, hybrid retrieval, top chunks with paper titles,
    final answer with citations. Walk through the debug trace.

  Query 2 (clarification then refusal):
    "Tell me about transformers."
    Show: CLARIFY action, one focused clarifying question returned.
    Then: "What is the best pizza recipe?"
    Show: REFUSE action with a clear reason.

  Query 3 (tool use edge case):
    "What papers on speculative decoding were published this week?"
    Show: USE_TOOL action, arXiv API call, fresh results incorporated into answer.

5:00–7:00 — One decision you are proud of + one you are unsure about
  Proud of: explain why hybrid + reranking was chosen, show the ablation table
            proving it helped, point to where in the code it lives.
  Unsure about: e.g., "My semantic memory extraction sometimes over-extracts —
                it stored 'interested in efficiency' when the user only mentioned
                it once. With more time I would add a confidence threshold and
                a minimum occurrence count before storing a fact."

7:00–8:00 — Face on camera, brief summary
  "Here is what I built, here is what I would improve, here is what I learned."
```

---

## 6. Component Deep Dives

### Query Rewriting — HyDE in Detail

HyDE (Hypothetical Document Embeddings) works like this:

```
Normal RAG:  embed(query) → similarity search → retrieve chunks
HyDE RAG:    LLM("write a paper passage answering: {query}")
             → embed(hypothetical_passage) → similarity search → retrieve chunks
```

Why it works: the embedding of a hypothetical answer is closer in embedding space to real answer passages than the embedding of a question. Questions and answers live in different regions of the embedding space — vocabulary mismatch is high for technical domains.

When to use it: questions phrased as questions ("How does X work?"). Skip for lookup queries ("What did paper 2401.12345 say about X?") where the query is already close to the target.

Show it working in the eval: pick one question where HyDE improves retrieval precision vs the raw query.

### Memory — Why Three Types Matter

| Type | What it stores | Scope | Used for |
|---|---|---|---|
| Conversational | Last 6 messages | Session only | Pronoun resolution, follow-up disambiguation |
| Semantic | Extracted user facts | Persistent (SQLite) | Retrieval bias, personalized answers |
| Episodic | Past query log | Persistent (JSONL) | Avoid repeat retrieval, surface prior answers |

This distinction maps to how human memory actually works: working memory, semantic long-term memory, episodic long-term memory. Mentioning this explicitly in your README shows you understand the theory, not just the code.

### Hybrid Search — Why RRF over Score Normalization

Score normalization approach: normalize BM25 scores to [0,1], normalize cosine to [0,1], take a weighted average. Problem: the distributions differ every query. A BM25 max of 8 on one query and 40 on another means the normalization is inconsistent across queries.

RRF approach: use only rank position, not score magnitude. Rank 1 from both lists is guaranteed to score higher than rank 5 from either list alone. Scale-invariant, query-invariant, empirically robust.

```
RRF(d) = 1 / (60 + rank_bm25(d))  +  1 / (60 + rank_dense(d))
```

k=60 is the standard constant. It dampens the rank-1 advantage so that documents ranked 2nd in both lists beat documents ranked 1st in only one list.

---

## 7. Evaluation Strategy

### What "correctness" means in this system

Correctness has three separate components:

1. **Action correctness**: did the agent take the right action type? Binary, objective.
2. **Answer faithfulness**: does the answer stick to what the retrieved chunks say? No fabrication.
3. **Answer completeness**: does the answer actually address the question, not just cite papers?

### Eval is not about a high score

The two refusal questions and two clarification questions should cause the agent to produce no answer. If your agent answers everything and scores 8/10 on answer quality, that is worse than scoring 6/10 with correct refusals — it means the agent does not know its own limits.

### Ablation table format for README

| Retrieval config | Avg precision@5 | Avg answer quality | Avg latency |
|---|---|---|---|
| Dense only | X / 5 | X.X / 2 | Xms |
| Hybrid (BM25 + Dense) | X / 5 | X.X / 2 | Xms |
| Hybrid + Reranking | X / 5 | X.X / 2 | Xms |

---

## 8. Observability Design

### Debug trace (--debug mode)

```
── DEBUG TRACE ──────────────────────────
  [rewrite_query] Rewrote "transformers" → "transformer architecture NLP attention BERT GPT"
                  using semantic memory (user is NLP researcher)
  [router]        Action: RETRIEVE
                  Reason: AI/ML research question, not ambiguous, corpus likely relevant
  [retrieve]      Dense search: 20 results. BM25 search: 20 results.
                  After RRF merge: 20 unified candidates.
                  After cross-encoder rerank: top 5 selected.
                  Top chunk: "Attention Is All You Need" — Section 3, page 4
  [answer]        Generated answer (312 tokens).
                  Contradiction check: NONE detected.
                  Confidence: high (top chunk score 0.87)
─────────────────────────────────────────
```

### Persistent trace log (traces.jsonl)

Every query writes one JSON line to `traces.jsonl` with full structured data. This allows post-hoc inspection, debugging, and building the eval harness on top of real system outputs.

---

## 9. README & Demo Checklist

### README

- [ ] Setup instructions tested from scratch (not just from your own machine)
- [ ] Architecture diagram present (ASCII or image)
- [ ] Decisions log covers: chunking, embeddings, vector DB, framework, retrieval technique, memory design, eval methodology
- [ ] Evaluation table with all 10 questions and scores
- [ ] Ablation table: dense vs hybrid vs hybrid+rerank
- [ ] "What I would do with another week" — at least 5 specific items
- [ ] Known limitations — at least 3 honest, observed limitations

### Code quality

- [ ] Type hints on all function signatures
- [ ] Docstrings on all public functions
- [ ] `requirements.txt` with pinned versions
- [ ] `.env.example` with all required key names (no values)
- [ ] No API keys hardcoded anywhere in the codebase
- [ ] `tests/` directory with at least 3 meaningful tests
- [ ] `python cli.py` works in under 10 minutes from a fresh clone

### Demo video

- [ ] Face on camera for at least part of it
- [ ] `--debug` trace shown for at least one query
- [ ] Refusal or clarification demonstrated (not only happy path)
- [ ] arXiv tool call shown in action
- [ ] Ablation result mentioned and explained
- [ ] Within 5–8 minutes

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PDF parsing fails on some papers | Medium | Medium | Use PyMuPDF fallback text extraction; log and skip bad PDFs |
| arXiv API rate limits during ingest | Low | Low | Add 1s sleep between calls; cache downloaded PDFs |
| LLM costs exceed budget | Low | Medium | Use Haiku/mini; set max_tokens=500 for routing; cache repeated queries |
| Cross-encoder is too slow per query | Low | Low | Only rerank top-20, not full corpus; add optional caching |
| Router misclassifies edge cases | Medium | Medium | Add few-shot examples to router prompt for tricky cases |
| Ablation shows no improvement from reranking | Low | High | Investigate and report honestly — unexpected results explained well are still impressive |
| Semantic memory extraction adds latency | Medium | Low | Make extraction async; skip if conversation is fewer than 3 turns |
| ChromaDB persistence issues across runs | Low | Medium | Always call `.persist()` after writes; verify on restart |

---

## What I Would Do With Another Week

Include this section verbatim in your README. Fill in your actual answers after building.

1. **Citation graph retrieval**: build a graph where papers are nodes and citations are edges. When retrieving, also fetch papers cited by the top result — the most relevant related work is often one citation hop away.

2. **Multi-modal retrieval**: extract figures and tables from PDFs using PyMuPDF. Embed figure captions. Allow queries like "show me papers with loss curves comparing X and Y."

3. **Streaming responses**: stream the LLM answer token-by-token in the CLI. Long answers on technical topics should not require waiting 5 seconds for a complete response.

4. **Human-labeled eval**: replace the self-scored 0–2 quality rubric with a small set of human-labeled golden answers for more reliable and reproducible benchmarking.

5. **Semantic chunking improvement**: use a sentence embedding model to detect topic shifts within sections, not just paragraph breaks — would reduce chunk boundary errors in dense methods sections of papers.

6. **Persistent semantic memory across sessions**: currently semantic memory resets between CLI runs. A proper user profile stored in SQLite would make the system genuinely personalized across multiple sessions.

---

*Built for Skyclad Ventures AI Engineering Intern Assignment.*
*Timeline: 7 days. Budget target: under $5 in compute.*
