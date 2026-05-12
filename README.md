# SkyClad Agentic RAG System

An autonomous Retrieval-Augmented Generation agent over ~70 arXiv cs.AI papers from the last 90 days, built for the Skyclad Ventures AI Engineering Intern Assignment.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ rewrite_node│  ← Load memory context + HyDE (Hypothetical Document Embeddings)
└──────┬──────┘
       │
    ▼
┌─────────────┐
│ router_node │  ← LLM classifies into 5 actions
└──────┬──────┘
       │
   ┌───┴──────────────────────────┐
   │                              │
RETRIEVE  USE_TOOL  CLARIFY  REFUSE  ANSWER
   │         │         │       │       │
retrieve_  tool_    clarify_ refuse_ answer_
  node      node     node    node    node
   │         │                │       │
   ▼         └────────────────┘       │
answer_node ◄────────────────────────┘
   │
   ▼
Memory update (conversational + semantic + episodic)
```

**Five actions:**
- `RETRIEVE` — hybrid BM25 + dense search → cross-encoder rerank → grounded answer
- `USE_TOOL` — arXiv API live search → answer from results
- `CLARIFY` — ambiguous query → one focused clarification question
- `REFUSE` — out-of-domain query → hard or soft refusal with confidence %
- `ANSWER` — answer directly from memory without retrieval

---

## Tech Stack

| Component | Library | Notes |
|---|---|---|
| Agent framework | LangGraph 0.2.28 | StateGraph with conditional edges |
| LLM | Claude Haiku (claude-haiku-4-5-20251001) | Routing, rewriting, answering |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim via OpenAI API |
| Vector DB | ChromaDB 0.5.18 | Cosine distance, PersistentClient |
| Sparse retrieval | rank-bm25 0.2.2 | BM25Okapi with `.lower().split()` tokenization |
| Reranker | sentence-transformers 3.3.1 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF parsing | PyMuPDF 1.24.13 | Font-size heuristic for section detection |
| Chunking | tiktoken 0.8.0 | cl100k_base, 400–700 token range |
| arXiv search | arxiv 2.1.3 | Live API for USE_TOOL path |
| Retry logic | tenacity 9.0.0 | Exponential backoff on API calls |

---

## Setup

```bash
# 1. Create virtual environment (requires Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

---

## Data Ingestion

```bash
# Fetch ~70 arXiv cs.AI papers from the last 90 days
python -m ingest.fetch_arxiv --max-results 70 --days-back 90

# Parse PDFs into sections
python -m ingest.parse_pdfs

# Chunk into 400–700 token segments
python -m ingest.chunk

# Embed to ChromaDB + serialize BM25 index
python -m ingest.embed_and_store
```

After ingestion, verify with:
```bash
python -c "from retrieval.dense import get_chroma_collection; c=get_chroma_collection(); print(f'Corpus: {c.count()} chunks')"
```

---

## Running the CLI

```bash
# Interactive REPL
python cli.py

# With debug output (action, confidence, trace node count)
python cli.py --debug

# No ANSI colors (for piping or logging)
python cli.py --no-color
```

**Built-in commands:**
- `/clear` — clear conversation memory
- `/trace` — show last session trace nodes
- `/quit` or `/exit` — exit

---

## Running Evaluation

```bash
# Full 10-question evaluation
python eval/run_eval.py

# Ablation study: dense-only vs hybrid vs hybrid+rerank
python eval/ablation.py
```

---

## Running Tests

```bash
# All 170 tests
pytest tests/ -v

# Single phase
pytest tests/test_graph.py -v
pytest tests/test_retrieval_integration.py -v  # requires indexed corpus
```

---

## Memory System

Three independent memory types:

| Type | Implementation | Purpose |
|---|---|---|
| Conversational | `collections.deque(maxlen=6)` | Last 6 turns for query rewriting |
| Semantic | SQLite (`data/semantic_memory.db`) | LLM-extracted facts with confidence |
| Episodic | JSONL (`data/episodic_memory.jsonl`) | Past Q&A pairs, difflib similarity search |

Memory is loaded at the start of every `run_agent()` call and updated after each answer.

---

## Trace Format

Each session writes `traces/{session_id}.jsonl` — one JSON array per line:

```json
[
  {"timestamp": "...", "node": "query_rewriter", "hyde_applied": true, ...},
  {"timestamp": "...", "node": "router", "action": "RETRIEVE", "defaulted": false},
  {"timestamp": "...", "node": "retrieve", "num_results": 5, "top_score": 0.87},
  {"timestamp": "...", "node": "answer", "confidence": 0.84, "contradiction_found": false},
  {"timestamp": "...", "node": "FINAL", "action": "RETRIEVE", "confidence": 0.84, "answer_length": 312}
]
```

Load with: `from observability.tracer import Tracer; Tracer.load_trace(session_id)`

---

## Design Decisions

**RRF fusion over score normalization**: Reciprocal Rank Fusion (`1/(60 + rank)`) is score-scale-agnostic — BM25 and cosine distance have incompatible scales. RRF only requires rank, making fusion stable across query types.

**HyDE for retrieval queries**: Hypothetical Document Embeddings embed a generated answer paragraph instead of the raw question. Questions and their answers occupy different regions of embedding space; this bridges the semantic gap.

**2-pass contradiction detection**: A second LLM call checks the initial answer against sources and revises it if contradictions are found. This reduces hallucination without requiring a separate fact-checking model.

**Hard vs soft refusal**: Hard refusals are for out-of-domain queries (no relevant papers exist). Soft refusals fire when retrieval confidence < 0.4 but results exist — they include a partial answer so the user gets something useful.

**Module-level singletons**: ChromaDB collection, BM25 index, and CrossEncoder are loaded once per process and cached. This avoids re-loading large models on every query.

---

## Limitations & Future Work

- **No PDF re-ingestion**: The corpus is static once ingested; new papers require re-running the full pipeline.
- **BM25 tokenization**: `.lower().split()` ignores stop words and morphology; a better tokenizer (e.g., spaCy) would improve sparse recall.
- **Cross-encoder latency**: The sentence-transformers reranker adds ~200ms per query; a lighter model or pre-computed scores would reduce this.
- **Episodic similarity**: `difflib.SequenceMatcher` is character-level and slow at scale; a vector-based episodic index would scale better.
- **Single-tenant**: `_conv_memory` and `_llm_client` are module-level globals; multi-user deployment would require per-session isolation.
