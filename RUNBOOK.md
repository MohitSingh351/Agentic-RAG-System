# SkyClad RAG Agent — Runbook

This guide covers everything needed to go from a fresh clone to a running interactive agent.

---

## Prerequisites

### 1. Python version

The project requires **Python 3.12**. Check what you have:

```bash
python3 --version
```

If you need 3.12, install it via pyenv:

```bash
brew install pyenv
pyenv install 3.12.10
pyenv local 3.12.10
```

### 2. API keys

You need two API keys. Copy the example file and fill them in:

```bash
cp .env.example .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com
OPENAI_API_KEY=sk-...             # from platform.openai.com (for embeddings only)
```

The app uses Claude Haiku for the agent (routing, answering, clarifying) and OpenAI `text-embedding-3-small` for dense retrieval embeddings.

---

## Step 1 — Create virtual environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed if running tests
```

Verify the install:

```bash
python -c "import anthropic, openai, chromadb, rank_bm25, fitz, sentence_transformers, arxiv, langgraph; print('All imports OK')"
```

---

## Step 2 — Ingest the arXiv corpus (one-time setup)

This fetches ~70 cs.AI papers from the last 90 days, parses their PDFs, chunks the text, and builds the vector + BM25 indexes.

Run each step in order:

```bash
# Fetch papers and download PDFs (~5–10 min, rate-limited at 3s per download)
python -m ingest.fetch_arxiv --max-results 70 --days-back 90

# Parse PDFs into sections using font-size heuristics
python -m ingest.parse_pdfs

# Chunk sections into 400–700 token segments
python -m ingest.chunk

# Embed to ChromaDB and serialize BM25 index
# (~10–20 min depending on OpenAI API speed)
python -m ingest.embed_and_store
```

Verify the corpus was built:

```bash
python -c "
from retrieval.dense import get_chroma_collection
c = get_chroma_collection()
print(f'Corpus: {c.count()} chunks indexed')
"
```

You should see `Corpus: N chunks indexed` where N ≥ 200.

**Important:** Ingestion only needs to run once. The ChromaDB database and BM25 index are persisted to `data/` and reloaded on every agent start. If you want fresher papers, re-run ingestion.

---

## Step 3 — Run the agent

```bash
source .venv/bin/activate
python cli.py
```

### Options

| Flag | Effect |
|---|---|
| `--debug` | Print action, confidence score, and trace node count after each response |
| `--no-color` | Disable ANSI colors (useful for piping output) |
| `--session-id <id>` | Use a fixed session ID (default: random UUID) |

```bash
python cli.py --debug
python cli.py --no-color
```

### Built-in commands

| Command | Effect |
|---|---|
| `/clear` | Clear the conversational memory window |
| `/trace` | Print the node names from the last query's trace |
| `/help` | Show available commands |
| `/quit` or `/exit` | Exit the REPL |

---

## Example Inputs

Below are example queries and the expected behavior for each of the five agent actions.

### RETRIEVE — grounded answer from indexed papers

```
You: What are the main innovations in recent transformer architectures?
```
The agent runs hybrid retrieval (BM25 + ChromaDB), reranks with a cross-encoder, and answers from the top-5 chunks. Expected response: a factual answer citing specific paper findings with confidence ≥ 0.4.

```
You: How does the attention mechanism work?
```
```
You: What are recent advances in diffusion models for image generation?
```
```
You: How do graph neural networks handle structured data?
```

### USE_TOOL — live arXiv search for time-sensitive questions

```
You: What papers on reinforcement learning from human feedback came out this week?
```
The agent calls the arXiv API live (not the indexed corpus) and summarises recent results. Useful for anything time-sensitive.

```
You: Are there any new papers on speculative decoding published recently?
```

### CLARIFY — disambiguating vague questions

```
You: Tell me about transformers.
```
Expected: the agent asks a focused follow-up like *"Are you asking about transformer model architectures in NLP, or a specific model like GPT or BERT?"*

```
You: Explain models.
```

### REFUSE — out-of-domain queries

```
You: What is the best chocolate cake recipe?
```
Expected: *"I'm specialized in cs.AI research papers from the last 90 days. I can't help with cooking recipes."*

```
You: Can you predict Tesla's stock price tomorrow?
```

### ANSWER — answering from memory without retrieval

After asking several questions in the same session, the agent may answer follow-up questions directly from memory:

```
You: What is attention in transformers?
You: Can you summarise what you just told me?
```

---

## Debug mode example

```bash
python cli.py --debug
```

```
You: How do diffusion models work?

Answer:
Diffusion models learn to reverse a gradual noising process...

[DEBUG] Action: RETRIEVE | Confidence: 0.83 | Nodes: 5
```

Use `/trace` to see which nodes ran:

```
You: /trace
  query_rewriter
  router
  retrieve
  answer
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

All 170 tests run with mocked LLM and retrieval calls — no API keys or indexed corpus needed.

To run a specific phase:

```bash
pytest tests/test_graph.py -v          # agent graph routing
pytest tests/test_retrieval_integration.py -v  # requires indexed corpus
pytest tests/test_answer.py -v         # answer generation
```

---

## Running Evaluation

Requires an indexed corpus and API keys.

```bash
# Score all 10 eval questions (action accuracy, topic coverage, refusal correctness)
python eval/run_eval.py

# Ablation: dense-only vs hybrid vs full pipeline
python eval/ablation.py
```

Results are written to `eval/results.json`.

---

## File Layout

```
SkyClad/
├── ingest/           # Fetch, parse, chunk, embed arXiv papers
├── retrieval/        # Dense (ChromaDB), sparse (BM25), hybrid (RRF), rerank
├── memory/           # Conversational (deque), semantic (SQLite), episodic (JSONL)
├── agent/            # LangGraph nodes: state, rewriter, router, answer, clarify, refuse, tools, graph
├── observability/    # JSONL tracer with per-node timing
├── eval/             # 10-question eval harness + ablation study
├── tests/            # 170 unit tests (all mocked, no API calls needed)
├── data/             # Generated: PDFs, ChromaDB, BM25 index, memory files
├── traces/           # Generated: per-session JSONL trace files
├── cli.py            # Interactive REPL entry point
├── .env.example      # Environment variable template
├── requirements.txt
└── requirements-dev.txt
```
