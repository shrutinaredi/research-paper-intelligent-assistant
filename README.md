# Intelligent Research Assistant

A multi-agent research assistant that answers questions over a private
library of PDF research papers with page-level citations.

Orchestrated with **LangGraph** on top of the **Groq API**
(Llama 3.1 8B Instant), with a local **ChromaDB** vector store.

## Architecture

```
question
   ↓
┌──────────┐
│  Router  │  classify (factual / multihop / comparison), pick top_k,
└──────────┘  rewrite query for semantic search      — Llama 3.1 8B Instant
   ↓
┌───────────┐
│ Retriever │  ChromaDB cosine search over chunked PDFs
└───────────┘
   ↓
┌─────────────┐
│ Synthesizer │  answer with inline [source, p.N] citations
└─────────────┘                                      — Llama 3.1 8B Instant
   ↓
┌────────┐
│ Critic │  verify every claim is grounded in a retrieved chunk
└────────┘                                           — Llama 3.1 8B Instant
   ↓                 ↘ if unsupported claims → retry with refined query
  END                  (hard cap at 2 attempts)
```

All telemetry (query, retrieved chunks, answer, critic verdict, latency,
user thumbs) is logged to a local SQLite DB — `research_assistant.db`.

## Directory layout

```
ingest.py              # PDF → chunks → embeddings → ChromaDB (pre-existing)
agents/
  state.py             # shared LangGraph state TypedDict
  router.py            # question classifier + query rewriter
  retriever.py         # ChromaDB lookup wrapper
  synthesizer.py       # citation-constrained answer generator
  critic.py            # claim-level verifier
  graph.py             # LangGraph state machine + run_query()
  telemetry.py         # SQLite logging for runs + feedback
app.py                 # Streamlit chat UI
eval/
  eval_set.json        # labeled Q/A triples
  run_eval.py          # pytest-style eval harness w/ regression thresholds
data/pdfs/             # drop your PDFs here
chroma_db/             # persistent vector store (auto-created)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# set your API key in .env
# get a free Groq key at https://console.groq.com/keys
echo "GROQ_API_KEY=gsk_..." > .env
```

## Running it

1. **Drop PDFs** into `data/pdfs/`.
2. **Ingest:** `python ingest.py` — extracts, chunks, embeds, stores in ChromaDB.
3. **Chat:** `streamlit run app.py` — opens the web UI at http://localhost:8501.
4. **Eval:** `python -m eval.run_eval` — runs the labeled set, prints pass rates,
   exits non-zero on regression.

## Metrics

Every run logs to `research_assistant.db` — inspect with any SQLite client:

- `runs` — one row per question with chunks, answer, critic verdict, latency
- `feedback` — thumbs up/down linked to `run_id`

The eval harness tracks three gates (tune thresholds in `eval/run_eval.py`):

| Metric | What it measures |
|---|---|
| `retrieval_hit` | Did any retrieved chunk come from the expected source? |
| `mention_hit` | Does the answer contain any of the expected keywords? |
| `critic_supported` | Did the critic think every claim was grounded? |

## Stack

- Orchestration: **LangGraph**
- Model via **Groq**: **Llama 3.1 8B Instant** — fast, generous free-tier quota (500K tokens/day)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, 384-d)
- Vector store: **ChromaDB** (persistent, cosine, HNSW)
- PDF: `pdfplumber`
- UI: **Streamlit**
- Telemetry: SQLite + pytest-style eval harness
