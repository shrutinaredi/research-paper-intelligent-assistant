"""Streamlit UI for the research assistant.

Users upload up to 5 PDFs in the sidebar; each session gets its own
in-memory ChromaDB so uploads never leak across users. Ask questions
and the multi-agent pipeline answers with page-level citations and a
critic verdict.

Run:  streamlit run app.py
"""

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from agents import run_query
from agents.retriever import set_retrieval_context
from agents.telemetry import log_feedback
from ingest import EMBEDDING_MODEL, ingest_file_obj

load_dotenv()

MAX_PDFS = 5

st.set_page_config(page_title="Research Assistant", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources (once per server instance)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Per-session state: ephemeral Chroma client + ingested-files registry
# ---------------------------------------------------------------------------

def _reset_collection() -> None:
    client = st.session_state.chroma_client
    try:
        client.delete_collection("session_papers")
    except Exception:
        pass
    st.session_state.collection = client.get_or_create_collection(
        name="session_papers",
        metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested = {}


if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
    st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
        name="session_papers",
        metadata={"hnsw:space": "cosine"},
    )
    st.session_state.ingested = {}  # filename -> chunk count
    st.session_state.history = []

model = get_embedding_model()

# Wire the retriever to this session's collection — does nothing on CLI paths
set_retrieval_context(model, st.session_state.collection)


# ---------------------------------------------------------------------------
# Sidebar: upload + ingest + manage
# ---------------------------------------------------------------------------

st.sidebar.title("Your papers")
st.sidebar.caption(f"Upload up to {MAX_PDFS} PDFs. Kept in memory — not shared.")

uploaded = st.sidebar.file_uploader(
    "Drop PDFs here",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if st.sidebar.button("Ingest", disabled=not uploaded, use_container_width=True):
    new_files = [f for f in uploaded if f.name not in st.session_state.ingested]
    room = MAX_PDFS - len(st.session_state.ingested)
    to_ingest = new_files[:room]
    if len(new_files) > room:
        st.sidebar.warning(
            f"Cap of {MAX_PDFS} PDFs; skipping {len(new_files) - room} file(s)."
        )

    for f in to_ingest:
        with st.sidebar.status(f"Ingesting {f.name}...", expanded=False):
            n_chunks = ingest_file_obj(f, f.name, st.session_state.collection, model)
        if n_chunks:
            st.session_state.ingested[f.name] = n_chunks
        else:
            st.sidebar.warning(f"{f.name}: no extractable text (scanned PDF?)")

if st.session_state.ingested:
    st.sidebar.markdown("### Ingested")
    for name, n in st.session_state.ingested.items():
        st.sidebar.markdown(f"- **{name}**  ·  {n} chunks")

    if st.sidebar.button("Clear all", use_container_width=True):
        _reset_collection()
        set_retrieval_context(model, st.session_state.collection)
        st.rerun()


# ---------------------------------------------------------------------------
# Main: ask + history
# ---------------------------------------------------------------------------

st.title("Research Assistant")
st.caption(
    "Upload papers on the left, then ask questions. Answers include inline "
    "[source, p.N] citations and a critic verdict. Multi-agent pipeline: "
    "router → retrieve → synthesize → critique (with retry)."
)

if not st.session_state.ingested:
    st.info("Upload at least one PDF on the left to start.")
else:
    with st.form("ask"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. How does self-attention differ from additive attention?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Routing → retrieving → synthesizing → critiquing..."):
            result = run_query(question.strip())
        st.session_state.history.insert(0, result)

# History
for run in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Q:** {run['question']}")

        critic = run.get("critic", {})
        supported = critic.get("supported", True)
        badge = "✅ supported" if supported else "⚠️ unsupported claims"
        st.caption(
            f"{badge}  ·  "
            f"{run.get('question_type', '?')}  ·  "
            f"top_k={run.get('top_k', '?')}  ·  "
            f"attempts={run.get('attempts', '?')}  ·  "
            f"{run.get('latency_ms', '?')} ms"
        )

        st.markdown(run.get("answer", "*(no answer)*"))

        if not supported and critic.get("unsupported_claims"):
            with st.expander("Unsupported claims flagged by critic"):
                for c in critic["unsupported_claims"]:
                    st.markdown(f"- {c}")

        with st.expander(f"Sources ({len(run.get('chunks', []))})"):
            for i, chunk in enumerate(run.get("chunks", []), 1):
                st.markdown(
                    f"**[{i}] {chunk['source']} — p.{chunk['page']}** "
                    f"_(score {chunk['score']})_"
                )
                st.text(chunk["text"])
                st.divider()

        run_id = run["run_id"]
        col_up, col_down, _ = st.columns([1, 1, 10])
        if col_up.button("👍", key=f"up_{run_id}"):
            log_feedback(run_id, "up")
            st.toast("Thanks — logged as 👍")
        if col_down.button("👎", key=f"down_{run_id}"):
            log_feedback(run_id, "down")
            st.toast("Thanks — logged as 👎")
