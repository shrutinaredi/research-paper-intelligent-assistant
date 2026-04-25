"""Streamlit UI for the research assistant.

Users upload up to 5 PDFs in the sidebar; each session gets its own
in-memory ChromaDB so uploads never leak across users. Ask questions
and the multi-agent pipeline answers with page-level citations and a
critic verdict.

Run:  streamlit run app.py
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Local dev reads .env; Streamlit Cloud's st.secrets doesn't auto-export to
# os.environ, so bridge it here before any Groq client is instantiated.
load_dotenv()
try:
    secret_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    secret_key = None
if secret_key and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = secret_key

st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.getenv("GROQ_API_KEY"):
    st.title("Research Assistant")
    st.error(
        "**`GROQ_API_KEY` is not configured.**\n\n"
        "If you're running locally: add it to `.env`.\n\n"
        "If you're on Streamlit Cloud: open **Manage app → Settings → "
        "Secrets** and add a line exactly like this (quotes required):\n\n"
        '`GROQ_API_KEY = "gsk_..."`\n\n'
        "Click **Save** — the app reboots automatically."
    )
    st.stop()

from agents import run_query  # noqa: E402  (imports after env is wired)
from agents.retriever import set_retrieval_context  # noqa: E402
from agents.telemetry import log_feedback  # noqa: E402
from ingest import EMBEDDING_MODEL, ingest_file_obj  # noqa: E402

MAX_PDFS = 5

# ---------------------------------------------------------------------------
# Custom CSS — typography, pills, cards
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      /* Tighter top padding so the hero starts higher */
      .block-container { padding-top: 2rem; }

      /* Hero */
      .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        color: #0F172A;
      }
      .hero-tag {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
      }

      /* Status pills */
      .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        line-height: 1.5;
      }
      .pill-good   { background:#DCFCE7; color:#166534; }
      .pill-warn   { background:#FEF3C7; color:#92400E; }
      .pill-info   { background:#E0E7FF; color:#3730A3; }
      .pill-muted  { background:#F1F5F9; color:#475569; }

      /* Q&A card */
      .qcard h3 {
        font-size: 1.1rem;
        margin: 0 0 0.5rem 0;
        color: #0F172A;
      }

      /* Source chip */
      .source-chip {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
      }

      /* Empty-state card */
      .empty-card {
        border: 1px dashed #CBD5E1;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #F8FAFC;
        color: #475569;
      }

      /* Footer */
      .footer-note {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    st.session_state.ingested = {}      # filename -> chunk count
    st.session_state.history = []
    st.session_state.prefilled_q = ""

model = get_embedding_model()
set_retrieval_context(model, st.session_state.collection)


# ---------------------------------------------------------------------------
# Sidebar: upload + manage
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Your library")
    st.caption(f"Up to {MAX_PDFS} PDFs · stored in memory · not shared")

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest", disabled=not uploaded, use_container_width=True, type="primary"):
        new_files = [f for f in uploaded if f.name not in st.session_state.ingested]
        room = MAX_PDFS - len(st.session_state.ingested)
        to_ingest = new_files[:room]
        if len(new_files) > room:
            st.warning(
                f"Cap of {MAX_PDFS} PDFs; skipping {len(new_files) - room} file(s)."
            )
        for f in to_ingest:
            with st.status(f"Ingesting {f.name}", expanded=False) as status:
                n_chunks = ingest_file_obj(f, f.name, st.session_state.collection, model)
                if n_chunks:
                    status.update(label=f"{f.name} · {n_chunks} chunks", state="complete")
                else:
                    status.update(label=f"{f.name} · no extractable text", state="error")
            if n_chunks:
                st.session_state.ingested[f.name] = n_chunks

    if st.session_state.ingested:
        st.divider()
        st.markdown("**Ingested papers**")
        for name, n in st.session_state.ingested.items():
            st.markdown(
                f"<div style='font-size:0.9rem;'><b>{name}</b><br>"
                f"<span class='source-chip'>{n} chunks</span></div>",
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("Clear all", use_container_width=True):
            _reset_collection()
            set_retrieval_context(model, st.session_state.collection)
            st.rerun()

    st.divider()
    st.caption(
        "**How it works**  \n"
        "Router → Retriever → Synthesizer → Critic.  \n"
        "If the critic flags an unsupported claim, the graph re-retrieves "
        "with a refined query (max 2 retries)."
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-tag">Ask questions over your own research papers · '
    'page-level citations · self-verifying critic loop</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Empty-state OR ask form
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What is the main contribution of this paper?",
    "Summarize the methodology in 3 sentences.",
    "What are the key results and which tables back them up?",
    "What are the limitations the authors acknowledge?",
]


def _click_example(q: str) -> None:
    st.session_state.prefilled_q = q


if not st.session_state.ingested:
    st.markdown(
        '<div class="empty-card">'
        '<b>No papers loaded yet.</b><br>'
        'Drop one or more PDFs into the sidebar on the left, click '
        '<b>Ingest</b>, then ask away.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    with st.form("ask", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            value=st.session_state.prefilled_q,
            placeholder="e.g. How does self-attention differ from additive attention?",
            label_visibility="collapsed",
        )
        col_submit, col_examples = st.columns([1, 4])
        with col_submit:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with col_examples:
            st.caption("Try one of these:")

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(
            ex,
            key=f"ex_{hash(ex)}",
            on_click=_click_example,
            args=(ex,),
            use_container_width=True,
        )

    if submitted and question.strip():
        st.session_state.prefilled_q = ""
        with st.spinner("Routing → retrieving → synthesizing → critiquing…"):
            try:
                result = run_query(question.strip())
                st.session_state.history.insert(0, result)
            except Exception as e:
                st.error(f"Query failed: {e}")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def render_run(run: dict) -> None:
    critic = run.get("critic", {}) or {}
    supported = critic.get("supported", True)

    pill_status = (
        '<span class="pill pill-good">grounded</span>'
        if supported
        else '<span class="pill pill-warn">unsupported claims</span>'
    )
    pill_type = f'<span class="pill pill-info">{run.get("question_type", "?")}</span>'

    with st.container(border=True):
        st.markdown(
            f'<div class="qcard"><h3>{run["question"]}</h3>{pill_status}{pill_type}</div>',
            unsafe_allow_html=True,
        )

        # Metric tiles
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency", f"{run.get('latency_ms', '—')} ms")
        c2.metric("Sources used", len(run.get("chunks", [])))
        c3.metric("Retries", max(0, run.get("attempts", 1) - 1))
        c4.metric("Top-k", run.get("top_k", "—"))

        st.markdown("---")
        st.markdown(run.get("answer", "_(no answer)_"))

        if not supported and critic.get("unsupported_claims"):
            with st.expander("Critic flagged claims"):
                for c in critic["unsupported_claims"]:
                    st.markdown(f"- {c}")

        with st.expander(f"Sources · {len(run.get('chunks', []))}"):
            for i, chunk in enumerate(run.get("chunks", []), 1):
                st.markdown(
                    f"**{i}. {chunk['source']} — page {chunk['page']}** "
                    f"<span class='source-chip'>· similarity {chunk['score']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.9rem;color:#334155;"
                    f"background:#F8FAFC;padding:0.75rem;border-radius:8px;"
                    f"margin-bottom:0.5rem;'>{chunk['text']}</div>",
                    unsafe_allow_html=True,
                )

        # Feedback
        run_id = run["run_id"]
        col_up, col_down, _spacer = st.columns([1, 1, 12])
        if col_up.button("👍 helpful", key=f"up_{run_id}", use_container_width=True):
            log_feedback(run_id, "up")
            st.toast("Logged as helpful")
        if col_down.button("👎 not helpful", key=f"down_{run_id}", use_container_width=True):
            log_feedback(run_id, "down")
            st.toast("Logged as not helpful")


for run in st.session_state.history:
    render_run(run)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer-note">'
    'Built with LangGraph · Llama 3.1 8B (Groq) · ChromaDB · Streamlit. '
    'Sources are returned with cosine similarity scores; the critic '
    'agent verifies every claim against retrieved excerpts.'
    '</div>',
    unsafe_allow_html=True,
)
