"""Streamlit UI for the research assistant.

Run with:  streamlit run app.py
Prerequisite: run `python ingest.py` at least once after dropping PDFs
into data/pdfs/.
"""

import streamlit as st
from dotenv import load_dotenv

from agents import run_query
from agents.telemetry import log_feedback

load_dotenv()

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Research Assistant")
st.caption(
    "Ask questions about the papers in your corpus. "
    "Answers include inline citations and a critic verdict."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of run dicts

# --- input ------------------------------------------------------------------

with st.form("ask"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. How does self-attention differ from additive attention?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Routing, retrieving, synthesizing, critiquing..."):
        result = run_query(question.strip())
    st.session_state.history.insert(0, result)

# --- history ----------------------------------------------------------------

for idx, run in enumerate(st.session_state.history):
    with st.container(border=True):
        st.markdown(f"**Q:** {run['question']}")

        critic = run.get("critic", {})
        supported = critic.get("supported", True)
        badge = "✅ supported" if supported else "⚠️ unsupported claims"
        meta = (
            f"{badge}  ·  "
            f"{run.get('question_type', '?')}  ·  "
            f"top_k={run.get('top_k', '?')}  ·  "
            f"attempts={run.get('attempts', '?')}  ·  "
            f"{run.get('latency_ms', '?')} ms"
        )
        st.caption(meta)

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

        # feedback buttons — keyed by run_id so reruns don't collide
        run_id = run["run_id"]
        col_up, col_down, _ = st.columns([1, 1, 10])
        if col_up.button("👍", key=f"up_{run_id}"):
            log_feedback(run_id, "up")
            st.toast("Thanks — logged as 👍")
        if col_down.button("👎", key=f"down_{run_id}"):
            log_feedback(run_id, "down")
            st.toast("Thanks — logged as 👎")
