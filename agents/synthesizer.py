"""Synthesizer agent — generates the final answer from retrieved chunks.

Uses Gemini 2.5 Flash with thinking enabled (adaptive budget): this is
the reasoning step, and the cost of getting it wrong is the whole
point of the critic loop.

The system prompt forces the model to answer *only* from retrieved
chunks and to cite every claim as [source, p.N].
"""

from google import genai
from google.genai import types

from .state import GraphState

SYNTH_MODEL = "gemini-2.5-flash-lite"

SYNTH_SYSTEM = """You are a research assistant answering questions about \
academic papers.

You will receive a question and a numbered list of excerpts from a corpus, \
each labeled with its source paper and page number. Answer the question \
using ONLY these excerpts.

Rules:
1. Every factual claim must be followed by an inline citation in the form \
   [source, p.N] — use the exact source name and page from the excerpts.
2. If the excerpts do not contain enough information to answer, say so \
   explicitly. Do not guess and do not fall back on prior knowledge.
3. Keep the answer focused. Prefer 2-5 sentences for factual questions, \
   a short paragraph for multi-hop or comparison questions.
4. Do not invent page numbers, paper titles, or quotations. If an excerpt \
   does not cover the claim, don't make the claim."""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] source={c['source']} page={c['page']} score={c['score']}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


def synthesize(state: GraphState) -> GraphState:
    question = state["question"]
    chunks = state.get("chunks", [])

    if not chunks:
        return {
            "answer": "I couldn't find any relevant passages in the corpus "
                      "to answer this question."
        }

    user_content = (
        f"Question: {question}\n\n"
        f"Excerpts:\n\n{_format_chunks(chunks)}\n\n"
        f"Answer the question using only these excerpts, with inline "
        f"[source, p.N] citations."
    )

    response = _get_client().models.generate_content(
        model=SYNTH_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYNTH_SYSTEM,
        ),
    )

    return {"answer": response.text}
