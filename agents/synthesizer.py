"""Synthesizer agent — generates the final answer from retrieved chunks.

Uses Llama 3.3 70B Versatile via Groq — the biggest model we use, for the
reasoning-heavy step. The system prompt constrains the model to cite
every claim as [source, p.N] from retrieved excerpts only.
"""

from groq import Groq

from .state import GraphState

SYNTH_MODEL = "llama-3.1-8b-instant"

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


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        # max_retries=5 + long timeout so the SDK can honor Groq's
        # retry-after header on 429s (free-tier TPM recoveries take ~20s)
        _client = Groq(max_retries=5, timeout=120.0)
    return _client


def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] source={c['source']} page={c['page']} score={c['score']}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


MAX_SYNTH_CHUNKS = 5  # Keeps the synth call under the 8B 6K TPM bucket


def synthesize(state: GraphState) -> GraphState:
    question = state["question"]
    chunks = state.get("chunks", [])

    if not chunks:
        return {
            "answer": "I couldn't find any relevant passages in the corpus "
                      "to answer this question."
        }

    synth_chunks = chunks[:MAX_SYNTH_CHUNKS]

    user_content = (
        f"Question: {question}\n\n"
        f"Excerpts:\n\n{_format_chunks(synth_chunks)}\n\n"
        f"Answer the question using only these excerpts, with inline "
        f"[source, p.N] citations."
    )

    response = _get_client().chat.completions.create(
        model=SYNTH_MODEL,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_completion_tokens=2048,
    )

    return {"answer": response.choices[0].message.content}
