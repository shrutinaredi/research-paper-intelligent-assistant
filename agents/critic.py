"""Critic agent — verifies that every claim in the answer is supported
by a retrieved chunk, and decides whether to trigger a retry.

Uses Gemini 2.5 Flash with thinking disabled — verification, not
generation. Output is a structured `CriticOutput` so the graph can
route on it without string parsing.
"""

from google import genai
from google.genai import types
from pydantic import BaseModel

from .state import GraphState

CRITIC_MODEL = "gemini-2.5-flash-lite"

CRITIC_SYSTEM = """You are a verifier for a research-paper Q&A system.

You will receive:
1. A question
2. A list of retrieved excerpts (with source names and page numbers)
3. A generated answer that cites those excerpts

Your job: decide whether every factual claim in the answer is actually \
supported by the excerpts.

A claim is SUPPORTED if an excerpt contains the information it asserts \
(wording does not need to match — the meaning does) AND the citation \
[source, p.N] in the answer matches a real excerpt.

A claim is UNSUPPORTED if no excerpt contains the information, or if the \
citation points at a source/page that isn't in the excerpts.

If any claim is unsupported, set needs_retry=true and emit a refined_query \
— a short reformulation of the question that might surface the missing \
information (more specific terms, a different paper's name). If everything \
checks out, needs_retry=false and refined_query=null.

Exception: an answer that says "the excerpts do not contain enough \
information" is considered supported (the model correctly abstained) \
and should not trigger a retry."""


class CriticOutput(BaseModel):
    supported: bool
    unsupported_claims: list[str]
    needs_retry: bool
    refined_query: str | None


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[source={c['source']} page={c['page']}]\n{c['text']}" for c in chunks
    )


def critique(state: GraphState) -> GraphState:
    question = state["question"]
    chunks = state.get("chunks", [])
    answer = state.get("answer", "")

    user_content = (
        f"Question: {question}\n\n"
        f"Excerpts:\n{_format_chunks(chunks)}\n\n"
        f"Answer to verify:\n{answer}"
    )

    response = _get_client().models.generate_content(
        model=CRITIC_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=CRITIC_SYSTEM,
            response_mime_type="application/json",
            response_schema=CriticOutput,
        ),
    )

    parsed: CriticOutput = response.parsed

    attempts = state.get("attempts", 0) + 1
    max_attempts = state.get("max_attempts", 2)
    needs_retry = parsed.needs_retry and attempts < max_attempts

    update: GraphState = {
        "critic": {
            "supported": parsed.supported,
            "unsupported_claims": parsed.unsupported_claims,
            "needs_retry": needs_retry,
            "refined_query": parsed.refined_query,
        },
        "attempts": attempts,
    }

    if needs_retry and parsed.refined_query:
        update["retrieval_query"] = parsed.refined_query

    return update
