"""Critic agent — verifies that every claim in the answer is supported
by a retrieved chunk, and decides whether to trigger a retry.

Uses Llama 3.1 8B Instant via Groq. Output is JSON validated against
`CriticOutput` (Pydantic) so the graph can route on `needs_retry`.
"""

import json

from groq import Groq
from pydantic import BaseModel, ValidationError

from .state import GraphState

CRITIC_MODEL = "llama-3.1-8b-instant"

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
and should not trigger a retry.

Respond with ONLY a JSON object. No prose before or after. Schema:
{
  "supported": bool,
  "unsupported_claims": [string, ...],
  "needs_retry": bool,
  "refined_query": string or null
}"""


class CriticOutput(BaseModel):
    supported: bool
    unsupported_claims: list[str]
    needs_retry: bool
    refined_query: str | None


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(max_retries=5, timeout=120.0)
    return _client


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[source={c['source']} page={c['page']}]\n{c['text']}" for c in chunks
    )


MAX_CRITIC_CHUNKS = 5  # Keeps the critic call under the 12K TPM free-tier bucket


def critique(state: GraphState) -> GraphState:
    question = state["question"]
    chunks = state.get("chunks", [])
    answer = state.get("answer", "")

    # Chunks come back score-sorted; the lowest-ranked ones rarely carry
    # claims the synthesizer actually used, so trimming is safe for verification.
    critic_chunks = chunks[:MAX_CRITIC_CHUNKS]

    user_content = (
        f"Question: {question}\n\n"
        f"Excerpts:\n{_format_chunks(critic_chunks)}\n\n"
        f"Answer to verify:\n{answer}"
    )

    response = _get_client().chat.completions.create(
        model=CRITIC_MODEL,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=1024,
    )

    raw = response.choices[0].message.content
    try:
        parsed = CriticOutput.model_validate_json(raw)
    except ValidationError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = CriticOutput.model_validate(json.loads(raw[start:end]))

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
