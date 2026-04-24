"""Router agent — classifies a question and picks retrieval parameters.

Uses Llama 3.1 8B Instant via Groq for speed. Output is a JSON object
validated against `RouterDecision` (Pydantic).
"""

import json
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from .state import GraphState

ROUTER_MODEL = "llama-3.1-8b-instant"

ROUTER_SYSTEM = """You are the routing layer of a research-paper Q&A system.

Classify the user's question into one of three types, and decide how many \
chunks the retriever should pull.

Types:
- factual: a single specific fact lookup ("what is the learning rate they used?"). \
  Usually one source, one page. top_k=4.
- multihop: requires combining information from multiple passages or pages of \
  the same paper ("how does the method's training compare to its evaluation?"). \
  top_k=6.
- comparison: compares claims, methods, or results across two or more papers \
  ("how does X's approach to attention differ from Y's?"). top_k=8.

Also emit a `retrieval_query`: a short, keyword-dense rewrite optimized for \
semantic search. Drop conversational filler, expand abbreviations, include \
domain terms the user implied.

Respond with ONLY a JSON object. No prose before or after. Schema:
{
  "question_type": "factual" | "multihop" | "comparison",
  "top_k": integer,
  "retrieval_query": string
}"""


class RouterDecision(BaseModel):
    question_type: Literal["factual", "multihop", "comparison"]
    top_k: int = Field(ge=1, le=20)
    retrieval_query: str


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(max_retries=5, timeout=120.0)
    return _client


def route(state: GraphState) -> GraphState:
    question = state["question"]

    response = _get_client().chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=256,
    )

    raw = response.choices[0].message.content
    try:
        decision = RouterDecision.model_validate_json(raw)
    except ValidationError:
        # Fallback if the model wraps JSON in prose — extract first object
        start, end = raw.find("{"), raw.rfind("}") + 1
        decision = RouterDecision.model_validate(json.loads(raw[start:end]))

    return {
        "question_type": decision.question_type,
        "top_k": decision.top_k,
        "retrieval_query": decision.retrieval_query,
        "attempts": 0,
        "max_attempts": 2,
    }
