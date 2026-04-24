"""Router agent — classifies a question and picks retrieval parameters.

Uses Gemini 2.5 Flash with thinking disabled: this is a fast, cheap
structured-output call, not a reasoning task. The output is parsed into
a `RouterDecision` Pydantic model.
"""

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .state import GraphState

ROUTER_MODEL = "gemini-2.5-flash-lite"

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
domain terms the user implied."""


class RouterDecision(BaseModel):
    question_type: Literal["factual", "multihop", "comparison"]
    top_k: int = Field(ge=1, le=20)
    retrieval_query: str


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def route(state: GraphState) -> GraphState:
    question = state["question"]

    response = _get_client().models.generate_content(
        model=ROUTER_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM,
            response_mime_type="application/json",
            response_schema=RouterDecision,
        ),
    )

    decision: RouterDecision = response.parsed

    return {
        "question_type": decision.question_type,
        "top_k": decision.top_k,
        "retrieval_query": decision.retrieval_query,
        "attempts": 0,
        "max_attempts": 2,
    }
