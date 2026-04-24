"""Shared state for the LangGraph pipeline.

The graph threads a single dict through every node. Each node reads the
fields it needs and returns a partial dict that LangGraph merges back in.
"""

from typing import Literal, TypedDict

QuestionType = Literal["factual", "multihop", "comparison"]


class Chunk(TypedDict):
    text: str
    source: str
    page: int
    score: float


class CriticVerdict(TypedDict):
    supported: bool               # did every sentence land on a retrieved chunk?
    unsupported_claims: list[str] # sentences the critic couldn't ground
    needs_retry: bool             # should we re-retrieve with a refined query?
    refined_query: str | None     # critic's suggested re-query, if any


class GraphState(TypedDict, total=False):
    # Input
    question: str

    # Router output
    question_type: QuestionType
    top_k: int
    retrieval_query: str   # may differ from `question` after critic retries

    # Retrieval output
    chunks: list[Chunk]

    # Synthesizer output
    answer: str

    # Critic output
    critic: CriticVerdict

    # Loop control
    attempts: int          # how many retrieval rounds we've run
    max_attempts: int      # hard cap before we give up and return as-is
