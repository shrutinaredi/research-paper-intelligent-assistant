"""LangGraph state machine wiring.

Topology:

    router -> retriever -> synthesizer -> critic -> {END | retriever}

The critic's `needs_retry` flag decides whether to loop back. The
critic itself caps retries (see `max_attempts` in state), so the loop
can't run forever even if the critic keeps flagging.
"""

import time
import uuid

from langgraph.graph import END, StateGraph

from .critic import critique
from .retriever import retrieve_chunks
from .router import route
from .state import GraphState
from .synthesizer import synthesize
from .telemetry import log_run


def _should_retry(state: GraphState) -> str:
    critic = state.get("critic", {})
    return "retriever" if critic.get("needs_retry") else "end"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", route)
    graph.add_node("retriever", retrieve_chunks)
    graph.add_node("synthesizer", synthesize)
    graph.add_node("critic", critique)

    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_retry,
        {"retriever": "retriever", "end": END},
    )

    return graph.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(question: str) -> dict:
    """Run the full pipeline on a question. Returns the final state
    plus a run_id and latency for telemetry."""
    run_id = str(uuid.uuid4())
    t0 = time.monotonic()

    result = _get_graph().invoke({"question": question})

    latency_ms = int((time.monotonic() - t0) * 1000)

    log_run(
        run_id=run_id,
        question=question,
        question_type=result.get("question_type"),
        retrieval_query=result.get("retrieval_query"),
        top_k=result.get("top_k"),
        attempts=result.get("attempts"),
        chunks=result.get("chunks", []),
        answer=result.get("answer", ""),
        critic=result.get("critic", {}),
        latency_ms=latency_ms,
    )

    return {
        "run_id": run_id,
        "latency_ms": latency_ms,
        **result,
    }
