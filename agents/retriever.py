"""Retriever node — wraps ChromaDB retrieval.

Supports two modes:

- **Injected** (set via `set_retrieval_context`): the Streamlit UI hands in
  the session's embedding model + ephemeral collection so each user queries
  only their own uploaded PDFs.
- **Default / fallback**: if nothing is injected, falls back to the
  persistent on-disk model + collection from `ingest.py`. Used by the eval
  harness and CLI scripts.
"""

from functools import lru_cache

from ingest import get_chroma_collection, load_embedding_model, retrieve

from .state import GraphState

_MODEL = None
_COLLECTION = None


def set_retrieval_context(model, collection) -> None:
    """Inject a specific embedding model + Chroma collection for
    subsequent `retrieve_chunks` calls. Call this from the UI layer."""
    global _MODEL, _COLLECTION
    _MODEL = model
    _COLLECTION = collection


@lru_cache(maxsize=1)
def _default_model():
    return load_embedding_model()


@lru_cache(maxsize=1)
def _default_collection():
    return get_chroma_collection()


def retrieve_chunks(state: GraphState) -> GraphState:
    query = state.get("retrieval_query") or state["question"]
    top_k = state.get("top_k", 5)

    model = _MODEL if _MODEL is not None else _default_model()
    collection = _COLLECTION if _COLLECTION is not None else _default_collection()

    chunks = retrieve(query, model, collection, top_k=top_k)
    return {"chunks": chunks}
