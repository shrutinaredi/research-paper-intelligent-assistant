"""Retriever node — wraps the existing ChromaDB retrieval in ingest.py.

Reuses the embedding model + collection singletons so we don't reload
the model on every call (it's ~80MB and takes a few seconds).
"""

from functools import lru_cache

from ingest import get_chroma_collection, load_embedding_model, retrieve

from .state import GraphState


@lru_cache(maxsize=1)
def _model():
    return load_embedding_model()


@lru_cache(maxsize=1)
def _collection():
    return get_chroma_collection()


def retrieve_chunks(state: GraphState) -> GraphState:
    query = state.get("retrieval_query") or state["question"]
    top_k = state.get("top_k", 5)

    chunks = retrieve(query, _model(), _collection(), top_k=top_k)
    return {"chunks": chunks}
