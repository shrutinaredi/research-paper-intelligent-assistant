"""
test_ingest.py — Test that retrieval works after ingesting a PDF.

Run this after ingesting at least one PDF:
  python test_ingest.py
"""

from ingest import load_embedding_model, get_chroma_collection, retrieve

model = load_embedding_model()
collection = get_chroma_collection()

print(f"\nChromaDB contains {collection.count()} chunks\n")

# Try a test query
query = "what is attention mechanism in transformers?"
print(f"Query: {query}\n")

results = retrieve(query, model, collection, top_k=3)

for i, chunk in enumerate(results):
    print(f"--- Result {i+1} (score: {chunk['score']}) ---")
    print(f"Source: {chunk['source']}  |  Page: {chunk['page']}")
    print(chunk['text'][:300])  # show first 300 chars
    print()
