"""
ingest.py — The Ingestion Pipeline
====================================
This file handles the entire process of:
  1. Reading a PDF file
  2. Extracting text from it
  3. Splitting the text into chunks
  4. Converting chunks into embeddings (vectors)
  5. Storing everything in ChromaDB

Run this file directly to ingest all PDFs in data/pdfs/:
  python ingest.py
"""

import os
import hashlib
import pdfplumber
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONSTANTS — tweak these to change behavior
# ─────────────────────────────────────────────

PDFS_DIR = "data/pdfs"          # Where to look for PDFs
CHROMA_DIR = "chroma_db"        # Where ChromaDB stores its files on disk
COLLECTION_NAME = "papers"      # Name of our ChromaDB collection (like a table)

CHUNK_SIZE = 500                # How many words per chunk (roughly)
CHUNK_OVERLAP = 50              # How many words overlap between consecutive chunks

# The embedding model we use — free, local, no API needed
# all-MiniLM-L6-v2 maps text to 384-dimensional vectors
# It's small (80MB) and fast while being high quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# STEP 1: PDF TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a PDF and returns a list of pages.
    Each page is a dict: { "page": 1, "text": "..." }

    We use pdfplumber because it handles tables and complex layouts
    better than basic pypdf.
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # extract_text() returns a string of all text on the page
            # It returns None if the page has no extractable text (e.g. scanned image)
            text = page.extract_text()

            if text and text.strip():  # skip empty pages
                pages.append({
                    "page": i + 1,       # 1-indexed page number
                    "text": text.strip() # remove leading/trailing whitespace
                })

    print(f"  Extracted {len(pages)} pages from {Path(pdf_path).name}")
    return pages


# ─────────────────────────────────────────────
# STEP 2: CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a long text into overlapping chunks.

    WHY CHUNK?
    We can't pass a 50-page PDF to Claude in one prompt — too many tokens.
    Instead we split it into small pieces and only retrieve the relevant ones.

    WHY OVERLAP?
    Imagine a sentence that spans two chunks: "...end of chunk 1. Start of chunk 2..."
    Without overlap, that sentence gets cut. With overlap, both chunks contain it.

    Example with chunk_size=10 words, overlap=3:
      Text:    [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17]
      Chunk 1: [1 2 3 4 5 6 7 8 9 10]
      Chunk 2:             [8 9 10 11 12 13 14 15 16 17]
                           ^^^overlap^^^
    """
    words = text.split()        # split text into individual words
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])  # join words back into a string
        chunks.append(chunk)
        start += chunk_size - overlap       # move forward, but back up by overlap amount

    return chunks


# ─────────────────────────────────────────────
# STEP 3: EMBEDDINGS
# ─────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformers embedding model.

    WHAT IS AN EMBEDDING?
    It converts text into a list of numbers (a vector) that captures meaning.
    Similar sentences get similar vectors.

    "transformer attention"    → [0.23, -0.41, 0.87, ...]  (384 numbers)
    "self-attention mechanism" → [0.21, -0.39, 0.85, ...]  (very close!)
    "banana bread recipe"      → [-0.91, 0.12, -0.34, ...]  (far away)

    The model downloads automatically on first use (~80MB).
    After that it's cached locally.
    """
    print("Loading embedding model (downloads on first run)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """
    Converts a list of text strings into a list of vectors.

    model.encode() returns a numpy array — we convert to plain Python lists
    because ChromaDB expects lists, not numpy arrays.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()  # convert numpy array → Python list


# ─────────────────────────────────────────────
# STEP 4: CHROMADB
# ─────────────────────────────────────────────

def get_chroma_collection() -> chromadb.Collection:
    """
    Creates (or opens) a ChromaDB collection stored on disk.

    WHAT IS CHROMADB?
    A vector database — like a regular database but designed to store
    and search vectors (embeddings) by similarity.

    PersistentClient = saves data to disk (survives app restarts).
    A "collection" is like a table — ours is called "papers".

    get_or_create_collection: if collection already exists, open it.
    If not, create a new empty one. This is safe to call repeatedly.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine similarity: measures angle between vectors (best for text)
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def make_chunk_id(pdf_name: str, page: int, chunk_index: int) -> str:
    """
    Creates a unique ID for each chunk.
    Format: "paper_name__page_3__chunk_7"

    ChromaDB requires each stored item to have a unique string ID.
    We use this deterministic format so re-ingesting a PDF doesn't
    create duplicates — it just overwrites existing chunks.
    """
    base = f"{pdf_name}__page_{page}__chunk_{chunk_index}"
    # hash it to avoid issues with special characters in filenames
    return hashlib.md5(base.encode()).hexdigest()


# ─────────────────────────────────────────────
# MAIN INGESTION FUNCTION
# ─────────────────────────────────────────────

def ingest_pdf(pdf_path: str, collection: chromadb.Collection, model: SentenceTransformer):
    """
    Full pipeline for one PDF:
      extract pages → chunk → embed → store in ChromaDB
    """
    pdf_name = Path(pdf_path).stem  # filename without extension, e.g. "attention_is_all_you_need"
    print(f"\nIngesting: {pdf_name}")

    # Step 1: Extract text page by page
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  WARNING: No text extracted from {pdf_name}. Skipping.")
        return

    # Step 2: Chunk each page's text
    all_chunks = []      # the text of each chunk
    all_metadatas = []   # info about each chunk (source, page number, etc.)
    all_ids = []         # unique ID for each chunk

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        chunks = chunk_text(page_text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(pdf_name, page_num, chunk_idx)

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": pdf_name,   # which paper this came from
                "page": page_num,     # which page
                "chunk": chunk_idx    # which chunk on that page
            })

    print(f"  Created {len(all_chunks)} chunks")

    # Step 3: Embed all chunks at once (batched = faster)
    print(f"  Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks, model)

    # Step 4: Store in ChromaDB
    # upsert = insert if new, update if ID already exists (safe to re-run)
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,      # the raw text (stored for retrieval)
        embeddings=embeddings,     # the vectors (stored for similarity search)
        metadatas=all_metadatas    # source/page info (stored for citations)
    )

    print(f"  Stored {len(all_chunks)} chunks in ChromaDB")


def ingest_all_pdfs(pdfs_dir: str = PDFS_DIR):
    """
    Ingests every PDF in the pdfs_dir folder.
    Called on app startup or when new PDFs are added.
    """
    pdf_files = list(Path(pdfs_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {pdfs_dir}/")
        print("Add PDF files to that folder and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to ingest")

    # Load the embedding model once (reused for all PDFs)
    model = load_embedding_model()

    # Open ChromaDB once (reused for all PDFs)
    collection = get_chroma_collection()

    for pdf_path in pdf_files:
        ingest_pdf(str(pdf_path), collection, model)

    total = collection.count()
    print(f"\nDone! ChromaDB now contains {total} total chunks.")


# ─────────────────────────────────────────────
# RETRIEVAL FUNCTION (used later by the agents)
# ─────────────────────────────────────────────

def retrieve(query: str, model: SentenceTransformer, collection: chromadb.Collection, top_k: int = 5) -> list[dict]:
    """
    Given a question, finds the top_k most relevant chunks.

    HOW IT WORKS:
    1. Embed the query into a vector
    2. Ask ChromaDB: "find me the 5 closest vectors to this one"
    3. ChromaDB returns the matching chunk texts + their metadata

    This is semantic search — it finds similar MEANING, not just keywords.
    """
    # Embed the query (same model as the chunks — must match!)
    query_embedding = model.encode([query]).tolist()[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
        # distances: lower = more similar (cosine distance)
    )

    # Reformat results into a clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "score": round(1 - results["distances"][0][i], 3)
            # convert distance → similarity score (1.0 = perfect match)
        })

    return chunks


# ─────────────────────────────────────────────
# RUN DIRECTLY TO INGEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ingest_all_pdfs()
