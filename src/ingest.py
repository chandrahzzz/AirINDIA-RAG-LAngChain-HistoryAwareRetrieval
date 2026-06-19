"""Build the search indexes from the PDFs.

Run:  python -m src.ingest

Creates:
  chroma_db/            persisted Chroma collection (Gemini embeddings)
  data/bm25.pkl         pickled BM25 index + chunk texts/metadata for hybrid search

Re-run any time the PDFs or extraction change; it rebuilds from scratch.
"""
from __future__ import annotations

import pickle
import re
import shutil

import chromadb
from rank_bm25 import BM25Okapi

import config
from src import embeddings, loaders


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build() -> None:
    config.require_key()
    chunks = loaders.load_all()
    if not chunks:
        raise SystemExit("No chunks produced — check the PDFs are present.")

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]
    metas = [
        {
            "source": c.source, "doc_type": c.doc_type, "page": c.page,
            "section": c.section,
            "cite": loaders.make_cite(c.doc_type, c.page, c.section),
        }
        for c in chunks
    ]

    # --- Vector index (Chroma) ---
    print(f"[embed] embedding {len(texts)} chunks with {config.EMBED_MODEL} "
          f"(free-tier paced to ~90/min, so this takes a few minutes) ...")
    vectors = embeddings.embed_documents(texts)

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = chroma.create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    coll.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"[chroma] stored {coll.count()} vectors -> {config.CHROMA_DIR}")

    # --- Keyword index (BM25) ---
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "ids": ids, "metas": metas}, f)
    print(f"[bm25] indexed {len(texts)} chunks -> {config.BM25_PATH}")
    print("[done] ingestion complete.")


if __name__ == "__main__":
    build()
