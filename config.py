"""Central configuration for the Air India RAG chatbot.

All paths are relative to the project root so the code runs the same on any machine.
Secrets come from the .env file (never hardcoded).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT                      # PDFs currently live in the project root
DATA_DIR = ROOT / "data"            # generated artifacts (extracted maps, etc.)
CHROMA_DIR = ROOT / "chroma_db"     # persisted vector store
BM25_PATH = DATA_DIR / "bm25.pkl"   # persisted keyword index
HISTORY_DB = DATA_DIR / "chat_history.sqlite"  # conversation memory

DATA_DIR.mkdir(exist_ok=True)

# --- Secrets ---
load_dotenv(ROOT / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# --- Models (verified available for this API key, June 2026) ---
EMBED_MODEL = "gemini-embedding-001"     # text-embedding-004 not available to this key
CHAT_MODEL = "gemini-2.5-flash"          # low-latency default generator
VISION_MODEL = "gemini-2.5-flash"        # multimodal extraction for route maps
RERANK_MODEL = "BAAI/bge-reranker-base"  # local cross-encoder

# --- Retrieval knobs ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVE_K = 20      # candidates pulled by hybrid search
RERANK_TOP_N = 5     # passed to the LLM after reranking

# --- Memory ---
MEMORY_WINDOW_TURNS = 6  # last N user/assistant exchanges kept verbatim

COLLECTION_NAME = "air_india"


def require_key() -> str:
    """Return the Gemini key or raise a clear, actionable error."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is empty. Add it to .env "
            "(get one at https://aistudio.google.com/app/apikey)."
        )
    return GOOGLE_API_KEY
