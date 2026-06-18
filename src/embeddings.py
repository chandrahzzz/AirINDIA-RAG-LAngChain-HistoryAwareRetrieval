"""Gemini embedding helpers, shared by ingestion and retrieval.

gemini-embedding-001 supports a `task_type` hint. Using RETRIEVAL_DOCUMENT when
indexing and RETRIEVAL_QUERY when searching measurably improves match quality,
because the model embeds documents and questions into aligned-but-distinct spaces.
"""
from __future__ import annotations

import re
import time

from google import genai
from google.genai import types

import config

# Free tier allows ~100 embed requests/min, and each item counts as one request.
# Stay safely under it and back off when the API tells us to.
_RPM_LIMIT = 90
_BATCH = 50
_client: genai.Client | None = None
_last_batch_start = 0.0


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_key())
    return _client


def _retry_delay_seconds(err: Exception) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(r"'retryDelay': '([\d.]+)s'", str(err))
    return float(m.group(1)) + 1.0 if m else 30.0


def _embed_batch(batch: list[str], task_type: str) -> list[list[float]]:
    """Embed one batch, honoring the per-minute rate limit and 429 backoff."""
    global _last_batch_start
    # Proactive pacing: spend at least (len(batch)/RPM*60)s per batch.
    min_interval = len(batch) / _RPM_LIMIT * 60.0
    wait = min_interval - (time.monotonic() - _last_batch_start)
    if wait > 0:
        time.sleep(wait)

    for attempt in range(6):
        _last_batch_start = time.monotonic()
        try:
            resp = client().models.embed_content(
                model=config.EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            delay = _retry_delay_seconds(e)
            print(f"  [rate-limit] waiting {delay:.0f}s (attempt {attempt + 1}/6) ...")
            time.sleep(delay)
    raise RuntimeError("Embedding failed after repeated rate-limit backoffs.")


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        out.extend(_embed_batch(texts[i:i + _BATCH], task_type))
    return out


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embed([text], "RETRIEVAL_QUERY")[0]
