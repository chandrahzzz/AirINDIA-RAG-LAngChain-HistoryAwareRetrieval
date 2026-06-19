"""FastAPI web server for the Air India chatbot.

Serves the chat UI (static/index.html) and a streaming /chat endpoint backed by
the LangChain RAG chain. Conversation memory is per browser session_id.

Run:  python -m src.server      (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
from src import lc_chain

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Air India Chat Bot")
_chain = None  # built once on first use (loads reranker, connects index)


def get_chain():
    global _chain
    if _chain is None:
        config.require_key()
        _chain = lc_chain.build_chain()
    return _chain


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


def _is_quota_error(msg: str) -> bool:
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _is_transient(msg: str) -> bool:
    # Gemini 5xx / deadline errors are transient and worth retrying.
    return any(t in msg for t in ("ServerError", "500", "503", "internal", "unavailable",
                                  "deadline", "DEADLINE"))


@app.post("/chat")
def chat(req: ChatRequest):
    chain = get_chain()
    cfg = {"configurable": {"session_id": req.session_id}}

    def generate():
        # Up to 3 attempts. We only retry BEFORE any token has been sent, so the
        # client never sees duplicated/garbled text. The LLM also retries 5xx
        # internally (max_retries); this guards the streaming layer (Fix 2).
        for attempt in range(3):
            emitted = False
            try:
                for chunk in chain.stream({"input": req.message}, config=cfg):
                    token = chunk.get("answer", "")
                    if token:
                        emitted = True
                        yield token
                return  # finished cleanly
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if _is_quota_error(msg):
                    yield ("\n\n[The free Gemini quota for today has been reached. "
                           "Please try again later.]")
                    return
                if emitted:
                    # Already streamed part of the answer — can't safely restart.
                    yield "\n\n[The connection was interrupted. Please resend your question.]"
                    return
                if _is_transient(msg) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))  # backoff, then retry fresh
                    continue
                yield f"\n\n[Sorry, something went wrong: {type(e).__name__}.]"
                return

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    print("Air India Chat Bot -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
