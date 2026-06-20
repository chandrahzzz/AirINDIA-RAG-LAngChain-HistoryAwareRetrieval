"""FastAPI web server for the Air India chatbot.

Serves the chat UI (static/index.html) and a streaming /chat endpoint backed by
the LangChain RAG chain. Conversation memory is per browser session_id.

Run:  python -m src.server      (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import config
from src import lc_chain

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_chain = None  # built at startup (loads reranker, connects index)


def get_chain():
    global _chain
    if _chain is None:
        config.require_key()
        _chain = lc_chain.build_chain()
    return _chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up at startup so the FIRST user question doesn't pay the ~10s
    # reranker-load / index-connect cold start.
    print("Warming up the RAG chain (loading reranker, connecting index) ...")
    get_chain()
    print("Ready -> http://127.0.0.1:8000")
    yield


app = FastAPI(title="Air India Chat Bot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Per-IP rate limiting (sliding window, in-memory, thread-safe) ----------
_rl_lock = threading.Lock()
_rl_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Behind a load balancer (AWS App Runner / ALB) the real IP is in X-Forwarded-For.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    window = config.RATE_LIMIT_WINDOW_SEC
    with _rl_lock:
        hits = _rl_hits[ip]
        while hits and now - hits[0] > window:   # drop timestamps outside the window
            hits.popleft()
        if len(hits) >= config.RATE_LIMIT_REQUESTS:
            return True
        hits.append(now)
        if not hits:
            _rl_hits.pop(ip, None)               # tidy empty entries
        return False


def _is_quota_error(msg: str) -> bool:
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _is_transient(msg: str) -> bool:
    # Gemini 5xx / deadline errors are transient and worth retrying.
    return any(t in msg for t in ("ServerError", "500", "503", "internal", "unavailable",
                                  "deadline", "DEADLINE"))


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    # Abuse protection: reject overlong inputs and throttle per IP.
    if len(req.message) > config.MAX_MESSAGE_CHARS:
        return PlainTextResponse(
            "Your message is too long. Please shorten it and try again.",
            status_code=413,
        )
    if _rate_limited(_client_ip(request)):
        return PlainTextResponse(
            "You're sending requests too quickly. Please wait a moment and try again.",
            status_code=429,
        )

    chain = get_chain()
    cfg = {"configurable": {"session_id": req.session_id}}

    # Greetings/thanks skip the whole RAG pipeline -> instant reply.
    quick = lc_chain.smalltalk_reply(req.message)
    if quick is not None:
        return StreamingResponse(iter([quick]), media_type="text/plain; charset=utf-8")

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
