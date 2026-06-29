"""FastAPI web server for the Air India chatbot.

Serves the chat UI (static/index.html) and a /chat endpoint backed by the LangGraph
agentic graph (RAG answers + interested-lead capture). Also serves an admin page to
approve/reject captured leads (the human-in-the-loop gate). State is per session_id.

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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import config
from src import graph as graph_mod
from src import lc_chain, leads

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_graph = None  # built at startup (loads reranker, connects index, compiles graph)


def get_graph():
    global _graph
    if _graph is None:
        config.require_key()
        _graph = graph_mod.build_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up at startup so the FIRST user question doesn't pay the ~10s
    # reranker-load / index-connect cold start.
    print("Warming up the agentic graph (loading reranker, connecting index) ...")
    get_graph()
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

    # Greetings/thanks skip the whole pipeline -> instant reply.
    quick = lc_chain.smalltalk_reply(req.message)
    if quick is not None:
        return StreamingResponse(iter([quick]), media_type="text/plain; charset=utf-8")

    graph = get_graph()
    cfg = {"configurable": {"thread_id": req.session_id}}

    def generate():
        # The graph runs to completion (router -> rag or lead-capture), then we stream
        # the final answer out in chunks. Retries transient 5xx before sending anything.
        for attempt in range(3):
            try:
                state = graph.invoke({"messages": [HumanMessage(req.message)]}, config=cfg)
                answer = state["messages"][-1].content or ""
                for i in range(0, len(answer), 24):
                    yield answer[i:i + 24]
                return
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if _is_quota_error(msg):
                    yield ("The free Gemini quota for today has been reached. "
                           "Please try again later.")
                    return
                if _is_transient(msg) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))  # backoff, then retry fresh
                    continue
                yield f"Sorry, something went wrong: {type(e).__name__}."
                return

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# --- Admin: human-in-the-loop lead approval --------------------------------
def _check_admin(token: str | None):
    if token != config.ADMIN_TOKEN:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/admin/leads")
def admin_leads(token: str = ""):
    if (err := _check_admin(token)) is not None:
        return err
    fmt = lambda L: [vars(x) for x in L]  # noqa: E731
    return {"pending": fmt(leads.list_pending()), "approved": fmt(leads.list_approved())}


@app.post("/admin/leads/{lead_id}/approve")
def admin_approve(lead_id: int, token: str = ""):
    if (err := _check_admin(token)) is not None:
        return err
    return {"ok": leads.approve(lead_id)}


@app.post("/admin/leads/{lead_id}/reject")
def admin_reject(lead_id: int, token: str = ""):
    if (err := _check_admin(token)) is not None:
        return err
    return {"ok": leads.reject(lead_id)}


if __name__ == "__main__":
    print("Air India Chat Bot -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
