# How My Air India RAG Chatbot Works

A short, precise reference: what it does, the tech stack and how each piece is used,
how the files fit together, and the key changes I made.

---

## What it is
A **RAG (Retrieval-Augmented Generation) chatbot** answering questions about Air India
(fleet, routes, service regulations, history) from 5 source PDFs. It retrieves only the
relevant passages per question and answers **grounded in them, with citations**. Served
as a **FastAPI web app** ("AIR INDIA CHAT BOT") and runs in **Docker**. Uses **Gemini**
on the free tier.

---

## Key features
- **Hybrid retrieval.** Every question is searched two ways at once — **semantic** (vector
  similarity in Chroma, catches paraphrases) and **keyword** (BM25, catches exact tokens
  like "A321neo" or clause numbers). The two ranked lists are fused, so I get the strengths
  of both instead of relying on one.
- **Cross-encoder reranking.** The fused candidates are re-scored by a cross-encoder and
  trimmed to the best 8, so the model only sees the most relevant chunks — better answers
  and a smaller, faster prompt.
- **Gemini Vision route extraction.** The two route PDFs are *map infographics* — plain
  text extraction yields a useless list of city names. I run them through Gemini Vision to
  recover the real destinations, routes, and notes as structured data. This is the biggest
  accuracy win and the part a normal "pypdf + LLM" bot gets wrong.
- **Grounded answers with real citations.** A strict prompt makes the model answer **only**
  from retrieved context (no hallucination), and each chunk's true source is fed in via
  `document_prompt`, so citations are real — e.g.
  `[Air India Service Regulations, CHAPTER IV - RETIREMENT, p16]`, not guessed.
- **Accurate regulation citations.** Regulation chunks are tagged with their CHAPTER using
  OCR-tolerant detection (the scan mangles "CHAPTER IV" into "GHAPTER"/"Xl"), so a retrieved
  clause cites the correct chapter and page.
- **Complete route answers.** "List/count all flights from X" needs *every* route, which
  top-k retrieval can't guarantee — so for route-style questions I inject the full route
  list, making those answers complete.
- **Conversational memory.** Follow-ups like "and how many are on order?" work: a
  history-aware step rewrites them into standalone questions before retrieving, and each
  conversation is stored per session in SQLite (so it survives restarts).
- **Instant greetings.** "hi"/"thanks"/"bye" are detected and answered immediately, skipping
  the whole retrieval pipeline — no point embedding/reranking a greeting.
- **Fast (~3–5s/answer).** Lighter reranker, model "thinking" disabled, chain warmed up at
  startup, and fewer rerank candidates — all without lowering retrieval quality (eval 5/5).
- **Streaming web UI.** A clean FastAPI-served page ("AIR INDIA CHAT BOT") streams the
  answer token-by-token, keeping a session id in the browser. A terminal version exists too.
- **Abuse protection.** Per-IP rate limiting (12 req/min) and a 1000-char input cap on
  `/chat`, the API key supplied only at runtime, and the container runs as a non-root user.
- **Reliable & reproducible.** Transient Gemini 5xx errors are retried, there's a
  healthcheck + auto-restart, and the whole app is Dockerized so production behaves exactly
  like local.

---

## Tech stack and how it's used
| Tech | How I use it | Where |
|---|---|---|
| **Gemini (`google-genai`)** | `gemini-2.5-flash` (thinking off) writes answers + reads route PDFs; `gemini-embedding-001` makes 3072-dim vectors | `embeddings.py`, `maps_extract.py`, `lc_chain.py` |
| **Gemini Vision** | extracts destinations/routes/notes from the map infographics as JSON | `maps_extract.py` |
| **Chroma** | local vector DB for semantic search | `ingest.py`, `lc_chain.py` |
| **BM25 (`rank_bm25`)** | keyword search for exact terms (codes, clause numbers) | `ingest.py`, `lc_chain.py` |
| **LangChain (1.x + classic)** | orchestrates retrieval → history rewrite → grounded answer → memory | `lc_chain.py` |
| **Cross-encoder reranker** | `ms-marco-MiniLM-L-6-v2` reranks candidates to the best 8 (light + fast on CPU) | `lc_chain.py` |
| **SQLite** | per-session conversation history | `lc_chain.py` |
| **FastAPI + Uvicorn** | web server: serves the UI + streaming `/chat`, rate limiting | `server.py` |
| **HTML/CSS/JS** | the chat UI (bubbles, streaming, session id) | `static/index.html` |
| **Docker** | packages app + pre-built index + reranker so prod == local | `Dockerfile` |
| **pypdf / dotenv / venv** | PDF text extraction / load API key from `.env` / isolated deps | `loaders.py`, `config.py` |

---

## How each file works (brief)
**App / serving**
- `src/server.py` — FastAPI app: serves the UI, streaming `/chat`, **per-IP rate limit + length cap**, warms up the chain at startup, retries transient 5xx.
- `static/index.html` — chat UI; streams answers, keeps a session id in localStorage.
- `main.py` — terminal chat entry point.
- `config.py` — paths, model names, and knobs (chunk size, `RETRIEVE_K=12`, `RERANK_TOP_N=8`, memory window, rate-limit settings). Loads the key.

**Build the index (run once)**
- `src/maps_extract.py` — Gemini Vision → `data/routes_*.json` + `routes_extracted.txt`.
- `src/clean.py` — strips glyphs/web-nav junk, fixes hyphenation.
- `src/loaders.py` — per-type chunking; tags regulation chunks with their **CHAPTER** (OCR-tolerant) and builds a clean **citation label**.
- `src/embeddings.py` — Gemini embeddings (task-type aware) with rate-limit/backoff.
- `src/ingest.py` — loads → embeds → Chroma + BM25 index.

**Answering**
- `src/lc_chain.py` — the RAG chain: hybrid `EnsembleRetriever` → reranker → history-aware rewrite → grounded answer (`document_prompt` injects the real citation) → SQL memory; **full route list injected for "list-all" route queries**; **smalltalk gate** for greetings.
- `src/lc_cli.py` — terminal chat loop.

**Scripts**
- `scripts/test_key.py` — verifies the Gemini key.
- `scripts/eval.py` — golden-set regression check (5/5).

---

## End-to-end flow
1. Build once: `python -m src.maps_extract` → `python -m src.ingest`.
2. Run: `python -m src.server` (or `docker run`) → `http://127.0.0.1:8000`.
3. Per question: greeting? → instant reply. Else: history-aware rewrite → hybrid retrieve (12) → rerank (8) → Gemini (grounded) → stream answer + citations → save turn to SQLite.
4. Quality check: `python scripts/eval.py` → 5/5.

---

## Key changes I made (latest session)
- **Cut latency from ~10–40s to ~3–5s — without losing quality.** I measured the pipeline
  and found the bottleneck was the **reranker**, not the LLM. Fixes:
  - Swapped the heavy `bge-reranker-base` (278M params) for the lighter
    `ms-marco-MiniLM-L-6-v2` (22M) — ~10× faster on CPU and far less RAM.
  - Disabled the model's internal "thinking" (`thinking_budget=0`) — grounded answers
    don't need it, and it roughly halved generation time.
  - Warmed up the chain at server startup so the *first* question no longer pays a ~10s
    cold start.
  - Reranked fewer candidates (`RETRIEVE_K` 20→12).
  - I verified **eval stayed 5/5** and the retirement citation stayed correct after each
    change, so none of this lowered accuracy.
- **Smalltalk gate.** Greetings/thanks/bye are answered instantly and never touch retrieval
  or the LLM — so a simple "hi" went from ~15s to instant.
- **Abuse protection (security).** Added per-IP **rate limiting** (12 requests/min, sliding
  window, reads `X-Forwarded-For` so it works behind a load balancer) and a **1000-char
  input cap**, returning friendly messages instead of errors. All tunable in `config.py`.
- **Dockerized for deployment.** A `Dockerfile` (CPU-only torch, runs as **non-root**,
  healthcheck) bakes the **pre-built index + reranker** into the image; the **API key is
  passed at runtime**, never baked in. Added `DEPLOY.md` (EC2 / App Runner steps). The PDFs
  aren't needed at runtime, so they stay off git and out of the image.
- **Accuracy fixes (earlier this work).** OCR-tolerant chapter tagging so regulation
  citations point to the right chapter/page, and full-route-list injection so "list-all"
  route questions are complete.

---

## Deployment (Docker)
The **image contains the pre-built index + reranker**, so it runs with no ingestion.
The **PDFs are NOT needed at runtime** (their content lives in the index) — they stay on
my machine as build inputs, never in git or the image. The **API key is supplied at
runtime** (`--env-file`/`-e`), never baked in. Target AWS EC2 `t3.small`/`t3.medium`
(or App Runner). Full steps in `DEPLOY.md`.

---

## Known limits
- **Free-tier quotas:** `gemini-2.5-flash` ≈ 20 generations/day; embeddings ≈ 100/min (ingestion is paced). Switch to `gemini-2.0-flash` or enable billing for more.
- **Route maps:** read from dense infographics, so route data can have rare gaps; fleet/regulation answers are reliable.
- **Memory/rate-limit are per-instance:** for multi-instance scaling, move history to a shared DB and rate-limiting to Redis.
