# Air India RAG Chatbot

A retrieval-augmented chatbot over five Air India PDFs (service regulations, fact
sheet, route maps, disaster history). Built for **accuracy** and **low latency**
(~3–5s/answer) on a **free** Gemini API tier, served as a **web app** (FastAPI + a
simple chat UI), **Dockerized**, with per-IP **rate limiting**. Terminal version too.

## Why RAG (not "stuff the PDF into the prompt")
The corpus is ~120 pages / 6.5 MB — far too large to put in every prompt, and the
two route documents are **map infographics** where plain text extraction loses the
route connections entirely. So we:
- extract route maps with **Gemini Vision** into structured data (the big accuracy win),
- index everything with **hybrid retrieval** (semantic + keyword) and **rerank**,
- answer **only** from retrieved context, with **citations**.

## Architecture
```
PDFs ─► per-type loaders + cleaning ─► structure-aware chunks ─┐
route MAPS ─► Gemini Vision ─► structured routes ─────────────┤
                                                              ▼
                          Gemini embeddings ─► Chroma (vectors) + BM25 (keywords)

query ─► history-aware rewrite ─► hybrid retrieve ─► rerank ─► Gemini 2.5 Flash ─► answer + citations
                 ▲                                                                      │
                 └──────────────── SQLite windowed memory (per session) ◄──────────────┘
```

## Models
- Embeddings: `gemini-embedding-001` (3072-dim, task-type aware)
- Chat + Vision: `gemini-2.5-flash` (thinking disabled for speed)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (light, fast on CPU; degrades gracefully)

## Setup
Use a virtual environment (keeps this project isolated from your other Python work):
```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (mac/linux: source .venv/bin/activate)
pip install -r requirements.txt
# put your key in .env:  GOOGLE_API_KEY=...
python scripts/test_key.py          # verify the key works
```

## Source documents (not included in the repo)
The PDFs themselves are **not committed** (they're third-party/copyrighted material).
The app doesn't read them at runtime — they're only used once to build the index.
To run locally, drop the source PDFs into the project root, then build the index below.
The repo ships everything else (code + the Vision-extracted route data in `data/`).

## Build the index (run once; re-run when PDFs change)
```bash
python -m src.maps_extract          # Gemini Vision -> data/routes_*.json
python -m src.ingest                # embed -> chroma_db/ + data/bm25.pkl
```
> Free-tier embeddings are capped at ~100 requests/min, so ingestion is paced and
> takes a few minutes. This is automatic.

## Chat
Web app (recommended):
```bash
python -m src.server                # then open http://127.0.0.1:8000
```
Command line:
```bash
python main.py                      # terminal chat (LangChain RAG: ensemble + rerank, SQL memory)
```

## Run with Docker
The image bakes in the pre-built index + reranker; the API key is passed at runtime
(never baked in). PDFs are not needed at runtime.
```bash
docker build -t air-india-chatbot .
docker run -p 8000:8000 --env-file .env air-india-chatbot   # http://127.0.0.1:8000
```

## Safety / abuse protection
- Per-IP **rate limit** (12 req/min) and **1000-char input cap** on `/chat`.
- Runs as a **non-root** container; API key only via env at runtime.
- Tunable in `config.py` (`RATE_LIMIT_*`, `MAX_MESSAGE_CHARS`).

## Evaluate (regression guard)
```bash
python scripts/eval.py              # runs the golden-question set (5/5)
```

## Layout
```
main.py              entry point -> terminal chat
Dockerfile           container image (index + reranker baked in)
DEPLOY.md            Docker + AWS EC2 deployment guide
src/server.py        FastAPI web server (UI + streaming /chat + rate limiting)
static/index.html    web chat UI (AIR INDIA CHAT BOT)
config.py            paths, model names, knobs, rate-limit settings
src/maps_extract.py  Gemini Vision route extraction
src/clean.py         per-source text cleaning
src/loaders.py       PDF loading + structure-aware chunking
src/embeddings.py    Gemini embeddings (rate-limited, task-type aware)
src/ingest.py        build Chroma + BM25
src/lc_chain.py      LangChain RAG chain: EnsembleRetriever + CrossEncoder rerank +
                     history-aware retrieval + RunnableWithMessageHistory (SQL memory)
src/lc_cli.py        chat loop used by main.py
scripts/test_key.py  key smoke test
scripts/eval.py      golden-set evaluation (runs through the LangChain chain)
```

## Known limits
- **Free-tier quotas:** `gemini-2.5-flash` allows ~20 generations/day; embeddings
  ~100/min (ingestion is paced for this). Set `CHAT_MODEL = "gemini-2.0-flash"` in
  `config.py` for more daily headroom.
- **Route maps:** extracted from dense map infographics via Gemini Vision, so route
  data can have occasional gaps. Fleet/regulation/history answers are reliable.

## Deploy
**Deployed on AWS EC2 (`t3.small`) via Docker.** The image
(`travisscotch/air-india-chatbot`) bundles the pre-built index + reranker; the API key is
supplied on the server at runtime (never in the image or git). Full steps — push to a
registry, launch EC2, open ports 22/8000, pull & run — are in `DEPLOY.md`.

> The public URL is a plain EC2 IP on port 8000 and may be offline when the instance is
> stopped (to save cost). For a permanent address, attach an Elastic IP; for HTTPS, front
> it with a reverse proxy / load balancer + certificate.

## Possible next steps
- Re-OCR the regulations PDF (it has OCR typos) for even cleaner retrieval.
- Semantic cache for repeated questions.
- For multi-instance scaling: shared chat history (RDS) + Redis-backed rate limiting.
- HTTPS via a reverse proxy (Caddy/Nginx) or AWS ALB + ACM.
