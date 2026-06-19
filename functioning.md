# How My Air India RAG Chatbot Works

My notes on exactly how the app works right now: the key features, the tech stack
and what each piece does, how every file works, and the end-to-end flow.

---

## Key features

- **RAG chatbot** over 5 Air India PDFs — it retrieves only the relevant passages
  per question instead of stuffing whole documents into the prompt.
- **Web chat app** (FastAPI + a simple HTML/JS frontend) titled **AIR INDIA CHAT BOT**,
  with streaming answers and a New-chat button. A terminal version also exists.
- **Gemini Vision route extraction** — the two route PDFs are map infographics, so I
  use Gemini Vision to read them into structured route data (the big accuracy win).
- **Hybrid retrieval + reranking** — semantic (vector) + keyword (BM25) search, then a
  cross-encoder reranker keeps the best chunks.
- **Conversational memory** — follow-ups ("and how many on order?") work via a
  history-aware rewrite; history is stored in SQLite and survives restarts.
- **Grounded answers with real citations** — answers come only from retrieved context,
  cited from actual metadata, e.g. `[Air India Service Regulations, CHAPTER IV - RETIREMENT, p16]`.
- **Robust** — retries transient Gemini 5xx errors so answers don't die mid-stream, and
  feeds the full route list for "list/count all" route questions so they're complete.

---

## What it is

A Retrieval-Augmented Generation chatbot answering questions about Air India from:

- `Aiesl Employees service regulation.pdf` — 94-page HR/service rules
- `Air India Fact Sheet.pdf` — fleet numbers, orders, facts
- `Domestic Routes Feb 2025.pdf` — map infographic
- `International Routes Feb 2025.pdf` — map infographic
- `List of Major Air India Disasters … Britannica.pdf` — web article

Built with **LangChain**, served as a **web app** (`python -m src.server`) on Gemini's free tier.

---

## Tech stack and what each piece does

| Tech | What it does in my project | Where |
|---|---|---|
| **Google Gemini (`google-genai` SDK)** | `gemini-2.5-flash` writes the answers and reads the route-map PDFs (vision); `gemini-embedding-001` turns text into 3072-dim vectors. | `src/embeddings.py`, `src/maps_extract.py`, `src/lc_chain.py` |
| **Gemini Vision** | Reads the two route-map infographics and extracts destinations + routes + notes as structured JSON. | `src/maps_extract.py` |
| **Chroma** | Local, persisted vector database for semantic similarity search. | `src/ingest.py`, `src/lc_chain.py` |
| **BM25 (`rank_bm25`)** | Keyword index — catches exact tokens like "A321neo" or clause numbers vectors miss. | `src/ingest.py`, `src/lc_chain.py` |
| **LangChain (1.x) + LangChain-Classic** | Orchestrates the RAG chain: hybrid retrieval → history-aware rewrite → grounded answer → memory. | `src/lc_chain.py` |
| **EnsembleRetriever** | Combines vector + BM25 results (hybrid retrieval). | `src/lc_chain.py` |
| **CrossEncoderReranker (`bge-reranker-base`)** | Re-scores ~20 candidates and keeps the best 8. Local, free. | `src/lc_chain.py` |
| **SQLite** | Stores conversation history per session; survives restarts. | `src/lc_chain.py` |
| **FastAPI + Uvicorn** | The web server: serves the chat page and a streaming `/chat` endpoint. | `src/server.py` |
| **HTML / CSS / JS** | The chat UI — message bubbles, streaming, session id in localStorage. | `static/index.html` |
| **pypdf** | Extracts text from the text-based PDFs. | `src/loaders.py` |
| **python-dotenv** | Loads `GOOGLE_API_KEY` from `.env` so the key isn't in code. | `config.py` |
| **venv** | Isolates this project's packages from my other Python projects. | `.venv/` |

---

## How each file works

### Entry points, config, secrets
- **`src/server.py`** — FastAPI web app. Serves `static/index.html` at `/`, and a
  streaming `/chat` endpoint that runs the LangChain chain per browser `session_id`.
  Retries transient 5xx before the first token and degrades gracefully on errors.
- **`main.py`** — terminal entry point (`python main.py`) for a CLI chat.
- **`config.py`** — paths, model names, and knobs (chunk size 1000, overlap 150,
  retrieve 20 candidates, rerank to top 8, memory window 6 turns). Loads the API key.
- **`.env` / `.env.example` / `.gitignore` / `requirements.txt`** — secrets, template,
  ignore rules (PDFs, `.env`, venv, index files), and pinned dependencies.

### Ingestion pipeline (build the index — run once)
- **`src/maps_extract.py`** — sends each route-map PDF to Gemini Vision and saves the
  routes as JSON (`data/routes_*.json`) plus a readable `data/routes_extracted.txt`.
- **`src/clean.py`** — strips bullet glyphs, fixes hyphenation, removes Britannica web
  navigation junk.
- **`src/loaders.py`** — loads each PDF by type and chunks it. For the regulations it
  tags each chunk with its **CHAPTER heading** using OCR-tolerant detection (handles
  "GHAPTER IV", "Xl"→"XI"), and builds a clean **citation label** per chunk.
- **`src/embeddings.py`** — Gemini embeddings with task types (document vs query) and
  **rate limiting + 429 backoff** for the free-tier ~100/min cap.
- **`src/ingest.py`** — loads → embeds → stores vectors in Chroma + builds the BM25
  index (`data/bm25.pkl`). Re-run when the PDFs change. (239 chunks.)

### The RAG chain
- **`src/lc_chain.py`** — the whole answering flow:
  - `EnsembleRetriever` (Chroma vector + BM25) → **hybrid retrieval**.
  - `CrossEncoderReranker` inside `ContextualCompressionRetriever` → **reranking** to top 8.
  - For **route/flight queries**, injects the complete route list so "list/count all"
    answers are complete.
  - `create_history_aware_retriever` → rewrites follow-ups into standalone questions.
  - `create_retrieval_chain` + `create_stuff_documents_chain` with a `document_prompt`
    that shows each chunk's **real citation label** → grounded answers with correct cites.
  - `RunnableWithMessageHistory` + windowed `SQLChatMessageHistory` → **memory**.
  - LLM is `ChatGoogleGenerativeAI` (Gemini 2.5 Flash) with `max_retries` + timeout.
- **`src/lc_cli.py`** — the terminal chat loop used by `main.py`.

### Scripts
- **`scripts/test_key.py`** — checks the Gemini key (one embed + one chat call).
- **`scripts/eval.py`** — golden-set evaluator (5 Q&A across fleet/routes/regulations);
  checks retrieval source + answer correctness. My regression guard (5/5 passing).

---

## End-to-end flow

**Setup (once):** make a venv, `pip install -r requirements.txt`, put the key in `.env`,
`python scripts/test_key.py`.

**Build the index (once, re-run if PDFs change):**
1. `python -m src.maps_extract` → Gemini Vision → structured routes.
2. `python -m src.ingest` → load + clean + chunk → embed → Chroma + BM25.

**Run the app:** `python -m src.server` → open `http://127.0.0.1:8000`.

**Per question:**
1. History-aware rewrite turns a follow-up into a standalone question.
2. Hybrid retrieval (vector + BM25) → ~20 candidates; reranker → best 8
   (route queries also get the full route list).
3. The chunks + question go to Gemini 2.5 Flash with a grounded prompt.
4. The answer streams back with citations; the turn is saved to SQLite.

**Check quality:** `python scripts/eval.py` → 5/5.

---

## Known limits
- **Free-tier quotas:** `gemini-2.5-flash` ≈ 20 generations/day; embeddings ≈ 100/min
  (ingestion is paced for this). Switching `CHAT_MODEL` to `gemini-2.0-flash` gives more
  daily headroom.
- **Route maps:** extracted from dense infographics, so route data can have occasional
  gaps. Core fleet/regulation answers are reliable.
