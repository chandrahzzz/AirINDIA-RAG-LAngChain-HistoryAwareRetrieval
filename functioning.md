# How My Air India RAG Chatbot Works

This is my write-up of what I built — the key features first, then how every file
works, how each piece of the tech stack fits in, and the complete end-to-end flow.

---

## ⭐ Key insights & features (the highlights)

- **It's a RAG chatbot, not a "stuff-the-PDF-in-the-prompt" bot.** My 5 PDFs are
  ~120 pages / 6.5 MB — too big to feed the model every time. So I index everything
  once and only retrieve the few most relevant chunks per question.
- **I solved the hardest part: the route maps.** Two of my PDFs are *map infographics*
  — plain text extraction gives a useless list of city names with no connections. I
  used **Gemini Vision** to read the maps and extract ~310 routes as structured data.
  This is the single biggest accuracy win and what a normal pypdf tutorial bot gets wrong.
- **Hybrid retrieval + reranking.** I combine **semantic (vector)** search with
  **keyword (BM25)** search, then a **cross-encoder reranker** picks the best chunks.
  Meaning-matches *and* exact-term matches (like "A321neo" or a clause number).
- **It remembers the conversation.** Follow-ups like "and how many are on order?" work,
  because a **history-aware rewrite** turns them into standalone questions before
  searching. History is stored in **SQLite**, so it survives restarts.
- **It doesn't hallucinate.** A grounded prompt forces the model to answer *only* from
  retrieved context and to **cite sources** inline, e.g. `[Air India Fact Sheet p3]`.
- **Built on LangChain (1.x)** — the recognizable RAG stack (`EnsembleRetriever`,
  `CrossEncoderReranker`, `create_history_aware_retriever`, `RunnableWithMessageHistory`).
- **Runs on the Gemini free tier**, in an isolated **virtual environment**, with a
  **golden-set eval** that proves accuracy (currently 5/5).
- **One command to run it:** `python main.py`.

---

## 1. What this project is (the short version)

A **RAG (Retrieval-Augmented Generation) chatbot** that answers questions about
Air India using 5 PDFs:

- `Aiesl Employees service regulation.pdf` (94 pages of HR/service rules)
- `Air India Fact Sheet.pdf` (fleet numbers, orders, facts)
- `Domestic Routes Feb 2025 (1).pdf` (a **map infographic**)
- `International Routes Feb 2025.pdf` (another **map infographic**)
- `List of Major Air India Disasters … Britannica.pdf` (a web article)

The chatbot is built with **LangChain** (`src/lc_chain.py`) and launched from `main.py`.

---

## 2. The tech stack and how I implemented each piece

| Tech | What it does in my project | Where |
|---|---|---|
| **Google Gemini (`google-genai` SDK)** | The LLM. `gemini-2.5-flash` generates answers + reads the route-map PDFs (vision). `gemini-embedding-001` turns text into 3072-dim vectors. | `src/embeddings.py`, `src/maps_extract.py` |
| **Gemini Vision** | Reads the two route-map infographics (basically pictures) and extracts destinations + routes + notes as structured JSON. My biggest accuracy trick. | `src/maps_extract.py` |
| **Chroma** | Local, persisted **vector database**. Stores the embeddings for semantic similarity search. No cloud DB needed. | `src/ingest.py`, `src/lc_chain.py` |
| **BM25 (`rank_bm25`)** | Keyword search index. Catches exact tokens like "A321neo" or clause numbers that vector search can miss. | `src/ingest.py`, `src/lc_chain.py` |
| **Hybrid retrieval** | Combines vector + BM25 via LangChain's `EnsembleRetriever` (weighted 0.6 / 0.4) so I get both *meaning* and *exact-word* matches. | `src/lc_chain.py` |
| **Cross-encoder reranker (`bge-reranker-base`)** | After hybrid search grabs ~20 candidates, this re-scores them and keeps the best 5. Runs locally, free. Degrades gracefully if unavailable. | `src/lc_chain.py` |
| **LangChain (1.x) + LangChain-Classic** | Orchestrates the whole RAG chain: retriever → history-aware rewrite → grounded answer → memory. | `src/lc_chain.py` |
| **SQLite** | Stores the conversation history so the bot remembers context and survives restarts. | `src/lc_chain.py` |
| **pypdf** | Pulls the raw text out of the text-based PDFs (regulations, fact sheet, article). | `src/loaders.py` |
| **python-dotenv** | Loads my `GOOGLE_API_KEY` from `.env` so the key never sits in the code. | `config.py` |
| **venv (virtual environment)** | Keeps this project's packages isolated from my other Python projects (they had conflicting LangChain/FastAPI/Pinecone versions). | `.venv/` |

---

## 3. How each file works

### Entry point, config & secrets
- **`main.py`** — The entry point. `python main.py` launches the chatbot.
- **`config.py`** — Central settings: all the paths (PDFs, `chroma_db/`, `data/bm25.pkl`,
  the SQLite history DB), the **model names** (`gemini-embedding-001`, `gemini-2.5-flash`,
  `bge-reranker-base`), and the **knobs** (chunk size 1000, overlap 150, retrieve 20
  candidates, rerank down to top 5, memory window of 6 turns). Loads the API key from
  `.env` and exposes `require_key()` which fails loudly if it's missing.
- **`.env`** — My `GOOGLE_API_KEY`. Git-ignored so it never gets committed.
- **`.env.example`** — Safe template showing what `.env` should look like.
- **`.gitignore`** — Ignores `.env`, the venv, `__pycache__`, and the generated
  `chroma_db/` / index files.
- **`requirements.txt`** — All dependencies, pinned (LangChain pinned to 1.x because
  1.x moved the legacy RAG helpers into `langchain-classic`).

### Ingestion pipeline (build the index — run once)
- **`src/maps_extract.py`** — Sends each **route-map PDF** to **Gemini Vision** with a
  prompt that asks for every destination + every route + any notes, returned as **JSON**.
  Saves `data/routes_domestic.json` / `routes_international.json`, then renders them into
  clean sentences in `data/routes_extracted.txt` so the retriever can use them.
  *(This recovered ~310 routes that plain text extraction completely lost — including
  facts like the Tel Aviv restart date.)*
- **`src/clean.py`** — Text cleaning. Strips private-use bullet glyphs, fixes hyphenated
  line breaks, normalizes whitespace, and removes Britannica website navigation junk
  ("Ask the Chatbot", "Games & Quizzes", etc.).
- **`src/loaders.py`** — Loads each PDF with a **strategy per document type**:
  - *Regulation:* chunk per page but **tag each chunk with its CHAPTER heading** so a
    retrieved clause keeps its context (that's why citations show `[CHAPTER III …]`).
  - *Fact sheet / article:* clean + paragraph-pack into ~1000-char chunks.
  - *Routes:* load the Vision-extracted prose.
  - It also **atomizes** oversized blocks (the route list was one giant block) so no chunk
    is way bigger than the target size. Every chunk carries metadata
    `{source, doc_type, page, section}` for citations and filtering.
- **`src/embeddings.py`** — Wraps Gemini embeddings. Uses **task types**
  (`RETRIEVAL_DOCUMENT` when indexing, `RETRIEVAL_QUERY` when searching) for better
  matching. Has **rate limiting + 429 backoff** because the free tier only allows ~100
  embed requests/min — it paces itself and retries when Google says to wait.
- **`src/ingest.py`** — The build step. Loads all chunks → embeds them → stores vectors
  in **Chroma** (`chroma_db/`) → builds the **BM25** keyword index → pickles it to
  `data/bm25.pkl`. Re-run anytime the PDFs change. *(I built 239 chunks / 239 vectors.)*

### The RAG chain (LangChain)
- **`src/lc_chain.py`** — The whole question-answering flow, built with **LangChain**:
  - Wraps my rate-limited Gemini embedder as a LangChain `Embeddings` (so it reuses the
    index I already built, no deprecated SDK).
  - `EnsembleRetriever` = Chroma vector retriever + `BM25Retriever` → **hybrid retrieval**.
  - `ContextualCompressionRetriever` + `CrossEncoderReranker` → **reranking** (top ~20 →
    best 5). Falls back to the ensemble order if the reranker model isn't available.
  - `create_history_aware_retriever` → **follow-up question rewriting** (turns "and how
    many on order?" into a full standalone question *before* retrieving).
  - `create_retrieval_chain` + `create_stuff_documents_chain` → retrieve then answer with
    a **grounded prompt** + inline citations.
  - `RunnableWithMessageHistory` + a windowed `SQLChatMessageHistory` (my
    `WindowedSQLHistory`) → **conversation memory**: last 6 turns, per `session_id`,
    survives restarts.
  - LLM is `ChatGoogleGenerativeAI` (Gemini 2.5 Flash), streamed.
- **`src/lc_cli.py`** — The command-line chat loop `main.py` calls. Streams tokens,
  supports `/new` (fresh session) and `/exit`.

### Scripts & docs
- **`scripts/test_key.py`** — Quick check that my Gemini key works (one embed call + one
  chat call). I run this first whenever I change the key.
- **`scripts/eval.py`** — My **golden-set evaluator**: 5 known Q&A pairs across fleet /
  routes / regulations. For each, it checks both "did it retrieve the right source?" and
  "does the answer contain the expected fact?". Runs through the LangChain chain. This is
  my **regression guard** — I run it after any change. *(Currently 5/5 passing.)*
- **`README.md`** — Setup + how to run.
- **`functioning.md`** — This file.

---

## 4. The complete end-to-end flow

**A) One-time setup**
1. Make a venv, `pip install -r requirements.txt`, put my key in `.env`.
2. `python scripts/test_key.py` → confirms Gemini works.

**B) Build the index (run once, re-run if PDFs change)**
3. `python -m src.maps_extract` → Gemini Vision reads the two map PDFs → structured routes.
4. `python -m src.ingest` → load + clean + chunk all PDFs → embed (rate-limited) → store
   in Chroma + build BM25. *(239 chunks indexed.)*

**C) Chatting (every question) — `python main.py`**
5. I type a question.
6. **History-aware rewrite** turns my follow-up into a standalone question using recent
   conversation (only if there's history).
7. **Hybrid retrieval** (vector + BM25) pulls ~20 candidate chunks.
8. **Reranker** scores them and keeps the best 5.
9. Those 5 chunks + my question go to **Gemini 2.5 Flash** with a **grounded prompt**
   ("answer only from this context, cite sources, don't make things up").
10. The answer **streams** back with inline citations like `[Air India Fact Sheet p3]`.
11. The turn is saved to **SQLite memory** so the next question has context.

**D) Checking quality**
12. `python scripts/eval.py` → runs my 5 golden questions → 5/5 passing.

---

## 5. The key decisions I made (and why)

- **RAG instead of stuffing the PDF in the prompt** — the corpus is too big and the maps
  are unreadable as plain text.
- **Gemini Vision on the route maps** — biggest accuracy win; a normal pypdf bot would get
  route questions wrong.
- **Hybrid retrieval + reranking** — semantic search alone misses exact terms; keyword
  alone misses meaning; reranking sharpens the final 5 chunks.
- **Structure-aware chunking** — regulation chunks keep their CHAPTER heading so answers
  cite the right section.
- **Grounded prompt + citations** — stops hallucination, which matters for a
  factual/regulatory bot.
- **Rate-limited embedder** — the Gemini free tier caps embeddings at ~100/min.
- **A virtual environment** — my global Python had conflicting packages; the venv keeps
  this project clean and reproducible.
- **LangChain for the orchestration** — gives me the standard, recognizable RAG components
  (`EnsembleRetriever`, reranker, history-aware retriever, `RunnableWithMessageHistory`)
  instead of hand-wiring them.

---

## 6. Good to know (limits)
- **Free-tier daily cap:** `gemini-2.5-flash` allows ~20 generations/day on the free tier
  (resets daily). Switching `CHAT_MODEL` to `gemini-2.0-flash` in `config.py` gives more
  daily headroom if I do heavy testing.
- **Embeddings:** ~100/min free-tier cap, which is why ingestion is paced.

## 7. What's not done yet (future ideas)
- FastAPI streaming endpoint + a web UI (Streamlit/React).
- Re-OCR the regulations PDF (it has scanner typos).
- A semantic cache for repeated questions.
- Move Chroma → pgvector/Pinecone only if the data grows a lot.
