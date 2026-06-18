# Air India RAG Chatbot

A retrieval-augmented chatbot over five Air India PDFs (service regulations, fact
sheet, route maps, disaster history). Built for **accuracy** and **low latency** on
a **free** Gemini API tier.

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

## Models (verified for this key)
- Embeddings: `gemini-embedding-001` (3072-dim, task-type aware)
- Chat + Vision: `gemini-2.5-flash`
- Reranker: `BAAI/bge-reranker-base` (local, optional — degrades gracefully)

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
```bash
python main.py                      # the chatbot (LangChain RAG: ensemble + rerank, SQL memory)
```

## Evaluate (regression guard)
```bash
python scripts/eval.py              # runs the golden-question set
```

## Layout
```
main.py              entry point -> runs the chatbot
config.py            paths, model names, knobs
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

## Possible next steps
- Re-OCR the regulations PDF (it has OCR typos) for even cleaner retrieval.
- FastAPI streaming endpoint + a small web UI.
- Semantic cache for repeated questions.
- Migrate Chroma → pgvector/Pinecone only if the corpus grows large.
