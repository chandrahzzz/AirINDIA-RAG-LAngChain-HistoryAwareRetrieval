# Architecture

A **RAG (Retrieval-Augmented Generation)** chatbot: it answers from *retrieved Air India
documents*, not the model's memory — so answers are grounded and cited, not hallucinated.

Two phases: **build the index once**, then **answer each question**.

---

## Phase 1 — Build the index (run once)
```
text PDFs ─► pypdf ─► clean ─► chunk (+chapter tags & citations) ─┐
route-map PDFs ─► Gemini Vision ─► structured routes ─────────────┤
                                                                  ▼
                       Gemini embeddings (3072-dim) ─► Chroma (vectors) + BM25 (keywords)
```
- Route maps are images, so **Gemini Vision** extracts routes as data (the key accuracy win).
- Each chunk carries metadata (source, chapter, page) for correct citations.
- Output: 239 chunks stored two ways. The live app reads this index, **not the PDFs**.

## Phase 2 — Answer a question (every query)
```
question + history
   ─► history-aware rewrite (standalone question)
   ─► embed query ─► hybrid retrieve (Chroma + BM25) ─► fuse
   ─► rerank (cross-encoder) → best 8 chunks
   ─► grounded prompt (chunks + citations) ─► Gemini 2.5 Flash
   ─► stream answer + citations ─► save turn to SQLite memory
```
(Greetings are detected up front and answered instantly, skipping retrieval.)

---

## Tech stack
| Layer | Tech | Role |
|---|---|---|
| LLM + Vision | Gemini 2.5 Flash | reads route maps; writes grounded answers |
| Embeddings | `gemini-embedding-001` | text → 3072-dim meaning vectors |
| Vector DB | Chroma | semantic search |
| Keyword search | BM25 (`rank_bm25`) | exact-term matching |
| Reranker | `ms-marco-MiniLM-L-6-v2` | pick the most relevant chunks |
| Orchestration | LangChain 1.x | retrieve → rewrite → answer → memory |
| Memory | SQLite | per-session chat history |
| Web | FastAPI + Uvicorn + HTML/JS | streaming `/chat` API + chat UI |
| Packaging / Host | Docker → AWS EC2 | prod == local; public deployment |

## Key techniques
- **RAG** — answer from retrieved context, with citations (no hallucination).
- **Multimodal extraction** — Gemini Vision turns map images into route data.
- **Hybrid retrieval** — semantic (vector) + keyword (BM25) together.
- **Two-stage retrieval** — broad recall, then a reranker picks the best few.
- **History-aware rewriting** — makes follow-up questions work.
- **Structure-aware chunking** — keeps chapter context for accurate citations.
- **Conversational memory** — remembers the dialogue per session.

> One-liner: *Hybrid-retrieval RAG over Air India PDFs — Gemini (vision/embeddings/generation),
> Chroma + BM25, a cross-encoder reranker, LangChain orchestration with memory, served via
> FastAPI and deployed on AWS with Docker.*
