# Architecture

An **agentic RAG chatbot**: it answers from *retrieved Air India documents* (grounded +
cited, not hallucinated), **and** can capture an "interested" lead — which an admin must
approve (human-in-the-loop). A **LangGraph** router decides, per message, whether to
answer a question or collect a lead.

Two phases: **build the index once**, then **handle each message**.

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

## Phase 2 — Handle a message (LangGraph agent, every turn)
```
message ─► [router] ─┬─► RAG node ─────────────────────────────────────┐
                     │     history-aware rewrite → hybrid retrieve       │
                     │     (Chroma + BM25) → rerank (cross-encoder, 8)   │
                     │     → grounded prompt → Gemini 2.5 Flash          │
                     │                                                   ▼
                     └─► Lead-capture node ──► collect name/contact/   answer + citations
                            routes over turns ──► save as PENDING
                                                        │
                              admin reviews /admin ──► APPROVE/REJECT  (human-in-the-loop)
                                                        │
                                                   interested list
```
- State (conversation + half-filled lead) is persisted per `session_id` via a **LangGraph
  SQLite checkpointer** — that's the conversation **memory**.
- Greetings are detected up front and answered instantly, skipping the graph.

---

## Tech stack
| Layer | Tech | Role |
|---|---|---|
| LLM + Vision | Gemini 2.5 Flash | reads route maps; writes grounded answers |
| Embeddings | `gemini-embedding-001` | text → 3072-dim meaning vectors |
| Vector DB | Chroma | semantic search |
| Keyword search | BM25 (`rank_bm25`) | exact-term matching |
| Reranker | `ms-marco-MiniLM-L-6-v2` | pick the most relevant chunks |
| Orchestration | LangChain 1.x | the RAG chain (retrieve → rewrite → grounded answer) |
| Agent / routing | **LangGraph** | router + RAG node + lead-capture node + HITL |
| Memory | SQLite (LangGraph checkpointer) | per-session conversation + lead state |
| Leads | SQLite + admin page | interested list with admin approval |
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
- **Agentic routing (LangGraph)** — picks RAG vs. lead-capture per message.
- **Human-in-the-loop (HITL)** — captured leads need admin approval before they're official.

> One-liner: *Agentic hybrid-retrieval RAG over Air India PDFs — Gemini
> (vision/embeddings/generation), Chroma + BM25, a cross-encoder reranker, LangChain +
> LangGraph orchestration with conversational memory and HITL lead capture, served via
> FastAPI and deployed on AWS with Docker.*
