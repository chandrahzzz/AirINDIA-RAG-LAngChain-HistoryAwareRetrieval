"""LangChain (LCEL) conversational RAG chain over the Air India corpus.

Reuses the artifacts we already built:
  - Chroma vector store (Gemini embeddings) from `python -m src.ingest`
  - BM25 chunk texts/metadata from data/bm25.pkl

Composes the canonical LangChain RAG stack on top:
  EnsembleRetriever (vector + BM25)                -> hybrid retrieval
  ContextualCompressionRetriever + CrossEncoder    -> reranking
  create_history_aware_retriever                   -> follow-up query rewriting
  create_retrieval_chain (+ stuff documents)       -> grounded answer
  RunnableWithMessageHistory + SQLChatMessageHistory -> conversational memory
"""
from __future__ import annotations

import os
import pickle
import re

import chromadb
# LangChain 1.x moved the legacy RAG helper chains into `langchain_classic`.
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from src import embeddings as gem

# --- Prompts ---------------------------------------------------------------
CONTEXTUALIZE_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question "
    "as a standalone question understandable without the history. Do NOT answer "
    "it; return it unchanged if already standalone."
)

QA_SYSTEM = (
    "You are the Air India information assistant. Answer ONLY using the context "
    "below. If the answer is not in the context, say you don't have that "
    "information in your documents — never invent facts, numbers, routes, or "
    "dates.\n"
    "Each context passage begins with its source in square brackets, e.g. "
    "[Air India Service Regulations, CHAPTER IV - RETIREMENT, p15]. When you state "
    "a fact, cite it by copying that bracketed label EXACTLY — never invent or alter "
    "a citation.\n"
    "Give a complete answer: include any relevant conditions, exceptions, or provisos "
    "stated in the context (e.g. notice periods, approvals, or cases where it does "
    "not apply).\n\n"
    "Context:\n{context}"
)

# Each retrieved chunk is rendered to the LLM with its REAL source label up front,
# so citations are grounded in metadata instead of guessed from the text (Fix B).
DOC_PROMPT = PromptTemplate.from_template("[{cite}]\n{page_content}")


# --- Embeddings adapter: reuse our rate-limited, task-type-aware Gemini embed
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return gem.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return gem.embed_query(text)


# --- Windowed persistent memory -------------------------------------------
class WindowedSQLHistory(SQLChatMessageHistory):
    """SQL-backed history that only exposes the last N turns to the prompt."""

    def __init__(self, session_id: str, window_turns: int):
        super().__init__(session_id=session_id, connection=f"sqlite:///{config.HISTORY_DB}")
        self._window = window_turns

    @property
    def messages(self):  # type: ignore[override]
        return super().messages[-2 * self._window:]


def _vector_retriever(k: int):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    store = Chroma(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding_function=GeminiEmbeddings(),
    )
    return store.as_retriever(search_kwargs={"k": k})


def _bm25_retriever(k: int):
    with open(config.BM25_PATH, "rb") as f:
        data = pickle.load(f)
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data["texts"], data["metas"])
    ]
    r = BM25Retriever.from_documents(docs)
    r.k = k
    return r


def _hybrid_retriever():
    vec = _vector_retriever(config.RETRIEVE_K)
    bm25 = _bm25_retriever(config.RETRIEVE_K)
    ensemble = EnsembleRetriever(retrievers=[vec, bm25], weights=[0.6, 0.4])

    # Reranking layer (optional — degrade gracefully if unavailable).
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 4)  # use all CPU cores for reranking
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL),
            top_n=config.RERANK_TOP_N,
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble
        )
    except Exception as e:  # noqa: BLE001
        print(f"[lc_chain] reranker unavailable ({type(e).__name__}); using ensemble only.")
        return ensemble


# --- Fix 1a: route/aggregation queries need EVERY route, not just top-k --------
_ROUTE_WORDS = (
    "route", "routes", "flight", "flights", "fly", "flies", "destination",
    "destinations", "connect", "connectivity", "non-stop", "nonstop", "sector",
)
_routes_doc_cache: Document | None = None


def _is_route_query(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _ROUTE_WORDS)


def _full_routes_doc() -> Document | None:
    """The entire (small) route list as one document, so 'list/count all flights'
    queries see every route instead of a top-k subset."""
    global _routes_doc_cache
    if _routes_doc_cache is None:
        path = config.DATA_DIR / "routes_extracted.txt"
        if not path.exists():
            return None
        _routes_doc_cache = Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"cite": "Air India Route Map, Feb 2025", "doc_type": "routes",
                      "source": "Route Maps Feb 2025", "page": 0, "section": "route network"},
        )
    return _routes_doc_cache


def _augmented_retriever():
    """Hybrid retriever, but for route-style queries prepend the COMPLETE route list
    (and drop the partial route chunks) so answers are complete (Fix 1a)."""
    base = _hybrid_retriever()

    def _retrieve(query: str) -> list[Document]:
        docs = base.invoke(query)
        if _is_route_query(query):
            full = _full_routes_doc()
            if full is not None:
                non_route = [d for d in docs if d.metadata.get("doc_type") != "routes"]
                return [full] + non_route
        return docs

    return RunnableLambda(_retrieve)


# --- Smalltalk gate: greetings don't need retrieval, so answer instantly --------
_GREETING = re.compile(r"^(hi+|hey+|hello+|hiya|yo|howdy|greetings|good\s*(morning|afternoon|evening))\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|thx|ty|appreciate it)\b", re.I)
_BYE = re.compile(r"^(bye|goodbye|see you|cya|good night)\b", re.I)


def smalltalk_reply(message: str) -> str | None:
    """Instant canned reply for greetings/thanks/bye so they skip the ~10s RAG
    pipeline entirely. Returns None for anything that needs a real answer."""
    m = message.strip()
    if len(m) > 40:                      # long messages are real questions
        return None
    if _GREETING.search(m):
        return ("Hello! I'm the Air India assistant. Ask me about the fleet, routes, "
                "service regulations, or history.")
    if _BYE.search(m):
        return "Goodbye! Come back anytime with questions about Air India."
    if _THANKS.search(m) and "?" not in m:
        return "You're welcome! Anything else about Air India you'd like to know?"
    return None


def build_chain():
    config.require_key()
    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL, temperature=0.2,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=3,      # auto-retry transient 5xx with backoff (Fix 2)
        timeout=60,         # don't hang forever on a stuck request
        thinking_budget=0,  # disable internal "thinking" — big latency cut, and the
                            # answers are grounded extraction so it doesn't help quality
    )
    retriever = _augmented_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=DOC_PROMPT)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda sid: WindowedSQLHistory(sid, config.MEMORY_WINDOW_TURNS),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
