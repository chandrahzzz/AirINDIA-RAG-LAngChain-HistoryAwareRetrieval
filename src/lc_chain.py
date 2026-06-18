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

import pickle

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    "dates. Be concise and cite sources inline like [Air India Fact Sheet p3].\n\n"
    "Context:\n{context}"
)


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


def build_chain():
    config.require_key()
    llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL, temperature=0.2,
        google_api_key=config.GOOGLE_API_KEY,
    )
    retriever = _hybrid_retriever()

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
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda sid: WindowedSQLHistory(sid, config.MEMORY_WINDOW_TURNS),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
