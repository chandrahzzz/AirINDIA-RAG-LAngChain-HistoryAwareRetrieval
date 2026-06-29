"""Agentic chat graph (LangGraph).

Per user turn, a router sends the message to one of two nodes:
  - rag      : answer an Air India question (reuses the hybrid-retrieval RAG chain)
  - capture  : collect an interested lead (name, contact, routes) over turns, then
               save it to the interested list as PENDING (an admin approves later = HITL)

State (messages + lead fields) is persisted per session_id via a SQLite checkpointer,
so the conversation and a half-finished lead survive across turns and restarts.
"""
from __future__ import annotations

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

import config
from src import lc_chain, leads

# Phrases that mean "put me on the interested list".
INTEREST_WORDS = (
    "interested", "sign me up", "add me", "notify me", "keep me posted", "register",
    "contact me", "reach out", "join the", "put me on", "interested list", "i want to be",
)
CANCEL_WORDS = ("cancel", "nevermind", "never mind", "stop", "forget it")

ASK = {
    "name": "I'd be happy to add you to our interested list! May I have your name?",
    "contact": "Thanks! What's the best email or phone number to reach you?",
    "routes": "Which routes or destinations are you interested in?",
}

EXTRACT_PROMPT = (
    "You are collecting a lead for Air India's 'interested' list. From the conversation, "
    "extract the person's name, their contact (email or phone), and the routes/destinations "
    "they're interested in. Return null for anything not yet clearly provided. Do not invent.\n\n"
    "Already known (keep these): {known}\n\nConversation:\n{convo}"
)


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    name: str | None
    contact: str | None
    routes: str | None
    capturing: bool


class LeadFields(BaseModel):
    name: str | None = None
    contact: str | None = None
    routes: str | None = None


# Built once at startup.
_rag = None
_llm = None


def _ensure_built():
    global _rag, _llm
    if _rag is None:
        _llm = lc_chain.build_llm()
        _rag = lc_chain.build_rag_chain(_llm)


def _route(state: ChatState) -> str:
    if state.get("capturing"):
        return "capture"
    last = state["messages"][-1].content.lower()
    if any(w in last for w in INTEREST_WORDS):
        return "capture"
    return "rag"


def _rag_node(state: ChatState) -> dict:
    msgs = state["messages"]
    question = msgs[-1].content
    history = msgs[:-1][-12:]  # windowed memory: last few turns, excluding current
    result = _rag.invoke({"input": question, "chat_history": history})
    return {"messages": [AIMessage(result["answer"])]}


def _capture_node(state: ChatState, config) -> dict:  # noqa: A002 — LangGraph injects by name
    last = state["messages"][-1].content.lower()
    if any(w in last for w in CANCEL_WORDS):
        return {"capturing": False,
                "messages": [AIMessage("No problem — let me know if you change your mind. "
                                       "Meanwhile, ask me anything about Air India.")]}

    convo = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-8:])
    known = {k: state.get(k) for k in ("name", "contact", "routes")}
    got = _llm.with_structured_output(LeadFields).invoke(
        EXTRACT_PROMPT.format(known=known, convo=convo)
    )

    name = state.get("name") or got.name
    contact = state.get("contact") or got.contact
    routes = state.get("routes") or got.routes

    for field, value in (("name", name), ("contact", contact), ("routes", routes)):
        if not value:
            return {"name": name, "contact": contact, "routes": routes,
                    "capturing": True, "messages": [AIMessage(ASK[field])]}

    # All fields present -> save as PENDING for admin approval (HITL).
    sid = (config or {}).get("configurable", {}).get("thread_id", "")
    leads.add_pending(name, contact, routes, session_id=sid)
    reply = (f"Thank you, {name}! I've noted your interest in {routes} and will pass your "
             f"details (contact: {contact}) to our team. You'll be added to the interested "
             f"list once they confirm.")
    return {"name": name, "contact": contact, "routes": routes,
            "capturing": False, "messages": [AIMessage(reply)]}


def build_graph():
    _ensure_built()
    conn = sqlite3.connect(str(config.GRAPH_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    g = StateGraph(ChatState)
    g.add_node("rag", _rag_node)
    g.add_node("capture", _capture_node)
    g.add_conditional_edges(START, _route, {"rag": "rag", "capture": "capture"})
    g.add_edge("rag", END)
    g.add_edge("capture", END)
    return g.compile(checkpointer=saver)
