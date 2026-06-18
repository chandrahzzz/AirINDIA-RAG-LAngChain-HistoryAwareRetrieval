"""LangChain conversational RAG — command-line chat loop.

Run:  python -m src.lc_cli
Commands:  /new  start a fresh session    /exit
"""
from __future__ import annotations

import sys
import uuid

import config
from src import lc_chain

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    config.require_key()
    print("Building LangChain RAG chain (loads reranker on first run) ...")
    chain = lc_chain.build_chain()
    session_id = f"lc-{uuid.uuid4().hex[:8]}"
    print("\nAir India assistant (LangChain). Ask about fleet, routes, regulations, history.")
    print("Commands: /new  /exit\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/new":
            session_id = f"lc-{uuid.uuid4().hex[:8]}"
            print("[new session]\n")
            continue

        print("Bot: ", end="", flush=True)
        cfg = {"configurable": {"session_id": session_id}}
        for chunk in chain.stream({"input": q}, config=cfg):
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
