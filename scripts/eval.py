"""Golden-set evaluator for the LangChain RAG chain.

Run:  python scripts/eval.py

For each question it checks two things:
  1. retrieval surfaced the expected source document, and
  2. the generated answer contains the expected fact.

This is the regression guard — run it after any change to chunking, retrieval,
prompts, or the chain. Expand GOLDEN as you find weak spots.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import lc_chain

# (question, substring expected in answer, expected source substring)
GOLDEN = [
    ("How many A350-900 aircraft are in Air India's fleet?", "6", "Fact Sheet"),
    ("How many total aircraft does Air India operate?", "138", "Fact Sheet"),
    ("Does Air India fly from Delhi to Frankfurt?", "Frankfurt", "Route"),
    ("When do flights to Tel Aviv restart?", "March", "Route"),
    ("What does the regulation say about probation and appointment?",
     "appoint", "regulation"),
]


def main() -> int:
    chain = lc_chain.build_chain()
    passed = 0
    for q, expect_ans, expect_src in GOLDEN:
        # Fresh session per question so history doesn't leak between tests.
        cfg = {"configurable": {"session_id": f"eval-{uuid.uuid4().hex[:8]}"}}
        result = chain.invoke({"input": q}, config=cfg)

        answer = (result.get("answer") or "").lower()
        sources = [d.metadata.get("source", "") for d in result.get("context", [])]

        src_ok = any(expect_src.lower() in s.lower() for s in sources)
        ans_ok = expect_ans.lower() in answer
        ok = src_ok and ans_ok
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        print(f"        source_hit={src_ok}  answer_has({expect_ans!r})={ans_ok}")
        if not ok:
            print(f"        retrieved sources: {sources}")

    print(f"\n{passed}/{len(GOLDEN)} passed")
    return 0 if passed == len(GOLDEN) else 1


if __name__ == "__main__":
    raise SystemExit(main())
