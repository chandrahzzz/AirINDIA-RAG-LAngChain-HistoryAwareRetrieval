"""Quick check that your Gemini API key works (new google-genai SDK).

Run:  python scripts/test_key.py

Makes one embedding call and one chat call. If both succeed, the key is good.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai

import config


def main() -> int:
    key = config.GOOGLE_API_KEY
    if not key:
        print("FAIL: GOOGLE_API_KEY is empty in .env")
        return 1

    client = genai.Client(api_key=key)
    print(f"Using key: {key[:6]}... (len={len(key)})")

    # 1) Embedding model
    try:
        r = client.models.embed_content(model=config.EMBED_MODEL, contents="hello world")
        dim = len(r.embeddings[0].values)
        print(f"OK  embeddings ({config.EMBED_MODEL}): vector length = {dim}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL embeddings: {type(e).__name__}: {e}")
        return 1

    # 2) Chat model
    try:
        resp = client.models.generate_content(
            model=config.CHAT_MODEL, contents="Reply with exactly: OK"
        )
        print(f"OK  chat ({config.CHAT_MODEL}): {resp.text.strip()!r}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL chat: {type(e).__name__}: {e}")
        return 1

    print("\nAll good — key works. Ready to build the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
