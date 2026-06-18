"""Turn the two route-MAP PDFs into structured, queryable data.

The Domestic/International route PDFs are infographics: plain text extraction
yields a bare list of city names with no connection information. We instead send
each PDF to Gemini (which reads PDFs natively) and ask for structured JSON, then
render that JSON into clean prose chunks the retriever can actually use.

Run:  python -m src.maps_extract
Outputs: data/routes_domestic.json, data/routes_international.json
         data/routes_extracted.txt   (human-readable, also fed into the index)
"""
from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types

import config

# Map PDFs and how to interpret them.
MAP_PDFS = {
    "domestic": {
        "file": config.PDF_DIR / "Domestic Routes Feb 2025 (1).pdf",
        "scope": "domestic routes within India",
    },
    "international": {
        "file": config.PDF_DIR / "International Routes Feb 2025.pdf",
        "scope": "international routes between India and other countries",
    },
}

PROMPT = """You are reading an Air India ROUTE MAP infographic (a PDF).
Extract EVERY piece of factual information you can read from it. Specifically:

1. Every city/airport shown as an Air India destination.
2. Every route line you can see connecting two cities (origin -> destination).
3. Any text annotations (e.g. "non-stop to 42 destinations", new/upcoming routes,
   effective dates, hubs).

Scope of this map: {scope}.

Return ONLY JSON matching this shape:
{{
  "scope": "domestic" | "international",
  "destinations": ["City1", "City2", ...],
  "routes": [{{"from": "CityA", "to": "CityB"}}, ...],
  "notes": ["free-text fact 1", "free-text fact 2", ...]
}}

Read carefully and do not invent connections you cannot see. If you can see a city
is on the map but cannot determine its connections, still list it under destinations.
"""


def extract_one(client: genai.Client, scope_key: str, spec: dict) -> dict:
    pdf_bytes = Path(spec["file"]).read_bytes()
    resp = client.models.generate_content(
        model=config.VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            PROMPT.format(scope=spec["scope"]),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    data = json.loads(resp.text)
    data.setdefault("scope", scope_key)
    return data


def to_prose(scope_key: str, data: dict) -> str:
    """Render extracted JSON into clean sentences for the retriever/index."""
    label = "domestic (within India)" if scope_key == "domestic" else "international"
    lines = [f"Air India {label} route network (source: {scope_key} route map, Feb 2025)."]

    dests = data.get("destinations", [])
    if dests:
        lines.append(f"Air India serves these {len(dests)} {label} destinations: "
                     + ", ".join(dests) + ".")

    routes = data.get("routes", [])
    if routes:
        lines.append("Direct routes shown on the map:")
        for r in routes:
            frm, to = r.get("from"), r.get("to")
            if frm and to:
                lines.append(f"- {frm} to {to}.")

    for note in data.get("notes", []):
        lines.append(note if note.endswith(".") else note + ".")

    return "\n".join(lines)


def main() -> None:
    config.require_key()
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    prose_blocks = []
    for scope_key, spec in MAP_PDFS.items():
        if not Path(spec["file"]).exists():
            print(f"[skip] missing {spec['file']}")
            continue
        print(f"[vision] extracting {scope_key} routes from {Path(spec['file']).name} ...")
        data = extract_one(client, scope_key, spec)
        out_json = config.DATA_DIR / f"routes_{scope_key}.json"
        out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"         destinations={len(data.get('destinations', []))} "
              f"routes={len(data.get('routes', []))} notes={len(data.get('notes', []))} "
              f"-> {out_json.name}")
        prose_blocks.append(to_prose(scope_key, data))

    combined = "\n\n".join(prose_blocks)
    (config.DATA_DIR / "routes_extracted.txt").write_text(combined, encoding="utf-8")
    print(f"[done] wrote {config.DATA_DIR / 'routes_extracted.txt'}")


if __name__ == "__main__":
    main()
