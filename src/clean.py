"""Text cleaning for extracted PDF content.

Each source has its own noise:
- Britannica scrape: site navigation / "Ask the Chatbot" boilerplate.
- Fact Sheet & Britannica: private-use bullet glyphs (\\ue000-\\uf8ff).
- All: ragged whitespace, hyphenated line breaks.
"""
from __future__ import annotations

import re

# Lines that are pure website navigation noise in the Britannica PDF.
_BRITANNICA_NOISE = re.compile(
    r"(Games & Quizzes|Ask the Chatbot|More Actions|Subscribe|Newsletters?|"
    r"Table of Contents|Print|Cite|Share|Feedback|Login|External Websites|"
    r"Click here to|Encyclopaedia Britannica|©.*Britannica)",
    re.IGNORECASE,
)


def _strip_private_use(text: str) -> str:
    # Replace private-use-area glyphs (broken bullets/icons) with a space.
    return re.sub(r"[-]", " ", text)


def _fix_hyphenation(text: str) -> str:
    # Join words split across line breaks: "appoint-\nment" -> "appointment".
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean_generic(text: str) -> str:
    text = _strip_private_use(text)
    text = _fix_hyphenation(text)
    return _normalize_ws(text)


def clean_britannica(text: str) -> str:
    text = _strip_private_use(text)
    text = _fix_hyphenation(text)
    kept = [ln for ln in text.split("\n") if not _BRITANNICA_NOISE.search(ln)]
    return _normalize_ws("\n".join(kept))
