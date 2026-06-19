"""Load each PDF with a strategy suited to its content, then chunk.

Produces a list of `dict` chunks: {id, text, source, doc_type, page, section}.
Metadata travels with every chunk so the chatbot can cite sources and we can
filter retrieval by document type.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

import config
from src import clean

# --- Source registry: filename -> how to treat it -------------------------
PROSE = "prose"
REGULATION = "regulation"
ARTICLE = "article"
ROUTES = "routes"

SOURCES = {
    "Aiesl Employees service regulation.pdf": REGULATION,
    "Air India Fact Sheet.pdf": PROSE,
    "List of Major Air India Disasters _ Crashes, Death Toll, Tata Group, History, & Accidents _ Britannica.pdf": ARTICLE,
}

# Heading detector. OCR-tolerant: the leading letter is often mangled
# ("GHAPTER IV", "OHAPTER"), so accept C/G/O. Anchored to line start and
# requiring a separator after the numeral avoids matching mid-sentence
# cross-references like "...framed under Chapter XII of these Regulations".
CHAPTER_RE = re.compile(
    r"^\s*[CGO]HAPTER\s+([IVXLC]+)\s*[-–._]\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s: str) -> int:
    s = s.upper()
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _detect_chapter(text: str, last_value: int) -> tuple[str, int] | None:
    """Return (normalized_label, roman_value) for the next real chapter heading on
    this page, or None. Picks the smallest chapter number greater than the last one
    seen, so backward cross-references are ignored and skipped numbers are caught."""
    best = None
    for m in CHAPTER_RE.finditer(text):
        # OCR renders 'I' as lowercase 'l'. Chapters only run I..XVII, so the letter
        # L (50) never legitimately appears — normalize any L back to I.
        numeral = m.group(1).upper().replace("L", "I")
        val = _roman_to_int(numeral)
        if val <= last_value:                 # cross-reference to an earlier chapter
            continue
        title = " ".join(m.group(2).split()).rstrip(",.")
        label = f"CHAPTER {numeral}" + (f" - {title}" if title else "")
        if best is None or val < best[1]:
            best = (label, val)
    return best


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    doc_type: str
    page: int = 0
    section: str = ""
    meta: dict = field(default_factory=dict)


# Friendly document names for citations (instead of raw filenames).
_DOC_NAME = {
    REGULATION: "Air India Service Regulations",
    PROSE: "Air India Fact Sheet",
    ARTICLE: "Air India Disasters (Britannica)",
    ROUTES: "Air India Route Map, Feb 2025",
}


def make_cite(doc_type: str, page: int, section: str) -> str:
    """A trustworthy, human-readable citation label, grounded in real metadata."""
    name = _DOC_NAME.get(doc_type, "Air India document")
    if doc_type == ROUTES:
        return name
    parts = [name]
    if doc_type == REGULATION and section:
        parts.append(section)
    if page:
        parts.append(f"p{page}")
    return ", ".join(parts)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _atomize(paragraphs: list[str], size: int) -> list[str]:
    """Break any paragraph longer than `size` into line- then char-sized units,
    so the packer never emits a chunk much larger than CHUNK_SIZE (important for
    the route lists, which are one giant newline-separated block)."""
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= size:
            units.append(p)
            continue
        buf = ""
        for line in p.split("\n"):
            if buf and len(buf) + len(line) + 1 > size:
                units.append(buf)
                buf = line
            elif len(line) > size:                      # single very long line
                for j in range(0, len(line), size):
                    units.append(line[j:j + size])
                buf = ""
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            units.append(buf)
    return units


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Greedy pack paragraphs into ~`size`-char windows with char `overlap`."""
    chunks, cur = [], ""
    for p in _atomize(paragraphs, size):
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n\n" + p).strip()
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def load_regulation(path: Path, doc_type: str) -> list[Chunk]:
    """94-page legal doc: chunk per page but tag each chunk with the most recent
    CHAPTER heading so a retrieved clause carries its structural context."""
    reader = pypdf.PdfReader(str(path))
    chunks: list[Chunk] = []
    current_chapter = ""
    last_value = 0
    for pno, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean.clean_generic(raw)
        if not text:
            continue
        found = _detect_chapter(text, last_value)
        if found:
            current_chapter, last_value = found
        for i, body in enumerate(_pack(_split_paragraphs(text), config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            # Prefix as context (not a bracketed token) so the model doesn't mistake
            # it for the citation — the real citation comes from metadata (Fix B).
            header = f"(Section: {current_chapter})\n" if current_chapter else ""
            chunks.append(Chunk(
                id=f"{path.stem}-p{pno}-{i}",
                text=f"{header}{body}",
                source=path.name, doc_type=doc_type, page=pno, section=current_chapter,
            ))
    return chunks


def load_paged(path: Path, doc_type: str, cleaner=clean.clean_generic) -> list[Chunk]:
    reader = pypdf.PdfReader(str(path))
    chunks: list[Chunk] = []
    for pno, page in enumerate(reader.pages, start=1):
        text = cleaner(page.extract_text() or "")
        if not text:
            continue
        for i, body in enumerate(_pack(_split_paragraphs(text), config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(Chunk(
                id=f"{path.stem}-p{pno}-{i}",
                text=body, source=path.name, doc_type=doc_type, page=pno,
            ))
    return chunks


def load_routes() -> list[Chunk]:
    """Use the Gemini-Vision-extracted prose (run src.maps_extract first)."""
    path = config.DATA_DIR / "routes_extracted.txt"
    if not path.exists():
        print("[warn] routes_extracted.txt missing — run `python -m src.maps_extract` first.")
        return []
    text = path.read_text(encoding="utf-8")
    chunks = []
    # One chunk per route-block line group keeps connections together but small.
    for i, body in enumerate(_pack(_split_paragraphs(text), config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
        chunks.append(Chunk(
            id=f"routes-{i}", text=body, source="Route Maps Feb 2025",
            doc_type=ROUTES, page=0, section="route network",
        ))
    return chunks


def load_all() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for fname, dtype in SOURCES.items():
        path = config.PDF_DIR / fname
        if not path.exists():
            print(f"[skip] missing {fname}")
            continue
        if dtype == REGULATION:
            got = load_regulation(path, dtype)
        elif dtype == ARTICLE:
            got = load_paged(path, dtype, cleaner=clean.clean_britannica)
        else:
            got = load_paged(path, dtype, cleaner=clean.clean_generic)
        print(f"[load] {fname}: {len(got)} chunks")
        all_chunks.extend(got)

    routes = load_routes()
    print(f"[load] Route Maps: {len(routes)} chunks")
    all_chunks.extend(routes)
    print(f"[load] TOTAL: {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    cs = load_all()
    # Show a couple of samples per doc_type for a sanity check.
    seen = set()
    for c in cs:
        if c.doc_type not in seen:
            seen.add(c.doc_type)
            print(f"\n--- {c.doc_type} | {c.source} p{c.page} | id={c.id}")
            print(c.text[:280])
