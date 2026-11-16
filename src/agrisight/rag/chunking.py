"""Load guideline documents (.md, .txt, .pdf) and split them into overlapping chunks."""
from __future__ import annotations

import re
from pathlib import Path


def load_documents(kb_dir: str | Path) -> list[dict]:
    docs = []
    for path in sorted(Path(kb_dir).rglob("*")):
        if path.name.lower() == "readme.md" or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            docs.append({"text": path.read_text(encoding="utf-8", errors="ignore"), "source": path.name, "page": None})
        elif suffix == ".pdf":
            from pypdf import PdfReader

            for i, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "source": path.name, "page": i})
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Pack paragraphs into chunks of about `chunk_size` characters.

    Consecutive chunks share the last `overlap` characters so a fact split across
    a boundary is still retrievable. Very long paragraphs are cut with a sliding window.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    pieces: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        step = max(1, chunk_size - overlap)
        for start in range(0, len(para), step):
            pieces.append(para[start : start + chunk_size])
            if start + chunk_size >= len(para):
                break

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    chunks = []
    for doc in docs:
        for text in chunk_text(doc["text"], chunk_size, overlap):
            chunks.append({"id": len(chunks), "text": text, "source": doc["source"], "page": doc["page"]})
    return chunks
