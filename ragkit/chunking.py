"""Paragraph/sentence aware chunking with overlap."""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Split text into chunks of at most ~max_chars, never cutting inside a sentence.

    Consecutive chunks share `overlap_sentences` sentences so that an answer that
    straddles a boundary can still be retrieved. A single sentence longer than
    max_chars becomes its own chunk.
    """
    if max_chars < 50:
        raise ValueError("max_chars must be >= 50")
    sentences = [s.strip() for s in _SENT.split(" ".join(text.split())) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        if current and size + len(sent) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(s) + 1 for s in current)
        current.append(sent)
        size += len(sent) + 1
    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
