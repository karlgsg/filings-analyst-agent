"""Split page text into overlapping chunks that never cross a page boundary.

Sizes are in words (~0.75 words per token, so 450 words is roughly 600 tokens).
Words are a simple, model-agnostic proxy; exact token counts aren't needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

CHUNK_WORDS = 450
OVERLAP_WORDS = 80


@dataclass
class Chunk:
    doc_name: str
    page: int  # 1-indexed, as shown in a PDF viewer
    chunk_index: int  # position within the page
    text: str


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Sliding window over words: each window starts `size - overlap` words after the last."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return pieces


def chunk_pages(doc_name: str, pages: list[str]) -> list[Chunk]:
    """pages[i] is the text of page i+1."""
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        for i, piece in enumerate(split_words(text)):
            chunks.append(Chunk(doc_name=doc_name, page=page_num, chunk_index=i, text=piece))
    return chunks
