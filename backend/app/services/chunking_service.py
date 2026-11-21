"""Split page text into chunks while keeping filename and page metadata."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.services.pdf_service import PageText

MIN_CHUNK_CHARS = 20  # drop fragments such as lone page numbers


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    filename: str
    page: int
    chunk_index: int  # running index inside the document
    text: str
    section: str = ""


def chunk_pages(
    pages: list[PageText], doc_id: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk each page separately so a chunk never spans two pages."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    filename=page.filename,
                    page=page.page_number,
                    chunk_index=len(chunks),
                    text=piece,
                    section=page.section,
                )
            )
    return chunks
