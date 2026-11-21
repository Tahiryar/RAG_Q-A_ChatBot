"""Persistent ChromaDB storage for chunk text, embeddings and metadata."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from backend.app.services.chunking_service import Chunk

logger = logging.getLogger(__name__)

_ADD_BATCH_SIZE = 256


class VectorStoreError(Exception):
    """ChromaDB could not be initialised or an operation on it failed."""


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    distance: float  # cosine distance: 0 = identical, larger = less similar
    section: str = ""


class VectorStore:
    """Thin wrapper around one persistent Chroma collection (cosine distance)."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            # embedding_function=None: we always pass our own vectors, so Chroma never
            # tries to download its default embedding model.
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as exc:  # chromadb raises many unrelated exception types
            logger.exception("ChromaDB initialisation failed")
            raise VectorStoreError(f"Could not open the vector database at {persist_dir}: {exc}") from exc

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Vector database error: {exc}") from exc

    def has_document(self, doc_id: str) -> bool:
        return self.get_document(doc_id) is not None

    def get_document(self, doc_id: str) -> DocumentSummary | None:
        for summary in self.list_documents():
            if summary.doc_id == doc_id:
                return summary
        return None

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], page_count: int) -> None:
        """Add chunks with upsert (re-adding the same chunk ids never duplicates them)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        doc_id = chunks[0].doc_id
        try:
            for start in range(0, len(chunks), _ADD_BATCH_SIZE):
                batch = chunks[start : start + _ADD_BATCH_SIZE]
                self._collection.upsert(
                    ids=[f"{c.doc_id}:{c.chunk_index}" for c in batch],
                    documents=[c.text for c in batch],
                    embeddings=embeddings[start : start + _ADD_BATCH_SIZE],
                    metadatas=[
                        {
                            "doc_id": c.doc_id,
                            "filename": c.filename,
                            "page": c.page,
                            "chunk_index": c.chunk_index,
                            "section": c.section,
                            "page_count": page_count,
                        }
                        for c in batch
                    ],
                )
        except Exception as exc:
            logger.exception("Failed to store chunks for %s; rolling back", doc_id)
            self._rollback(doc_id)
            raise VectorStoreError(f"Could not store document in the vector database: {exc}") from exc

    def _rollback(self, doc_id: str) -> None:
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception:  # best effort cleanup; the original error is raised by the caller
            logger.exception("Rollback of partially stored document %s failed", doc_id)

    def list_documents(self) -> list[DocumentSummary]:
        try:
            records = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Vector database error: {exc}") from exc
        grouped: dict[str, dict] = {}
        for meta in records.get("metadatas") or []:
            entry = grouped.setdefault(
                meta["doc_id"],
                {"filename": meta["filename"], "page_count": int(meta["page_count"]), "chunks": 0},
            )
            entry["chunks"] += 1
        summaries = [
            DocumentSummary(doc_id, e["filename"], e["page_count"], e["chunks"])
            for doc_id, e in grouped.items()
        ]
        return sorted(summaries, key=lambda s: (s.filename.lower(), s.doc_id))

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of one document. Returns how many chunks were deleted."""
        summary = self.get_document(doc_id)
        if summary is None:
            return 0
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Could not delete document: {exc}") from exc
        return summary.chunk_count

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        total = self.count()
        if total == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc
        hits: list[SearchHit] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page=int(meta["page"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=text,
                    section=meta.get("section", ""),
                    distance=float(distance),
                )
            )
        return hits
