"""Embed a question and fetch the closest chunks from the vector store."""
from __future__ import annotations

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store_service import SearchHit, VectorStore


def merge_hits(*hit_lists: list[SearchHit]) -> list[SearchHit]:
    """Combine several result lists: keep each chunk once (best distance), nearest first."""
    best: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for hit in hits:
            current = best.get(hit.chunk_id)
            if current is None or hit.distance < current.distance:
                best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda h: h.distance)


class RetrievalService:
    def __init__(self, embedder: EmbeddingService, store: VectorStore, top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def has_documents(self) -> bool:
        return self._store.count() > 0

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return up to top_k nearest chunks (empty list if nothing is stored)."""
        query = query.strip()
        if not query or not self.has_documents():
            return []
        embedding = self._embedder.embed_query(query)
        return self._store.search(embedding, top_k or self._top_k)
