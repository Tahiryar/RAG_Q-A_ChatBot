"""Local CPU embeddings with sentence-transformers/all-MiniLM-L6-v2."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """The embedding model could not be loaded or failed while encoding."""


class EmbeddingService:
    """Lazy-loading wrapper. The same model embeds both chunks and questions."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model once (first call downloads it, about 90 MB)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run: pip install -r backend/requirements.txt"
                ) from exc
            except Exception as exc:  # download / disk / corrupted-cache errors vary by library
                logger.exception("Embedding model failed to load")
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}. "
                    "The first run needs an internet connection to download it."
                ) from exc

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # torch raises several unrelated error types
            logger.exception("Embedding failed")
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
