"""FastAPI application: wiring, error handling and health check."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.core.config import Settings, get_settings
from backend.app.routes import chat, documents
from backend.app.services import rag_service
from backend.app.services.embedding_service import EmbeddingError, EmbeddingService
from backend.app.services.pdf_service import EmptyPdfError, PdfProcessingError
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store_service import VectorStore, VectorStoreError

logger = logging.getLogger("rag")


def _error_response(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


async def _pdf_error_handler(request: Request, exc: PdfProcessingError) -> JSONResponse:
    status_code = 422 if isinstance(exc, EmptyPdfError) else 400
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingService | None = None,
    llm: rag_service.LLMClient | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake embedder / LLM; production uses the defaults."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        cfg = settings or get_settings()
        embed = embedder or EmbeddingService(cfg.embedding_model, cfg.embedding_device)
        store = VectorStore(cfg.chroma_dir, cfg.collection_name)
        retrieval = RetrievalService(embed, store, cfg.top_k)

        llm_client = llm
        if llm_client is None and cfg.llm_configured:
            llm_client = rag_service.GroqClient(cfg.groq_api_key, cfg.groq_model, cfg.groq_base_url)
        if llm_client is None:
            logger.warning("GROQ_API_KEY is not set: uploads work, but chat answers will fail.")

        try:
            embed.load()
        except EmbeddingError as exc:
            logger.error("Embedding model not ready at startup (will retry on first use): %s", exc)

        app.state.settings = cfg
        app.state.embedder = embed
        app.state.vector_store = store
        app.state.ingest_lock = threading.Lock()
        app.state.llm_configured = llm_client is not None
        app.state.rag_service = rag_service.RagService(
            retrieval,
            llm_client,
            max_distance=cfg.max_distance,
            top_k=cfg.top_k,
            max_history_messages=cfg.max_history_messages,
            excerpt_chars=cfg.excerpt_chars,
            max_context_chunks=cfg.max_context_chunks,
        )
        yield

    app = FastAPI(title="Grounded Document Q&A Assistant", version="1.0.0", lifespan=lifespan)
    app.include_router(documents.router)
    app.include_router(chat.router)

    app.add_exception_handler(PdfProcessingError, _pdf_error_handler)
    app.add_exception_handler(EmbeddingError, _error_response(503))
    app.add_exception_handler(VectorStoreError, _error_response(500))
    app.add_exception_handler(rag_service.LLMConfigError, _error_response(503))
    app.add_exception_handler(rag_service.LLMAuthError, _error_response(502))
    app.add_exception_handler(rag_service.LLMRateLimitError, _error_response(429))
    app.add_exception_handler(rag_service.LLMUnavailableError, _error_response(503))
    app.add_exception_handler(rag_service.LLMError, _error_response(502))

    @app.get("/health", tags=["health"])
    def health(request: Request) -> dict:
        state = request.app.state
        return {
            "status": "ok",
            "documents": len(state.vector_store.list_documents()),
            "llm_configured": state.llm_configured,
            "embedding_model_loaded": state.embedder.is_loaded,
        }

    return app


app = create_app()
