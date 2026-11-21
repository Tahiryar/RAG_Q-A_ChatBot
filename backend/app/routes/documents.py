"""Endpoints for uploading, listing and deleting PDF documents."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from backend.app.schemas.document import DeleteResponse, DocumentInfo, DocumentListResponse, UploadResult
from backend.app.services import pdf_service
from backend.app.services.chunking_service import chunk_pages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

_DOC_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def _safe_disk_name(doc_id: str, filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return f"{doc_id[:12]}_{cleaned}"


def _to_info(summary) -> DocumentInfo:
    return DocumentInfo(
        doc_id=summary.doc_id,
        filename=summary.filename,
        page_count=summary.page_count,
        chunk_count=summary.chunk_count,
    )


@router.post("/upload", response_model=UploadResult)
def upload_document(request: Request, file: UploadFile) -> UploadResult:
    """Validate, extract, chunk, embed and store one PDF. Duplicates are detected by file hash."""
    state = request.app.state
    settings = state.settings

    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are supported.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB.")

    doc_id = _hash_bytes(data)

    with state.ingest_lock:
        existing = state.vector_store.get_document(doc_id)
        if existing is not None:
            return UploadResult(
                filename=filename,
                status="duplicate",
                message=f"This file was already indexed (as '{existing.filename}'). Nothing was added.",
                document=_to_info(existing),
            )

        extracted = pdf_service.extract_pages(data, filename)  # raises PdfProcessingError
        chunks = chunk_pages(extracted.pages, doc_id, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise pdf_service.EmptyPdfError(f"'{filename}' contains no usable text.")

        embeddings = state.embedder.embed_documents([c.text for c in chunks])
        state.vector_store.add_chunks(chunks, embeddings, extracted.page_count)

        try:
            settings.upload_dir.mkdir(parents=True, exist_ok=True)
            (settings.upload_dir / _safe_disk_name(doc_id, filename)).write_bytes(data)
        except OSError as exc:
            logger.exception("Could not save %s to disk; removing its vectors", filename)
            state.vector_store.delete_document(doc_id)
            raise HTTPException(status_code=500, detail=f"Could not save the uploaded file: {exc}") from exc

    logger.info("Indexed %s (%d pages with text, %d chunks)", filename, len(extracted.pages), len(chunks))
    return UploadResult(
        filename=filename,
        status="indexed",
        message=f"Indexed {len(chunks)} chunks from {extracted.page_count} pages.",
        document=DocumentInfo(
            doc_id=doc_id, filename=filename, page_count=extracted.page_count, chunk_count=len(chunks)
        ),
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    summaries = request.app.state.vector_store.list_documents()
    return DocumentListResponse(documents=[_to_info(s) for s in summaries])


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, request: Request) -> DeleteResponse:
    """Delete one document's vectors and its saved PDF. Other documents are untouched."""
    if not _DOC_ID_PATTERN.match(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    state = request.app.state
    with state.ingest_lock:
        deleted = state.vector_store.delete_document(doc_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
        for path in state.settings.upload_dir.glob(f"{doc_id[:12]}_*"):
            try:
                path.unlink()
            except OSError:
                logger.exception("Could not remove %s", path)
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)
