"""Pydantic models for the document endpoints."""
from typing import Literal

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate"]
    message: str
    document: DocumentInfo


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
