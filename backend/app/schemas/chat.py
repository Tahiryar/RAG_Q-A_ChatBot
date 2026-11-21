"""Pydantic models for the chat endpoint."""
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=5000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)


class Source(BaseModel):
    filename: str
    doc_id: str
    chunk_id: str
    page: int
    section: str = ""
    excerpt: str
    distance: float


class ChatResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]
    # Filled when the evidence check rejects every passage, so users can see how close the
    # best passage was (cosine distance) and the limit that was applied.
    closest_distance: float | None = None
    distance_limit: float | None = None
