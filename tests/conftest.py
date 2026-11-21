"""Shared test helpers: fake embedder, fake Groq, in-memory PDF builder, app fixture."""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "its", "my", "of", "on", "or", "our", "that", "the", "their", "to", "was",
    "what", "when", "which", "who", "will", "with", "you", "your", "have", "has", "many", "much",
}

REFUND_V1 = [
    "Acme Company Handbook. Our head office is located in Lahore and opens at nine in the morning.",
    "Refund policy. Customers may return a product within 30 days of purchase for a full refund. "
    "The item must be unused and in its original packaging.",
]
REFUND_V2 = [
    "Updated refund policy. Customers may return a product within 14 days of purchase for a full refund. "
    "Items returned after that period are not accepted.",
]
SHIPPING = [
    "Shipping information. Standard shipping takes five to seven business days within Pakistan. "
    "Express shipping costs 500 rupees.",
]


class FakeEmbedder:
    """Deterministic bag-of-words embedder (hashing trick). NOT semantic: it only matches shared words."""

    dim = 512
    is_loaded = True

    def load(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOPWORDS:
                continue
            word = re.sub(r"(ing|ed|es|s)$", "", word) if len(word) > 4 else word
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    """Records every call and answers with whatever `responder(system, user)` returns."""

    def __init__(self, responder=None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.responder = responder or (lambda system, user: llm_json(False, "Not found in the documents."))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.responder(system_prompt, user_prompt)


def llm_json(found: bool, answer: str, citations: list[tuple[str, str]] | None = None) -> str:
    return json.dumps(
        {"found": found, "answer": answer, "citations": [{"source": s, "quote": q} for s, q in citations or []]}
    )


def prompt_sources(user_prompt: str) -> list[dict]:
    """Parse the <source ...> blocks out of the prompt that was sent to the LLM."""
    pattern = r'<source id="(S[0-9]+)" .*?file="([^"]*)" page="([0-9]+)"[^>]*>\n(.*?)\n</source>'
    return [
        {"id": i, "file": f, "page": int(p), "text": t}
        for i, f, p, t in re.findall(pattern, user_prompt, flags=re.S)
    ]


def make_pdf(pages: list[str]) -> bytes:
    """Build a small PDF in memory. An empty string creates a blank page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = Settings(
        groq_api_key="test-key",
        groq_model="test-model",
        groq_base_url="https://api.groq.com/openai/v1",
        embedding_model="fake",
        embedding_device="cpu",
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        collection_name="documents",
        chunk_size=1000,
        chunk_overlap=150,
        top_k=5,
        max_context_chunks=20,
        max_distance=0.9,  # tuned for the fake embedder; real MiniLM default is 0.65
        max_upload_mb=5,
        max_history_messages=6,
        excerpt_chars=300,
    )
    return replace(base, **overrides)


def upload(client: TestClient, name: str, data: bytes, content_type: str = "application/pdf"):
    return client.post("/documents/upload", files={"file": (name, data, content_type)})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM):
    app = create_app(settings=settings, embedder=FakeEmbedder(), llm=fake_llm)
    with TestClient(app) as test_client:
        yield test_client
