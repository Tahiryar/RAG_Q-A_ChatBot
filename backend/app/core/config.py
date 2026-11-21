"""Application settings, loaded once from environment variables / .env."""
from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

# backend/app/core/config.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Default evidence threshold (cosine distance). See README "Evidence threshold" for how it was
# chosen and how to tune it for your own documents.
DEFAULT_MAX_DISTANCE = 0.80

KNOWN_SETTINGS = {
    "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "UPLOAD_DIR",
    "CHROMA_DIR", "CHROMA_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "MAX_DISTANCE",
    "MAX_CONTEXT_CHUNKS", "MAX_UPLOAD_MB", "MAX_HISTORY_MESSAGES", "EXCERPT_CHARS", "BACKEND_URL",
}


def warn_about_unknown_settings(env_file: Path) -> list[str]:
    """Warn about names in .env that the app does not read (for example HUNK_SIZE for CHUNK_SIZE)."""
    if not env_file.exists():
        return []
    unknown = [name for name in dotenv_values(env_file) if name not in KNOWN_SETTINGS]
    for name in unknown:
        close = difflib.get_close_matches(name.upper(), KNOWN_SETTINGS, n=1, cutoff=0.6)
        hint = f" Did you mean {close[0]}?" if close else ""
        logger.warning("Unknown setting %r in %s is ignored.%s", name, env_file.name, hint)
    return unknown


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    groq_base_url: str
    embedding_model: str
    embedding_device: str
    upload_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chunks: int
    max_distance: float
    max_upload_mb: int
    max_history_messages: int
    excerpt_chars: int

    @property
    def llm_configured(self) -> bool:
        key = (self.groq_api_key or "").strip()
        return bool(key) and key != "your_api_key_here"


def load_settings() -> Settings:
    """Read settings from the environment (and the project's .env file)."""
    env_file = PROJECT_ROOT / ".env"
    load_dotenv(env_file)
    warn_about_unknown_settings(env_file)

    def _path(name: str, default: str) -> Path:
        value = Path(os.getenv(name, default))
        return value if value.is_absolute() else PROJECT_ROOT / value

    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        upload_dir=_path("UPLOAD_DIR", "data/uploads"),
        chroma_dir=_path("CHROMA_DIR", "data/chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "documents"),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        max_context_chunks=_env_int("MAX_CONTEXT_CHUNKS", 20),
        max_distance=_env_float("MAX_DISTANCE", DEFAULT_MAX_DISTANCE),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 25),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 6),
        excerpt_chars=_env_int("EXCERPT_CHARS", 300),
    )
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if settings.top_k < 1:
        raise ValueError("TOP_K must be at least 1")
    if settings.max_context_chunks < settings.top_k:
        raise ValueError("MAX_CONTEXT_CHUNKS must be greater than or equal to TOP_K")
    logger.info(
        "Settings: chunk_size=%d chunk_overlap=%d top_k=%d max_context_chunks=%d max_distance=%.2f embedding_model=%s groq_model=%s",
        settings.chunk_size, settings.chunk_overlap, settings.top_k, settings.max_context_chunks, settings.max_distance,
        settings.embedding_model, settings.groq_model,
    )
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
