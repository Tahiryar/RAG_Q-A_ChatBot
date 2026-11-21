"""PDF structure extraction with Docling, preserving physical page numbers and section headings."""
from __future__ import annotations

import io
import logging
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_converter = None
_converter_lock = threading.Lock()


def _get_converter():
    global _converter
    if _converter is None:
        with _converter_lock:
            if _converter is None:
                from docling.document_converter import DocumentConverter
                _converter = DocumentConverter()
    return _converter


class PdfProcessingError(Exception):
    """Base class for problems with an uploaded PDF."""


class InvalidPdfError(PdfProcessingError):
    """The file is not a readable PDF (wrong type, corrupted, or password protected)."""


class EmptyPdfError(PdfProcessingError):
    """The PDF has no pages or no extractable text."""


@dataclass(frozen=True)
class PageText:
    filename: str
    page_number: int
    text: str
    section: str = ""


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    page_count: int
    pages: list[PageText]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes, filename: str) -> ExtractedDocument:
    """Convert a PDF through Docling and collect its structured items page by page."""
    if not pdf_bytes:
        raise InvalidPdfError(f"'{filename}' is an empty file.")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise InvalidPdfError(f"'{filename}' is not a valid PDF file.")

    try:
        from docling.datamodel.base_models import DocumentStream
        source = DocumentStream(name=filename, stream=io.BytesIO(pdf_bytes))
        result = _get_converter().convert(source)
        document = result.document
        page_count = len(document.pages)
        if page_count == 0:
            raise EmptyPdfError(f"'{filename}' has no pages.")

        page_items: dict[int, list[str]] = {i: [] for i in range(1, page_count + 1)}
        page_sections: dict[int, str] = {i: "" for i in range(1, page_count + 1)}
        current_section = ""

        # Docling's item provenance maps text, headings and tables back to source pages.
        for item, _level in document.iterate_items():
            label = str(getattr(item, "label", "") or "").lower()
            text = getattr(item, "text", None)
            if not text and hasattr(item, "export_to_dataframe"):
                try:
                    text = item.export_to_dataframe(doc=document).to_string(index=False)
                except Exception:
                    text = ""
            text = _clean(str(text or ""))
            prov = getattr(item, "prov", None) or []
            page_numbers = sorted({int(p.page_no) for p in prov if getattr(p, "page_no", None)})
            if not page_numbers:
                continue
            if "title" in label or "section_header" in label or "heading" in label:
                if text:
                    current_section = text[:300]
            for page_no in page_numbers:
                if page_no not in page_items:
                    continue
                if text:
                    page_items[page_no].append(text)
                if current_section:
                    page_sections[page_no] = current_section

        pages = []
        for page_no in range(1, page_count + 1):
            text = _clean("\\n".join(page_items[page_no]))
            if text:
                pages.append(PageText(filename, page_no, text, page_sections[page_no]))

        if not pages:
            raise EmptyPdfError(
                f"'{filename}' contains no extractable text. Check whether the PDF is scanned or image-only."
            )
        logger.info("Docling extracted %d/%d text pages from %s", len(pages), page_count, filename)
        return ExtractedDocument(filename, page_count, pages)

    except (InvalidPdfError, EmptyPdfError):
        raise
    except Exception as exc:
        logger.exception("Docling could not process %s", filename)
        message = str(exc).lower()
        if "password" in message or "encrypted" in message:
            raise InvalidPdfError(f"'{filename}' is password protected and cannot be read.") from exc
        if "no pages" in message or "not a pdf" in message or "invalid pdf" in message:
            raise InvalidPdfError(f"'{filename}' could not be read. It may be corrupted.") from exc
        raise InvalidPdfError(f"Docling could not process '{filename}': {exc}") from exc
