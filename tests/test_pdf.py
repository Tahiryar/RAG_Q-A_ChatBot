"""PDF extraction, chunking and upload validation."""
import pymupdf
import pytest

from backend.app.services.chunking_service import chunk_pages
from backend.app.services.pdf_service import EmptyPdfError, InvalidPdfError, extract_pages
from tests.conftest import REFUND_V1, make_pdf, upload


def test_extracts_text_with_filename_and_page_numbers():
    doc = extract_pages(make_pdf(REFUND_V1), "handbook.pdf")
    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert all(p.filename == "handbook.pdf" for p in doc.pages)
    assert "Lahore" in doc.pages[0].text
    assert "30 days" in doc.pages[1].text


def test_blank_page_is_skipped_but_page_numbers_stay_correct():
    doc = extract_pages(make_pdf(["First page text is here.", "", "Third page has the answer."]), "x.pdf")
    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert "answer" in doc.pages[1].text


@pytest.mark.parametrize(
    "data",
    [b"", b"just some plain text, not a pdf", b"%PDF-1.4 this is garbage after the header"],
    ids=["empty-file", "not-a-pdf", "corrupted"],
)
def test_invalid_pdfs_raise_invalid_pdf_error(data):
    with pytest.raises(InvalidPdfError):
        extract_pages(data, "bad.pdf")


def test_pdf_without_any_text_raises_empty_pdf_error():
    with pytest.raises(EmptyPdfError):
        extract_pages(make_pdf(["", ""]), "blank.pdf")


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret text")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner")
    with pytest.raises(InvalidPdfError):
        extract_pages(data, "locked.pdf")


def test_chunks_keep_filename_and_original_page():
    long_page = " ".join(f"Sentence number {i} about the warranty terms." for i in range(120))
    doc = extract_pages(make_pdf(["Short first page with enough words.", long_page[:2800]]), "terms.pdf")
    chunks = chunk_pages(doc.pages, "doc123", chunk_size=500, chunk_overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    assert sum(1 for c in chunks if c.page == 2) > 1  # long page was split
    assert all(c.filename == "terms.pdf" and c.doc_id == "doc123" for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ---- upload endpoint validation -------------------------------------------------
def test_upload_rejects_non_pdf_extension(client):
    response = upload(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_fake_pdf(client):
    response = upload(client, "fake.pdf", b"this is not a pdf")
    assert response.status_code == 400
    assert client.get("/documents").json()["documents"] == []


def test_upload_rejects_pdf_without_text(client):
    response = upload(client, "blank.pdf", make_pdf(["", ""]))
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]


def test_upload_rejects_oversized_file(client):
    response = upload(client, "big.pdf", b"%PDF-" + b"0" * (6 * 1024 * 1024))
    assert response.status_code == 413
