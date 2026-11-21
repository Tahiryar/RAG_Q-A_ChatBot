"""Vector store persistence, duplicate handling and retrieval metadata."""
import os

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services.chunking_service import Chunk
from backend.app.services.retrieval_service import RetrievalService, merge_hits
from backend.app.services.vector_store_service import SearchHit, VectorStore
from tests.conftest import REFUND_V1, REFUND_V2, SHIPPING, FakeEmbedder, FakeLLM, make_pdf, upload


def _retrieval(settings) -> RetrievalService:
    return RetrievalService(FakeEmbedder(), VectorStore(settings.chroma_dir, "documents"), top_k=3)


def test_retrieves_relevant_chunk_with_correct_filename_and_page(client, settings):
    assert upload(client, "refund_v1.pdf", make_pdf(REFUND_V1)).status_code == 200
    assert upload(client, "shipping.pdf", make_pdf(SHIPPING)).status_code == 200

    hits = _retrieval(settings).retrieve("How many days to return a product for a refund?")
    assert hits, "expected at least one hit"
    best = hits[0]
    assert (best.filename, best.page) == ("refund_v1.pdf", 2)
    assert "30 days" in best.text
    assert 0.0 <= best.distance <= 1.0
    assert hits == sorted(hits, key=lambda h: h.distance)


def test_shipping_question_points_to_shipping_pdf(client, settings):
    upload(client, "refund_v1.pdf", make_pdf(REFUND_V1))
    upload(client, "shipping.pdf", make_pdf(SHIPPING))
    best = _retrieval(settings).retrieve("How much does express shipping cost?")[0]
    assert (best.filename, best.page) == ("shipping.pdf", 1)


def test_empty_store_returns_no_results(settings):
    retrieval = _retrieval(settings)
    assert retrieval.has_documents() is False
    assert retrieval.retrieve("anything at all") == []


def test_blank_query_returns_no_results(client, settings):
    upload(client, "shipping.pdf", make_pdf(SHIPPING))
    assert _retrieval(settings).retrieve("   ") == []


def test_vectors_survive_backend_restart(settings):
    first = create_app(settings=settings, embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(first) as client:
        assert upload(client, "refund_v1.pdf", make_pdf(REFUND_V1)).status_code == 200

    second = create_app(settings=settings, embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(second) as client:
        documents = client.get("/documents").json()["documents"]
        assert [d["filename"] for d in documents] == ["refund_v1.pdf"]
    hits = _retrieval(settings).retrieve("return a product refund")
    assert hits and hits[0].filename == "refund_v1.pdf"


def test_duplicate_upload_does_not_create_duplicate_chunks(client):
    data = make_pdf(REFUND_V1)
    first = upload(client, "refund_v1.pdf", data).json()
    second = upload(client, "renamed_copy.pdf", data).json()  # same bytes, different name
    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"
    documents = client.get("/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["chunk_count"] == first["document"]["chunk_count"]


def test_new_pdf_does_not_remove_existing_documents(client, settings):
    upload(client, "refund_v1.pdf", make_pdf(REFUND_V1))
    upload(client, "refund_v2.pdf", make_pdf(REFUND_V2))
    names = sorted(d["filename"] for d in client.get("/documents").json()["documents"])
    assert names == ["refund_v1.pdf", "refund_v2.pdf"]
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 2  # originals are saved to disk


def test_delete_removes_only_the_chosen_document(client, settings):
    keep = upload(client, "shipping.pdf", make_pdf(SHIPPING)).json()["document"]
    drop = upload(client, "refund_v1.pdf", make_pdf(REFUND_V1)).json()["document"]
    response = client.delete(f"/documents/{drop['doc_id']}")
    assert response.status_code == 200
    remaining = client.get("/documents").json()["documents"]
    assert [d["doc_id"] for d in remaining] == [keep["doc_id"]]
    assert len(list(settings.upload_dir.glob("*.pdf"))) == 1
    assert client.delete(f"/documents/{drop['doc_id']}").status_code == 404
    assert client.delete("/documents/..%2F..%2Fetc").status_code in (404, 400)


def test_merge_hits_keeps_best_distance_once():
    def hit(chunk_id, distance):
        return SearchHit(chunk_id, "d", "f.pdf", 1, 0, "text", distance)

    merged = merge_hits([hit("a", 0.5), hit("b", 0.3)], [hit("a", 0.2), hit("c", 0.9)])
    assert [(h.chunk_id, h.distance) for h in merged] == [("a", 0.2), ("b", 0.3), ("c", 0.9)]


def test_store_upsert_is_idempotent(settings):
    store = VectorStore(settings.chroma_dir, "documents")
    chunk = Chunk("doc1", "a.pdf", 1, 0, "some chunk text that is long enough")
    vector = FakeEmbedder().embed_query(chunk.text)
    store.add_chunks([chunk], [vector], page_count=1)
    store.add_chunks([chunk], [vector], page_count=1)
    assert store.count() == 1


@pytest.mark.skipif(
    os.getenv("RUN_REAL_EMBEDDING_TESTS") != "1",
    reason="Set RUN_REAL_EMBEDDING_TESTS=1 to test with the real all-MiniLM-L6-v2 model (needs download).",
)
def test_real_model_threshold_calibration(tmp_path):
    """Calibration check with the real model: do on-topic questions pass DEFAULT_MAX_DISTANCE,
    and does clearly unrelated text fail it? The printed table shows the real distances."""
    from backend.app.core.config import DEFAULT_MAX_DISTANCE
    from backend.app.services.embedding_service import EmbeddingService

    overview = (
        "Project Overview. Build a small web-based assistant that answers questions using a collection of "
        "uploaded documents. The system must retrieve relevant passages before generating an answer, and it "
        "must show citations that let a user verify where each answer came from. The main goal is to make "
        "answers useful while reducing unsupported claims and hallucinations."
    )
    tools = (
        "Suggested technology. Use Python for the backend. Suggested components are FastAPI or Flask for the "
        "API, PyMuPDF or pdfplumber for PDF text extraction, sentence-transformers for embeddings, and "
        "ChromaDB or FAISS for vector search. A minimal Streamlit or Gradio interface is sufficient."
    )
    embedder = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", "cpu")
    store = VectorStore(tmp_path / "chroma", "documents")
    chunks = [Chunk("d1", "brief.pdf", 1, 0, overview), Chunk("d1", "brief.pdf", 2, 1, tools)]
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]), 2)
    retrieval = RetrievalService(embedder, store, top_k=2)

    on_topic = {
        "What are the main objectives of this project?": 1,
        "Which tools are suggested for extracting PDF text?": 2,
    }
    off_topic = ["hi", "Who won the football world cup in 1998?", "What is the capital of France?"]
    table = []
    for question, expected_page in on_topic.items():
        best = retrieval.retrieve(question)[0]
        table.append(f"ON  {best.distance:.3f} page={best.page} (expected {expected_page}) {question}")
    for question in off_topic:
        best = retrieval.retrieve(question)[0]
        table.append(f"OFF {best.distance:.3f} page={best.page} {question}")
    report = "\n".join(table)
    print("\n" + report)

    for question, expected_page in on_topic.items():
        best = retrieval.retrieve(question)[0]
        assert best.page == expected_page, report
        assert best.distance <= DEFAULT_MAX_DISTANCE, f"on-topic question rejected by default limit\n{report}"
    for question in off_topic[1:]:
        assert retrieval.retrieve(question)[0].distance > DEFAULT_MAX_DISTANCE, f"off-topic accepted\n{report}"


def test_same_filename_different_pdf_content_gets_distinct_document_ids(client):
    first = upload(client, "Operations Manual.pdf", make_pdf(REFUND_V1)).json()["document"]
    second = upload(client, "Operations Manual.pdf", make_pdf(REFUND_V2)).json()["document"]
    assert first["doc_id"] != second["doc_id"]
    same_name = [d for d in client.get("/documents").json()["documents"]
                 if d["filename"] == "Operations Manual.pdf"]
    assert len(same_name) == 2
    assert len({d["doc_id"] for d in same_name}) == 2


def test_chunk_section_metadata_is_preserved():
    from backend.app.services.pdf_service import PageText
    from backend.app.services.chunking_service import chunk_pages
    chunks = chunk_pages([PageText("manual.pdf", 4, "The return policy states products can be returned within 30 days.", "Returns")],
                         "doc-1", chunk_size=100, chunk_overlap=10)
    assert chunks
    assert all(c.section == "Returns" for c in chunks)
