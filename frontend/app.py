"""Streamlit front end for the Grounded Document Q&A Assistant."""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_TIMEOUT_SECONDS = 1800  # the first upload can include downloading the embedding model
CHAT_TIMEOUT_SECONDS = 120
SHORT_TIMEOUT_SECONDS = 15
HISTORY_MESSAGES_SENT = 10


class BackendError(Exception):
    """Any problem talking to the backend, with a message that is safe to show the user."""


def _request(method: str, path: str, timeout: int, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise BackendError(
            f"Cannot reach the backend at {BACKEND_URL}. Start it with: "
            "uvicorn backend.app.main:app --port 8000"
        ) from exc
    except requests.Timeout as exc:
        raise BackendError("The backend took too long to respond. Please try again.") from exc
    except requests.RequestException as exc:
        raise BackendError(f"Request failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        raise BackendError(str(detail) or f"Backend error {response.status_code}")
    return response.json()


def get_health() -> dict:
    return _request("GET", "/health", SHORT_TIMEOUT_SECONDS)


def list_documents() -> list[dict]:
    return _request("GET", "/documents", SHORT_TIMEOUT_SECONDS)["documents"]


def upload_pdf(name: str, data: bytes) -> dict:
    files = {"file": (name, data, "application/pdf")}
    return _request("POST", "/documents/upload", UPLOAD_TIMEOUT_SECONDS, files=files)


def delete_document(doc_id: str) -> dict:
    return _request("DELETE", f"/documents/{doc_id}", SHORT_TIMEOUT_SECONDS)


def ask_question(question: str, history: list[dict]) -> dict:
    payload = {"question": question, "history": history}
    return _request("POST", "/chat", CHAT_TIMEOUT_SECONDS, json=payload)


def render_sources(sources: list[dict]) -> None:
    st.divider()
    st.markdown("**Sources**")
    for source in sources:
        st.markdown(f"📄 **{source['filename']}** — page {source['page']} · document `{source.get('doc_id', '')[:8]}`")
        if source.get("section"):
            st.caption(f"Section: {source['section']}")
        st.caption(f"Chunk: {source['chunk_id']}")
        st.caption(f"“{source['excerpt']}”")


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("upload_notes", [])


def process_uploads(files) -> None:
    notes: list[tuple[str, str]] = []
    for uploaded in files:
        with st.spinner(f"Processing {uploaded.name} ..."):
            try:
                result = upload_pdf(uploaded.name, uploaded.getvalue())
            except BackendError as exc:
                notes.append(("error", f"{uploaded.name}: {exc}"))
                continue
        level = "success" if result["status"] == "indexed" else "info"
        notes.append((level, f"{uploaded.name}: {result['message']}"))
    st.session_state.upload_notes = notes
    st.session_state.uploader_key += 1  # resets the uploader widget
    st.rerun()


def render_sidebar(backend_ok: bool) -> None:
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            disabled=not backend_ok,
        )
        if st.button("Upload and process", disabled=not (files and backend_ok)):
            process_uploads(files)

        for level, text in st.session_state.upload_notes:
            getattr(st, level)(text)

        st.subheader("Uploaded documents")
        if backend_ok:
            try:
                documents = list_documents()
            except BackendError as exc:
                st.error(str(exc))
                documents = []
            if not documents:
                st.caption("No documents yet.")
            for doc in documents:
                st.markdown(f"**{doc['filename']}** · ID `{doc['doc_id'][:8]}`  \n{doc['page_count']} pages · {doc['chunk_count']} chunks")
                if st.button("Delete", key=f"delete_{doc['doc_id']}"):
                    try:
                        delete_document(doc["doc_id"])
                        st.session_state.upload_notes = []
                    except BackendError as exc:
                        st.error(str(exc))
                    st.rerun()

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Grounded Document Q&A", page_icon="📄", layout="wide")
    init_state()

    st.title("Grounded Document Q&A Assistant")
    st.caption("Ask questions about your uploaded PDFs. Answers come only from the documents and cite file and page.")

    backend_ok = True
    try:
        health = get_health()
        if not health.get("llm_configured"):
            st.warning("GROQ_API_KEY is not configured on the backend. Uploads work, but questions will fail.")
    except BackendError as exc:
        backend_ok = False
        st.error(str(exc))

    render_sidebar(backend_ok)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about your documents", disabled=not backend_ok)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents ..."):
            try:
                reply = ask_question(question, history[-HISTORY_MESSAGES_SENT:])
            except BackendError as exc:
                st.error(str(exc))
                return
        st.markdown(reply["answer"])
        if reply["sources"]:
            render_sources(reply["sources"])
        elif reply.get("closest_distance") is not None:
            st.caption(
                f"Closest passage distance: {reply['closest_distance']} "
                f"(passages must be within {reply['distance_limit']})"
            )
    st.session_state.messages.append(
        {"role": "assistant", "content": reply["answer"], "sources": reply["sources"]}
    )


main()
