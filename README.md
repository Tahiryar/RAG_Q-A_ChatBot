# Grounded Document Q&A Assistant (RAG with Citations)

Upload PDFs, ask questions, and get answers that come **only** from your documents, with the
file name, page number and a short excerpt for every source. If the documents do not contain
the answer, the assistant says so instead of guessing.

## Features

- Upload several PDFs; duplicates are detected by file hash and never indexed twice
- Structure-aware PDF extraction with Docling that keeps the original filename and page number
- Chunking, local CPU embeddings, and persistent vector storage (data survives restarts)
- Adaptive retrieval: fetch a wider candidate pool, then include all evidence-qualified chunks up to MAX_CONTEXT_CHUNKS
- Groq answers restricted to the retrieved passages, returned as JSON with source ids
- Citations are built by the backend from real retrieved chunks (the LLM cannot invent a page number)
- Honest "not found" answers, conflict handling across documents, basic follow-up questions
- Delete a document without touching the others

## Technology stack

| Part | Tool |
|---|---|
| Backend API | Python, FastAPI, Pydantic |
| Frontend | Streamlit |
| PDF extraction | Docling (layout, headings, tables and page provenance) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector database | ChromaDB (persistent, cosine distance) |
| LLM | Groq API (OpenAI-compatible Chat Completions endpoint) |
| Config / tests | python-dotenv, Pytest |

## How it works (RAG architecture)

```
Ingestion:  PDF -> Docling structure per page -> chunks (+document id, filename, page, section) -> embeddings -> ChromaDB
Question:   question -> embedding -> wider candidate pool -> adaptive evidence filter -> Groq -> concise answer + verified citations -> UI
```

1. **Upload** (`routes/documents.py`): the file is validated, hashed (SHA-256) and checked for duplicates.
2. **Extract** (`pdf_service.py`): Docling parses the PDF structure and provenance page by page. Blank pages are skipped but page numbers stay correct.
3. **Chunk** (`chunking_service.py`): each page is split on its own, so a chunk never crosses a page boundary.
4. **Embed + store** (`embedding_service.py`, `vector_store_service.py`): vectors, chunk text and metadata
   (`doc_id`, `filename`, `page`, `section`, `chunk_index`, `page_count`) go into ChromaDB. The original PDF is saved in `data/uploads/`.
5. **Retrieve** (`retrieval_service.py`): the question is embedded with the same model and the nearest chunks are returned with distances.
6. **Evidence check** (`rag_service.py`): chunks farther than `MAX_DISTANCE` are dropped. If none are left, Groq is **not called**
   and the assistant answers "not found".
7. **Generate** (`rag_service.py`, `prompts.py`): only the surviving chunks (labelled S1, S2, ...) and recent chat history are sent to Groq.
8. **Verify citations**: Groq returns `found`, `answer` and `citations` (`S1`, short quote). The backend ignores unknown ids, replaces
   quotes that are not really in the chunk with real text from the chunk, and refuses answers that have no valid citation.

**After upgrading an older project database:** existing ChromaDB chunks are retained for compatibility and are not automatically re-parsed. If you want an already-uploaded PDF reprocessed by Docling, delete that document in the sidebar and upload it again.

## Document identity and adaptive retrieval

Every uploaded PDF gets a SHA-256-derived `doc_id`. Different PDFs with the same filename remain separate and are shown with their document ID. Every chunk stores its document ID, filename, original physical page number, section heading, and chunk ID. The answer sources return these fields so identical filenames cannot make citations ambiguous.

`TOP_K` is the initial retrieval setting, not a hard final limit. The system fetches a wider candidate pool and sends all candidates within `MAX_DISTANCE`, up to `MAX_CONTEXT_CHUNKS` (default 20). Raise `MAX_CONTEXT_CHUNKS` when your documents require evidence spread across more chunks. The model is instructed to answer only the asked point in 1–3 short sentences unless more detail is necessary.

## Folder structure

```
rag-document-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                      FastAPI app, wiring, error handlers, /health
│   │   ├── routes/ documents.py chat.py
│   │   ├── services/ pdf_service.py chunking_service.py embedding_service.py
│   │   │             vector_store_service.py retrieval_service.py rag_service.py
│   │   ├── core/ config.py prompts.py
│   │   └── schemas/ document.py chat.py
│   └── requirements.txt
├── frontend/app.py                      Streamlit UI
├── data/uploads/  data/chroma_db/       saved PDFs and the vector database (git-ignored)
├── sample_pdfs/                         3 small demo PDFs (two of them disagree on purpose)
├── tests/ test_pdf.py test_retrieval.py test_rag.py conftest.py
├── pytest.ini  .env.example  .gitignore  README.md
```

`tests/conftest.py`, `pytest.ini` and `sample_pdfs/` are small additions to the required layout.

## Installation (Windows PowerShell)

```powershell
cd rag-document-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.
Docling downloads its layout/OCR models on first PDF processing and can take time and disk space. The local embedding model (~90 MB) downloads automatically the first time the backend starts.

## Configure the Groq API key

1. Create an API key in [Groq Console](https://console.groq.com/keys).
2. Create your `.env` file and edit it:

```powershell
Copy-Item .env.example .env
notepad .env
```

```
GROQ_API_KEY=your_real_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
MAX_CONTEXT_CHUNKS=20
```

Change GROQ_MODEL if you want another model currently available to your Groq account. Model availability and limits can change.
`.env` is git-ignored. Never commit it.

## Run the app

Terminal 1, backend (from the project folder, venv active):

```powershell
uvicorn backend.app.main:app --port 8000
```

Terminal 2, frontend:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\app.py
```

Open http://localhost:8501, upload the PDFs from `sample_pdfs/`, and try:

- "How much does express shipping cost?" (answer with a citation)
- "How many days do customers have to return a product?" (the 2023 and 2024 policies disagree; both should be cited)
- "Who won the football world cup in 1998?" (not in the documents, so it must refuse)

The API docs are at http://localhost:8000/docs. If the backend runs on another address, set `BACKEND_URL` before starting Streamlit
(`$env:BACKEND_URL="http://127.0.0.1:9000"`).

## Run the tests

```powershell
python -m pytest -v
```

The tests use a fake embedder and a mocked LLM, so they need no internet, no API key and no model download.
To test the real embedding model (downloads it once):

```powershell
$env:RUN_REAL_EMBEDDING_TESTS="1"; python -m pytest tests\test_retrieval.py -v
```

## Design choices

**Chunk size 1000 characters, overlap 150.** About 150-200 words: large enough to keep a full policy sentence with its context, small enough that the
retrieved context stays focused and that all-MiniLM-L6-v2 (which only reads about 256 tokens) does not cut it off. The 15% overlap keeps a
sentence that falls on a boundary intact in at least one chunk. Change them with `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`, then re-upload
(old chunks are not re-chunked automatically; delete and upload again).

**Embedding model.** `all-MiniLM-L6-v2` is small (~90 MB), fast on CPU, free, and good enough for English semantic search. The same model embeds
chunks and questions, which is required for the distances to mean anything.

**Vector database.** ChromaDB stores vectors, text and metadata in one local folder with no server. It is persistent, so restarting the backend does not
re-embed anything. Cosine distance is used (0 = identical).

**Evidence threshold (`MAX_DISTANCE`, default 0.80).** Cosine distance is `1 - similarity` (0 = identical). A chunk is only sent to Groq if
its distance is at most `MAX_DISTANCE`; if no chunk qualifies, Groq is not called and the answer is "not found".

Why 0.80: the first default (0.65) was too strict. A real on-topic question ("What are the main objectives of this project?") had its best chunk at
distance 0.675, so it was refused. MiniLM gives modest similarities for a short question against a long paragraph (often 0.3-0.6, so distances of
0.4-0.7), while unrelated text is usually far above 0.8. 0.80 is a starting value based on that one observed case and typical MiniLM behaviour; it has
**not** been measured on your documents, so check it (below). A looser gate is safe because Groq must still answer `found: false` when the text does not
answer the question, and the backend rejects any answer without a valid citation.

Limits of any distance threshold: it measures similarity, not "this chunk answers the question". It can pass an off-topic chunk that shares
vocabulary, and it can reject a valid but very short question (one word like "RAG" or "langchain" has little meaning to compare, so it often fails).

**How to tune it for your documents.** Every question logs all retrieved candidates in the backend console, for example:

```
Retrieval for 'What are the main objectives of this project?': 5 candidates (limit 0.800)
  #1 dist=0.6751 KEEP | brief.pdf p.1 chunk=0 | Project Overview Build a small web-based assistant ...
  #2 dist=0.7420 KEEP | brief.pdf p.3 chunk=7 | ...
```

When a question is refused, the chat also shows "Closest passage distance: X (passages must be within Y)". Ask 5 questions you know are answered
and 5 that are not. Set `MAX_DISTANCE` a little above the largest distance of the answered ones, and check the unanswered ones stay above it.
Restart the backend after changing `.env`. To check the default with the real model on sample text: `$env:RUN_REAL_EMBEDDING_TESTS="1"; python -m pytest tests\test_retrieval.py -s`
(it prints a table of real distances).

**Settings names.** `.env` names must match exactly (`CHUNK_SIZE`, not `HUNK_SIZE`). Lines starting with `#` are comments. A misspelt name is ignored,
and the backend now logs a warning such as `Unknown setting 'HUNK_SIZE' ... Did you mean CHUNK_SIZE?`.

## LLM provider and usage limits

Groq is used through the Groq API. Usage limits and pricing vary by model and account. Check the Groq Console for current limits and data-handling terms for your account and model.
Each question makes one Groq call (none when the evidence check refuses). A 429 error means you hit a limit; the app shows a clear message.


## Known limitations

- Page numbers are physical PDF pages (1 = first page), which can differ from page numbers printed on the page.
- Docling includes OCR/layout processing, but scanned pages, unusual fonts and complex tables should still be spot-checked; OCR quality depends on the PDF and language.
- Follow-ups: a short question (6 words or fewer, like "Does it have to be unused?") is also searched together with the previous user question.
  A very short unrelated message right after a related question can therefore pull in the old topic's passages; the refusal then depends on
  Groq following the prompt. Longer questions are searched on their own, so the distance check fully applies to them.
- Conflict handling depends on Groq following the prompt. The code guarantees both conflicting passages are sent and that only real sources are cited,
  but it cannot guarantee the explanation is good.
- Excerpts are the quote Groq chose if it really appears in the chunk, otherwise the most relevant sentences of the chunk.
- Duplicate detection compares file content. A renamed copy is a duplicate; an edited PDF is a new document, so delete the old version if you don't want both.
- Single user, no login, one backend process (ChromaDB's local mode).
