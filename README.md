## AskDoc – AI Research Assistant (RAG)

AskDoc is a Retrieval-Augmented Generation (RAG) app that ingests academic PDFs, indexes them with FAISS, and answers questions with grounded citations using Google Gemini.

### Features
- Upload PDFs, extract text, chunk, embed via Gemini, index in FAISS
- Ask questions; retrieves top-k chunks and generates an answer with citations
- Transparent results: show sources, snippets, and similarity scores

### Stack
- Backend: FastAPI + Uvicorn
- RAG: FAISS + Gemini `text-embedding-004` for embeddings; `gemini-1.5-flash` for generation
- Parsing: PyPDF
- UI: Streamlit

### Setup
1) Python 3.10+
2) Create and activate a virtual environment
```
python -m venv .venv
./.venv/Scripts/activate
```
3) Install dependencies
```
pip install -r requirements.txt
```
4) Environment
Create a `.env` file:
```
GEMINI_API_KEY=YOUR_KEY
DATA_DIR=data
INDEX_DIR=index
EMBED_MODEL=text-embedding-004
GEN_MODEL=gemini-1.5-flash
TOP_K=5
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

### Run API
```
uvicorn askdoc.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run UI
```
streamlit run streamlit_app.py
```

### API Endpoints
- POST `/ingest` (multipart/form-data): `file` PDF upload
- POST `/query`: `{ "question": "...", "top_k": 5 }`

### Notes
- Indices are stored in `index/`; uploads preserved in `data/`
- This MVP stores metadata in `index/meta.jsonl`
- For local-only mode, you can swap Gemini with local embeddings/LLM later


