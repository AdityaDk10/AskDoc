from __future__ import annotations

import os
import uuid
from typing import List

from PyPDF2 import PdfReader

from .chunking import sliding_window_chunks
from .config import settings
from .embeddings import GeminiEmbeddings
from .vectorstore import DocumentChunk, FaissStore


def extract_pdf_text(path: str) -> List[str]:
	reader = PdfReader(path)
	pages: List[str] = []
	for p in reader.pages:
		text = p.extract_text() or ""
		pages.append(text)
	return pages


def ingest_pdf(path: str) -> int:
	pages = extract_pdf_text(path)
	full_text = "\n\n".join(pages)
	chunks = sliding_window_chunks(full_text, settings.chunk_size, settings.chunk_overlap)
	metas: List[DocumentChunk] = []
	for i, ch in enumerate(chunks):
		metas.append(DocumentChunk(
			id=str(uuid.uuid4()),
			text=ch,
			source=os.path.basename(path),
			page_start=None,
			page_end=None,
		))
	embedder = GeminiEmbeddings()
	embs = embedder.embed_documents([m.text for m in metas])
	store = FaissStore(settings.index_dir)
	store.add(embs, metas)
	return len(chunks)


