from __future__ import annotations

from typing import Dict, List

from .config import settings
from .embeddings import GeminiEmbeddings
from .llm import GeminiLLM
from .vectorstore import DocumentChunk, FaissStore


def retrieve(query: str, top_k: int | None = None) -> List[DocumentChunk]:
	embedder = GeminiEmbeddings()
	q_vec = embedder.embed_query(query)
	store = FaissStore(settings.index_dir)
	return store.search(q_vec, top_k or settings.top_k)


def answer_question(question: str, top_k: int | None = None) -> Dict:
	contexts = retrieve(question, top_k)
	llm = GeminiLLM()
	answer = llm.generate(question, contexts)
	return {
		"answer": answer,
		"contexts": [
			{
				"id": c.id,
				"source": c.source,
				"page_start": c.page_start,
				"page_end": c.page_end,
				"score": c.score,
				"text": c.text,
			}
			for c in contexts
		],
	}


