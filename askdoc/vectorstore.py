from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np


@dataclass
class DocumentChunk:
	id: str
	text: str
	source: str
	page_start: int | None = None
	page_end: int | None = None
	score: float | None = None


class FaissStore:
	def __init__(self, index_dir: str) -> None:
		self.index_dir = index_dir
		self.index_path = os.path.join(index_dir, "faiss.index")
		self.meta_path = os.path.join(index_dir, "meta.jsonl")
		self.index: faiss.Index | None = None

	def _ensure_index(self, dim: int) -> None:
		if self.index is None:
			self.index = faiss.IndexFlatIP(dim)

	def _load_index(self) -> None:
		if os.path.exists(self.index_path):
			self.index = faiss.read_index(self.index_path)

	def add(self, embeddings: List[List[float]], metadatas: List[DocumentChunk]) -> None:
		vecs = np.array(embeddings, dtype=np.float32)
		# Normalize for cosine via inner product
		norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
		vecs = vecs / norms
		dim = vecs.shape[1]
		self._load_index()
		self._ensure_index(dim)
		self.index.add(vecs)
		with open(self.meta_path, "a", encoding="utf-8") as f:
			for m in metadatas:
				f.write(json.dumps(m.__dict__, ensure_ascii=False) + "\n")
		faiss.write_index(self.index, self.index_path)

	def search(self, query_vector: List[float], top_k: int) -> List[DocumentChunk]:
		self._load_index()
		if self.index is None:
			return []
		q = np.array([query_vector], dtype=np.float32)
		q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
		dists, idxs = self.index.search(q, top_k)
		results: List[DocumentChunk] = []
		# Load meta lines once
		if not os.path.exists(self.meta_path):
			return []
		with open(self.meta_path, "r", encoding="utf-8") as f:
			meta_lines = [json.loads(line) for line in f if line.strip()]
		for pos, score in zip(idxs[0], dists[0]):
			if pos < 0 or pos >= len(meta_lines):
				continue
			obj = meta_lines[pos]
			results.append(DocumentChunk(
				id=obj.get("id", str(pos)),
				text=obj.get("text", ""),
				source=obj.get("source", ""),
				page_start=obj.get("page_start"),
				page_end=obj.get("page_end"),
				score=float(score),
			))
		return results


