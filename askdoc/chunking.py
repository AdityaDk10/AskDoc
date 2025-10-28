from __future__ import annotations

from typing import Iterable, List


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
	if chunk_size <= 0:
		raise ValueError("chunk_size must be positive")
	if overlap < 0:
		raise ValueError("overlap cannot be negative")
	if overlap >= chunk_size:
		raise ValueError("overlap must be smaller than chunk_size")

	words: List[str] = text.split()
	if not words:
		return []

	chunks: List[str] = []
	start: int = 0
	step: int = max(1, chunk_size - overlap)
	while start < len(words):
		end = min(len(words), start + chunk_size)
		chunks.append(" ".join(words[start:end]))
		start += step
	return chunks


def batched(iterable: Iterable, n: int) -> Iterable[list]:
	batch: list = []
	for item in iterable:
		batch.append(item)
		if len(batch) == n:
			yield batch
			batch = []
	if batch:
		yield batch


