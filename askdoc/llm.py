from __future__ import annotations

from typing import List

import google.generativeai as genai

from .config import settings
from .vectorstore import DocumentChunk


SYSTEM_PROMPT = (
	"You are AskDoc, an AI research assistant. Answer the question using only the provided context. "
	"Cite sources in square brackets like [1], [2] mapping to the provided context list. "
	"If the answer is not contained in the context, say you don't know and suggest relevant sections."
)


class GeminiLLM:
	def __init__(self, model: str | None = None) -> None:
		if not settings.gemini_api_key:
			raise RuntimeError("GEMINI_API_KEY is not set")
		genai.configure(api_key=settings.gemini_api_key)
        name = model or settings.gen_model
        if not (name.startswith("models/") or name.startswith("tunedModels/")):
            name = f"models/{name}"
        self.model_name = name
        self.model = genai.GenerativeModel(self.model_name)

	def compose_prompt(self, question: str, contexts: List[DocumentChunk]) -> str:
		lines: List[str] = [SYSTEM_PROMPT, "", "Context:"]
		for i, c in enumerate(contexts, 1):
			lines.append(f"[{i}] {c.text}\n(Source: {c.source}, pages {c.page_start}-{c.page_end})")
		lines.append("")
		lines.append(f"Question: {question}")
		lines.append("Answer:")
		return "\n".join(lines)

	def generate(self, question: str, contexts: List[DocumentChunk]) -> str:
		prompt = self.compose_prompt(question, contexts)
		resp = self.model.generate_content(prompt)
		return resp.text or ""


