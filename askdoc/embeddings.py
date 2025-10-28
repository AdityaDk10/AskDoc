from __future__ import annotations

from typing import List

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings


class GeminiEmbeddings:
    def __init__(self, model: str | None = None) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=settings.gemini_api_key)
        name = model or settings.embed_model
        if not (name.startswith("models/") or name.startswith("tunedModels/")):
            name = f"models/{name}"
        self.model_name = name

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
    def _embed_one(self, text: str) -> List[float]:
        resp = genai.embed_content(model=self.model_name, content=text)
        vec = resp.get("embedding")
        if isinstance(vec, dict) and "values" in vec:
            return vec["values"]
        if isinstance(vec, list):
            return vec
        vals = resp.get("values")
        if vals:
            return vals
        raise RuntimeError("Unexpected embedding response structure from Gemini")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            vectors.append(self._embed_one(t))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_documents([text])
        return vectors[0]


