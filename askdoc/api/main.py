from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from ..config import settings
from ..ingest import ingest_pdf
from ..rag import answer_question


app = FastAPI(title="AskDoc RAG API")


@app.get("/status")
def status() -> Dict[str, Any]:
	return {
		"ok": True,
		"embed_model": settings.embed_model,
		"gen_model": settings.gen_model,
	}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
	try:
		filename = file.filename or "upload.pdf"
		path = os.path.join(settings.data_dir, filename)
		data = await file.read()
		with open(path, "wb") as f:
			f.write(data)
		count = ingest_pdf(path)
		return JSONResponse({"ingested_chunks": count, "file": filename})
	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/query")
async def query(body: Dict[str, Any]) -> JSONResponse:
	try:
		question = body.get("question", "").strip()
		if not question:
			return JSONResponse({"error": "question is required"}, status_code=400)
		top_k = body.get("top_k")
		res = answer_question(question, top_k)
		return JSONResponse(res)
	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)


