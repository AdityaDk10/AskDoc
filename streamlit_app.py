from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

import requests
import streamlit as st

API_URL = os.getenv("ASKDOC_API", "http://localhost:8000")


st.set_page_config(page_title="AskDoc – RAG", layout="wide")
st.title("AskDoc – AI Research Assistant")

with st.sidebar:
	st.header("Ingest PDFs")
	uploaded = st.file_uploader("Upload PDF", type=["pdf"])
	if uploaded is not None:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
			tmp.write(uploaded.getbuffer())
			tmp.flush()
			with open(tmp.name, "rb") as f:
				files = {"file": (uploaded.name, f, "application/pdf")}
				r = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
				if r.ok:
					st.success(f"Ingested {r.json().get('ingested_chunks')} chunks from {uploaded.name}")
				else:
					st.error(f"Ingest failed: {r.text}")

st.divider()

col_q, col_ctx = st.columns([2, 1])

with col_q:
	q = st.text_input("Ask a question about your papers")
	if st.button("Ask") and q.strip():
		resp = requests.post(f"{API_URL}/query", json={"question": q}, timeout=120)
		if resp.ok:
			data: Dict[str, Any] = resp.json()
			st.subheader("Answer")
			st.write(data.get("answer", ""))
			st.subheader("Citations")
			for i, c in enumerate(data.get("contexts", []), 1):
				st.markdown(f"**[{i}] {c.get('source')}** – score: {round(c.get('score') or 0, 3)}")
				with st.expander("Show snippet"):
					st.write(c.get("text", ""))
		else:
			st.error(f"Query failed: {resp.text}")

with col_ctx:
	st.caption("Upload PDFs in the sidebar, then ask domain questions here. Answers are grounded in retrieved chunks with citations.")


