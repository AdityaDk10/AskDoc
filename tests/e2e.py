import requests


API = "http://127.0.0.1:8001"


def main() -> None:
	# Status
	r = requests.get(f"{API}/status", timeout=30)
	print("STATUS:", r.status_code, r.text)

	# Ingest sample.pdf
	with open("sample.pdf", "rb") as f:
		files = {"file": ("sample.pdf", f, "application/pdf")}
		r = requests.post(f"{API}/ingest", files=files, timeout=300)
		print("INGEST:", r.status_code, r.text)

	# Query
	body = {"question": "What are transformers and attention mechanisms?", "top_k": 5}
	r = requests.post(f"{API}/query", json=body, timeout=120)
	print("QUERY:", r.status_code)
	print(r.text)


if __name__ == "__main__":
	main()


