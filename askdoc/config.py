import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
	def __init__(self) -> None:
		self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
		self.data_dir: str = os.getenv("DATA_DIR", "data")
		self.index_dir: str = os.getenv("INDEX_DIR", "index")
		self.embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-004")
		self.gen_model: str = os.getenv("GEN_MODEL", "gemini-1.5-flash")
		self.top_k: int = int(os.getenv("TOP_K", "5"))
		self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
		self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

	def ensure_dirs(self) -> None:
		os.makedirs(self.data_dir, exist_ok=True)
		os.makedirs(self.index_dir, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


