"""Configuration centralisée, surchargeable par variables d'environnement."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", ROOT_DIR / "chroma_db"))
    collection: str = os.getenv("CHROMA_COLLECTION", "langgraph_docs")
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_chars: int = 80


settings = Settings()
