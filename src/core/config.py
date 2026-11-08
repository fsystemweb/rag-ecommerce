"""All settings live here. Nothing else in the code reads environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_API = os.getenv("NEBIUS_API", "https://api.studio.nebius.com/v1/")
GENERATOR_LLM = os.getenv("GENERATOR_LLM", "meta-llama/Llama-3.3-70B-Instruct")
# Topic classifier: a one-word answer, so use a small non-reasoning model (a reasoning model wastes ~580 tokens and ~2 s)
CLASSIFIER_LLM = os.getenv("CLASSIFIER_LLM", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-multilingual-gemma2")
EMBEDDINGS_PRICE_PER_1M = float(os.getenv("EMBEDDINGS_PRICE_PER_1M", "0.01"))  # USD, check Nebius pricing
DATA_DIR = _path("DATA_DIR", "data")
VECTOR_STORE_DIR = _path("VECTOR_STORE_DIR", "data/chroma")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.2"))
OPIK_URL_OVERRIDE = os.getenv("OPIK_URL_OVERRIDE", "")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "rag-ecommerce")

# Fail early: answering from fewer than 5 products is not allowed.
if RETRIEVAL_K < 5:
    raise ValueError("RETRIEVAL_K must be 5 or more")
