"""All settings live here. Nothing else in the code reads environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GENERATOR_LLM = os.getenv("GENERATOR_LLM", "gpt-4o-mini")
EVALUATOR_LLM = os.getenv("EVALUATOR_LLM", "gpt-4o-mini")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
DATA_DIR = _path("DATA_DIR", "data")
VECTOR_STORE_DIR = _path("VECTOR_STORE_DIR", "data/chroma")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.2"))
RERANK = os.getenv("RERANK", "false").lower() == "true"
OPIK_URL_OVERRIDE = os.getenv("OPIK_URL_OVERRIDE", "")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "rag-ecommerce")

# Fail early: answering from fewer than 5 products is not allowed.
if RETRIEVAL_K < 5:
    raise ValueError("RETRIEVAL_K must be 5 or more")
