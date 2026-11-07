"""Indexing: chunks -> embeddings -> Chroma (the catalog and chunking live in loader.py and chunker.py).

Run: python -m src.ingestion [--limit N] [--yes]

Safe to re-run: chunk ids are the product ids, and ids already in
the store are skipped, so an interrupted run just continues.
"""

import argparse
import time

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.core import config
from src.core.llm import make_embeddings
from src.ingestion.chunker import to_chunk
from src.ingestion.loader import load_catalog

BATCH_SIZE = 100


def get_store(embeddings=None) -> Chroma:
    """Open (or create) the persistent vector store. Cosine distance."""
    return Chroma(
        collection_name="products",
        embedding_function=embeddings or make_embeddings(),
        persist_directory=str(config.VECTOR_STORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_with_retry(store: Chroma, docs: list[Document], tries: int = 5):
    for attempt in range(tries):
        try:
            store.add_documents(docs, ids=[d.id for d in docs])
            return
        except Exception as e:  # rate limit or network: wait and retry
            if attempt == tries - 1:
                raise
            wait = 2**attempt
            print(f"  error ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)


def ingest(df: pd.DataFrame, store: Chroma, batch_size: int = BATCH_SIZE) -> int:
    """Embed and store every chunk that is not in the store yet. Returns new chunks."""
    chunks = [to_chunk(row) for _, row in df.iterrows()]
    done = set(store.get(include=[])["ids"])
    todo = [c for c in chunks if c.id not in done]
    for i in range(0, len(todo), batch_size):
        add_with_retry(store, todo[i : i + batch_size])
        print(f"  stored {min(i + batch_size, len(todo))}/{len(todo)}")
    return len(todo)


def estimate_cost(chunks: list[Document]) -> tuple[int, float]:
    tokens = sum(len(c.page_content) for c in chunks) // 4  # rough: 4 chars per token
    return tokens, tokens / 1e6 * config.EMBEDDINGS_PRICE_PER_1M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    df = load_catalog(args.limit)
    chunks = [to_chunk(row) for _, row in df.iterrows()]
    print(f"products: {len(df)}  chunks: {len(chunks)} ")
    print("sample:", chunks[0].page_content[:200], chunks[0].metadata)

    store = get_store()
    pending = len(chunks) - len(set(store.get(include=[])["ids"]) & {c.id for c in chunks})
    tokens, usd = estimate_cost(chunks)
    print(f"pending chunks: {pending}  (full run ~{tokens:,} tokens, ~${usd:.2f})")
    if pending and not args.yes and input("Embed with Nebius (costs money)? [y/N] ").lower() != "y":
        return
    print(f"new chunks stored: {ingest(df, store)}")


if __name__ == "__main__":
    main()
