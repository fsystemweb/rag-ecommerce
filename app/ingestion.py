"""Ingestion: catalog -> product documents -> chunks -> embeddings -> Chroma.

Run: python -m app.ingestion [--limit N] [--yes]

Safe to re-run: chunk ids are "product_id:chunk_index", and ids already in
the store are skipped, so an interrupted run just continues.
"""

import argparse
import html
import re
import time

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

# One product usually fits in one chunk. ~1000 chars is about 250 tokens: small
# enough to stay focused, big enough to keep a description together.
# Overlap 100 keeps a sentence cut at the border readable in both chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TEXT_CHARS = 3000  # very long descriptions are noise; cut them
BATCH_SIZE = 100
PRICE_PER_1M_TOKENS = 0.02  # text-embedding-3-small, USD (check OpenAI pricing)


def load_catalog(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "catalog.parquet")
    return df.head(limit) if limit else df


def clean(text) -> str:
    """Remove HTML tags/entities and repeated spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def build_document(row) -> tuple[str, str]:
    """Return (header, body). The header is repeated in every chunk."""
    header = f"Título: {clean(row['product_title'])}"
    brand, color = clean(row["product_brand"]), clean(row["product_color"])
    if brand:
        header += f" | Marca: {brand}"
    if color:
        header += f" | Color: {color}"
    body = ""
    desc, bullets = clean(row["product_description"]), clean(row["product_bullet_point"])
    if desc:
        body += f"Descripción: {desc} "
    if bullets:
        body += f"Características: {bullets}"
    return header, body.strip()[:MAX_TEXT_CHARS]


_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def split(row) -> list[Document]:
    """One product -> one or more chunks, each with title/brand and metadata."""
    header, body = build_document(row)
    pieces = _splitter.split_text(body) if body else [""]
    meta = {
        "product_id": str(row["product_id"]),
        "title": clean(row["product_title"]),
        "brand": clean(row["product_brand"]),
        "color": clean(row["product_color"]),
    }
    return [
        Document(
            page_content=f"{header}\n{piece}".strip(),
            metadata={**meta, "chunk_index": i},
            id=f"{meta['product_id']}:{i}",
        )
        for i, piece in enumerate(pieces)
    ]


def get_store(embeddings=None) -> Chroma:
    """Open (or create) the persistent vector store. Cosine distance."""
    return Chroma(
        collection_name="products",
        embedding_function=embeddings or OpenAIEmbeddings(model=config.EMBEDDINGS_MODEL),
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
    chunks = [c for _, row in df.iterrows() for c in split(row)]
    done = set(store.get(include=[])["ids"])
    todo = [c for c in chunks if c.id not in done]
    for i in range(0, len(todo), batch_size):
        add_with_retry(store, todo[i : i + batch_size])
        print(f"  stored {min(i + batch_size, len(todo))}/{len(todo)}")
    return len(todo)


def estimate_cost(chunks: list[Document]) -> tuple[int, float]:
    tokens = sum(len(c.page_content) for c in chunks) // 4  # rough: 4 chars per token
    return tokens, tokens / 1e6 * PRICE_PER_1M_TOKENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    df = load_catalog(args.limit)
    chunks = [c for _, row in df.iterrows() for c in split(row)]
    print(f"products: {len(df)}  chunks: {len(chunks)}  ({len(chunks) / len(df):.2f} per product)")
    print("sample:", chunks[0].page_content[:200], chunks[0].metadata)

    store = get_store()
    pending = len(chunks) - len(set(store.get(include=[])["ids"]) & {c.id for c in chunks})
    tokens, usd = estimate_cost(chunks)
    print(f"pending chunks: {pending}  (full run ~{tokens:,} tokens, ~${usd:.2f})")
    if pending and not args.yes and input("Embed with OpenAI (costs money)? [y/N] ").lower() != "y":
        return
    print(f"new chunks stored: {ingest(df, store)}")


if __name__ == "__main__":
    main()
