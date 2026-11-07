"""Clean product text and split each product into chunks."""

import html
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# One product usually fits in one chunk. ~1000 chars is about 250 tokens: small
# enough to stay focused, big enough to keep a description together.
# Overlap 100 keeps a sentence cut at the border readable in both chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TEXT_CHARS = 3000  # very long descriptions are noise; cut them


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
