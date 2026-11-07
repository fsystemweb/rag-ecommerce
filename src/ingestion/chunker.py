"""Clean product text and turn each product into one chunk."""

import html
import re

from langchain_core.documents import Document

# One product = one chunk. Very long descriptions are noise, so they are cut.
MAX_TEXT_CHARS = 3000


def clean(text) -> str:
    """Remove HTML tags/entities and repeated spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def build_document(row) -> tuple[str, str]:
    """Return (header, body). The header holds the title, brand and color."""
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


def to_chunk(row) -> Document:
    """One product -> one chunk, with title/brand and metadata."""
    header, body = build_document(row)
    product_id = str(row["product_id"])
    return Document(
        page_content=f"{header}\n{body}".strip(),
        metadata={
            "product_id": product_id,
            "title": clean(row["product_title"]),
            "brand": clean(row["product_brand"]),
            "color": clean(row["product_color"]),
        },
        id=product_id,
    )
