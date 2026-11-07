import pandas as pd

from src.ingestion.chunker import build_document, clean, to_chunk
from src.ingestion.index import ingest


def test_clean_removes_html_and_spaces():
    assert clean("<p>Hola&amp;   mundo</p>") == "Hola& mundo"
    assert clean(None) == ""


def test_document_has_title_brand_color(catalog_df):
    header, body = build_document(catalog_df.iloc[0])
    assert "Sony" in header and "Negro" in header
    assert "<p>" not in body


def test_one_product_is_one_chunk(catalog_df):
    row = catalog_df.iloc[0].copy()
    row["product_description"] = "palabra " * 600
    chunk = to_chunk(row)
    assert "Sony" in chunk.page_content
    assert chunk.id == "P1"
    assert set(chunk.metadata) == {"product_id", "title", "brand", "color"}


def test_ingest_is_idempotent(store, catalog_df):
    n = len(store.get()["ids"])
    assert n == len(catalog_df)
    assert ingest(catalog_df, store) == 0
    assert len(store.get()["ids"]) == n


def test_ingest_resumes(store, catalog_df):
    store.delete(ids=["P3", "P4"])
    assert ingest(catalog_df, store) == 2
    assert pd.Series(store.get()["ids"]).nunique() == len(catalog_df)
