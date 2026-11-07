import pandas as pd

from app.ingestion import build_document, clean, ingest, split


def test_clean_removes_html_and_spaces():
    assert clean("<p>Hola&amp;   mundo</p>") == "Hola& mundo"
    assert clean(None) == ""


def test_document_has_title_brand_color(catalog_df):
    header, body = build_document(catalog_df.iloc[0])
    assert "Sony" in header and "Negro" in header
    assert "<p>" not in body


def test_long_description_is_split_and_header_repeated(catalog_df):
    row = catalog_df.iloc[0].copy()
    row["product_description"] = "palabra " * 600
    chunks = split(row)
    assert len(chunks) > 1
    assert all("Sony" in c.page_content for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "P1:0"
    assert set(chunks[0].metadata) >= {"product_id", "title", "brand", "color", "chunk_index"}


def test_ingest_is_idempotent(store, catalog_df):
    n = len(store.get()["ids"])
    assert n == len(catalog_df)
    assert ingest(catalog_df, store) == 0
    assert len(store.get()["ids"]) == n


def test_ingest_resumes(store, catalog_df):
    store.delete(ids=["P3:0", "P4:0"])
    assert ingest(catalog_df, store) == 2
    assert pd.Series(store.get()["ids"]).nunique() == len(catalog_df)
