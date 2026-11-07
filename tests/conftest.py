import pandas as pd
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.ingestion.index import get_store, ingest

PRODUCTS = [
    ("P1", "Auriculares inalámbricos Sony para correr", "Sony", "Negro"),
    ("P2", "Auriculares bluetooth JBL resistentes al agua", "JBL", "Azul"),
    ("P3", "Funda negra para iPhone 13", "Spigen", "Negro"),
    ("P4", "Funda transparente para iPhone 12", "Apple", ""),
    ("P5", "Botella de agua 500ml acero inoxidable", "Chilly", "Plata"),
    ("P6", "Camión de juguete de madera", "Hape", "Rojo"),
    ("P7", "Cargador USB-C rápido 20W", "Anker", "Blanco"),
    ("P8", "Ratón inalámbrico ergonómico", "Logitech", "Gris"),
    ("P9", "Teclado mecánico retroiluminado", "Corsair", "Negro"),
    ("P10", "Mochila impermeable para portátil", "Samsonite", "Azul"),
]


@pytest.fixture(autouse=True)
def fake_classifier(monkeypatch):
    """The pipeline classifies the topic with an LLM; offline every question is a product search."""
    ok = AIMessage(content="PRODUCT_SEARCH")
    monkeypatch.setattr("src.agent.rag_agent.make_llm", lambda *a, **k: GenericFakeChatModel(messages=iter([ok] * 50)))


@pytest.fixture
def catalog_df():
    return pd.DataFrame(
        [
            {
                "product_id": pid,
                "product_title": title,
                "product_description": f"<p>Descripción de {title}</p>",
                "product_bullet_point": "Muy buena calidad",
                "product_brand": brand,
                "product_color": color,
            }
            for pid, title, brand, color in PRODUCTS
        ]
    )


@pytest.fixture
def store(tmp_path, monkeypatch, catalog_df):
    """Real Chroma on a temp folder, with fake (deterministic) embeddings."""
    monkeypatch.setattr("src.core.config.VECTOR_STORE_DIR", tmp_path / "chroma")
    s = get_store(DeterministicFakeEmbedding(size=64))
    ingest(catalog_df, s)
    return s
