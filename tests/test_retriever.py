import pytest

from src.agent.retriever import Retriever, rrf, tokenize
from src.core import config


def test_tokenize_folds_accents_and_keeps_units():
    assert tokenize("Camión 500ml a 5.95€") == ["camion", "500ml", "a", "5.95"]


def test_rrf_prefers_items_ranked_high_in_both():
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"] and scores["a"] > scores["d"]


def test_returns_at_least_5_distinct_products(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", -1.0)  # fake embeddings: scores are random
    res = Retriever(store).retrieve("auriculares", k=3)  # k below 5 is raised to 5
    ids = [c.metadata["product_id"] for c in res]
    assert len(ids) == 5 and len(set(ids)) == 5


def test_exact_term_found_by_bm25(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", -1.0)
    res = Retriever(store).retrieve("Anker", k=5)
    assert res[0].metadata["product_id"] == "P7"
    assert res[0].bm25_rank == 1


def test_accent_insensitive(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", -1.0)
    res = Retriever(store).retrieve("camion", k=5)
    assert res[0].metadata["product_id"] == "P6"


def test_min_score_returns_nothing(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", 2.0)  # cosine never reaches 2
    assert Retriever(store).retrieve("lo que sea") == []


def test_metadata_filter(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SCORE", -1.0)
    res = Retriever(store).retrieve("negro", k=5, where={"color": "Negro"})
    assert res and all(c.metadata["color"] == "Negro" for c in res)


def test_config_rejects_k_below_5(monkeypatch):
    import importlib

    monkeypatch.setenv("RETRIEVAL_K", "3")
    with pytest.raises(ValueError):
        importlib.reload(config)
    monkeypatch.setenv("RETRIEVAL_K", "5")
    importlib.reload(config)
