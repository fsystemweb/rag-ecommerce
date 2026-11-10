import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.evaluation import evaluator as run
from src.evaluation.dataset import GOLDEN_CSV, load_golden, sync
from src.evaluation.test_metrics import *  # noqa: F401,F403  (run the metric checks under pytest too)

HEADER = "id,input,expected_product_ids,acceptable_product_ids,expected_output,answerable,question_type,source,history\n"


def write(tmp_path, body):
    p = tmp_path / "g.csv"
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_real_golden_file_is_valid_and_big_enough():
    items = load_golden()
    assert len(items) >= 100
    assert {"unanswerable", "adversarial", "typo"} <= {i["question_type"] for i in items}


def test_validation_catches_problems(tmp_path):
    p = write(tmp_path, "a,hola,,,x,true,exact,esci,\na,hola,P1,,x,true,exact,esci,\n")
    with pytest.raises(ValueError) as e:
        load_golden(p)
    assert "duplicated id" in str(e.value) and "no expected_product_ids" in str(e.value)


def test_validation_checks_catalog(tmp_path):
    p = write(tmp_path, "a,hola,P1,,x,true,exact,esci,\n")
    with pytest.raises(ValueError, match="not in catalog"):
        load_golden(p, catalog_ids={"P9"})
    assert load_golden(p, catalog_ids={"P1"})[0]["expected_product_ids"] == ["P1"]


class FakeRetriever:
    def retrieve(self, query, k=5, where=None):
        class C:
            text = "Auriculares Sony"
            metadata = {"product_id": "P1", "title": "Auriculares Sony"}

        return [C()]


def test_run_item_retrieval_only():
    item = {"answerable": True, "expected_product_ids": ["P1"], "acceptable_product_ids": [], "history": [], "input": "x"}
    scores = run.run_item(item, FakeRetriever(), None, retrieval_only=True)
    assert scores["precision@5"] == 1 / 5 and scores["recall@5"] == 1.0


def test_unanswerable_items_have_no_retrieval_scores():
    item = {"answerable": False, "expected_product_ids": [], "acceptable_product_ids": [], "history": [], "input": "x"}
    assert run.run_item(item, FakeRetriever(), None, retrieval_only=True) == {}


def test_aggregate_and_floor():
    table = run.aggregate([("exact", {"recall@5": 1.0}), ("brand", {"recall@5": 0.0})])
    assert table["ALL"]["recall@5"] == 0.5 and table["exact"]["recall@5"] == 1.0
    assert run.check_floors(table, ["recall@5=0.4"]) == []
    assert run.check_floors(table, ["recall@5=0.9"])


def test_golden_csv_path_exists():
    assert GOLDEN_CSV.exists()


class FakeDataset:
    """Stands in for an Opik dataset: rows have an Opik "id" and our "item_id"."""

    def __init__(self, rows):
        self.rows = rows
        self.deleted = []
        self.inserted = []

    def get_items(self):
        return list(self.rows)

    def delete(self, ids):
        self.deleted.extend(ids)
        self.rows = [r for r in self.rows if r["id"] not in ids]

    def insert(self, items):
        self.inserted.extend(items)


def test_sync_deletes_items_removed_from_csv_and_inserts_new_ones():
    remote = FakeDataset([{"id": "u1", "item_id": "q1", "input": "a"}, {"id": "u2", "item_id": "f001", "input": "x"}])
    deleted, inserted = sync(remote, [{"id": "q1", "input": "a"}, {"id": "q2", "input": "b"}])
    assert (deleted, inserted) == (1, 1)
    assert remote.deleted == ["u2"]
    assert [i["item_id"] for i in remote.inserted] == ["q2"]


def test_sync_replaces_items_edited_under_the_same_id():
    remote = FakeDataset([{"id": "u1", "item_id": "a002", "input": "old text"}])
    deleted, inserted = sync(remote, [{"id": "a002", "input": "new text"}])
    assert (deleted, inserted) == (1, 1)
    assert remote.deleted == ["u1"]
    assert remote.inserted[0]["item_id"] == "a002" and remote.inserted[0]["input"] == "new text"


def test_sync_is_a_no_op_when_already_in_sync():
    remote = FakeDataset([{"id": "u1", "item_id": "q1", "input": "a"}])
    assert sync(remote, [{"id": "q1", "input": "a"}]) == (0, 0)
    assert remote.deleted == [] and remote.inserted == []


def test_check_item_reports_missing_fields():
    from src.evaluation.evaluator import check_item

    with pytest.raises(ValueError, match="x1.*answerable"):
        check_item({"item_id": "x1", "input": "q", "question_type": "exact", "expected_product_ids": []})
    ok = check_item({"input": "q", "answerable": False, "question_type": "exact", "expected_product_ids": []})
    assert ok["history"] == []


# --- full mode: one search per item, scored on what the pipeline retrieved -------------------------------------------


class CountingEmbeddings(DeterministicFakeEmbedding):
    queries: list = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


def reply(*texts):
    return GenericFakeChatModel(messages=iter([AIMessage(content=t) for t in texts]))


def full_item(answerable=True, expected=("P1",)):
    return {"input": "auriculares sony", "answerable": answerable, "expected_product_ids": list(expected), "history": []}


def test_full_mode_searches_once_and_scores_the_pipeline_chunks(tmp_path, monkeypatch, catalog_df):
    from src.agent.rag_agent import Pipeline
    from src.agent.retriever import Retriever
    from src.ingestion.index import get_store, ingest

    monkeypatch.setattr("src.core.config.VECTOR_STORE_DIR", tmp_path / "chroma")
    monkeypatch.setattr("src.core.config.RETRIEVAL_MIN_SCORE", -1.0)  # fake embeddings are random: do not gate
    monkeypatch.setattr("src.core.config.BM25_FIELDS", "full")
    emb = CountingEmbeddings(size=64)
    store = get_store(emb)
    ingest(catalog_df, store)
    emb.queries.clear()
    retriever = Retriever(store)

    def make_pipeline(rec):
        return Pipeline(rec, reply("auriculares sony", "Te recomiendo [P1] Sony"), reply("PRODUCT_SEARCH"))

    out = {}
    scores = run.run_item(full_item(), retriever, make_pipeline, False, out, k=10)
    assert len(emb.queries) == 1  # only the pipeline's own search
    assert scores["recall@10"] == 1.0 and "precision@10" in scores
    assert scores["product_precision"] == 1.0 and scores["blocked_answerable"] == 0.0
    assert out["answer"].startswith("Te recomiendo")


def test_k_below_10_reports_no_at_10_metrics():
    from src.agent.rag_agent import Pipeline

    def make_pipeline(rec):
        return Pipeline(rec, reply("x", "[P1]"), reply("PRODUCT_SEARCH"))

    scores = run.run_item(full_item(), FakeRetriever(), make_pipeline, False, k=5)
    assert "recall@5" in scores and "recall@10" not in scores and "precision@10" not in scores


def test_blocked_answerable_item_is_left_out_of_retrieval_metrics():
    from src.agent.rag_agent import Pipeline

    def make_pipeline(rec):
        return Pipeline(rec, reply("x"), reply("OFF_TOPIC"))

    scores = run.run_item(full_item(), FakeRetriever(), make_pipeline, False, k=10)
    assert scores == {"refusal_accuracy": 0.0, "blocked_answerable": 1.0}


def test_refused_answerable_item_counts_in_refusal_accuracy_not_precision():
    from src.agent.rag_agent import Pipeline

    def make_pipeline(rec):
        return Pipeline(rec, reply("x", "NO_ANSWER"), reply("PRODUCT_SEARCH"))

    scores = run.run_item(full_item(), FakeRetriever(), make_pipeline, False, k=10)
    assert scores["refusal_accuracy"] == 0.0
    assert not {"precision@5", "precision@10", "product_precision", "product_recall"} & scores.keys()


def test_recording_retriever_keeps_the_chunks_and_query():
    rec = run.RecordingRetriever(FakeRetriever())
    chunks = rec.retrieve("hola", k=5)
    assert rec.calls == [("hola", chunks)]


def test_llm_timeout_comes_from_config(monkeypatch):
    from src.core import llm

    monkeypatch.setattr("src.core.config.LLM_TIMEOUT", 7.0)
    monkeypatch.setattr("src.core.config.NEBIUS_API_KEY", "test")  # CI has no .env
    assert llm.make_llm("m").request_timeout == 7.0
