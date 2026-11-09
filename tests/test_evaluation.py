import pytest

from src.evaluation import evaluator as run
from src.evaluation.dataset import GOLDEN_CSV, load_golden
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
    def retrieve(self, query, k=5):
        class C:
            metadata = {"product_id": "P1"}

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
