"""Offline checks with hand-made examples. Run: python -m src.evaluation.test_metrics (or pytest)."""

from src.evaluation.metrics import product_precision, product_recall, recall_at_k, refusal_accuracy

REL = {"a", "b"}


def test_recall_at_k():
    got = ["a", "x", "b", "y", "z"]
    assert recall_at_k(got, REL) == 1.0
    assert recall_at_k(got, REL, k=1) == 0.5
    assert recall_at_k(["a", "a"], REL) == 0.5  # duplicates count once
    assert recall_at_k(["a"], set()) == 0.0


def test_product_precision_counts_expected_and_acceptable():
    good = REL | {"s"}  # "s" is an acceptable (ESCI label S) product
    assert product_precision(["a", "s"], good) == 1.0
    assert product_precision(["a", "x"], good) == 0.5
    assert product_precision([], good) == 0.0


def test_product_recall():
    assert product_recall(["a"], REL) == 0.5
    assert product_recall(["a", "a", "x"], REL) == 0.5


def test_refusal_accuracy():
    assert refusal_accuracy(True, False) == 1.0
    assert refusal_accuracy(False, False) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all metric checks passed")
