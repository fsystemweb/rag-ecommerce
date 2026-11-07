"""Offline checks with hand-made examples. Run: python -m evaluation.test_metrics (or pytest)."""

from evaluation.metrics import (
    precision_at_k,
    product_precision,
    product_recall,
    recall_at_k,
    refusal_accuracy,
)

REL = {"a", "b"}


def test_precision_recall():
    got = ["a", "x", "b", "y", "z"]
    assert precision_at_k(got, REL, 5) == 2 / 5
    assert recall_at_k(got, REL, 5) == 1.0
    assert recall_at_k(got, REL, 1) == 0.5


def test_fewer_than_k_results_and_no_relevant():
    assert precision_at_k(["a"], REL, 5) == 1 / 5  # missing slots count as misses
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_duplicates_count_once():
    assert precision_at_k(["a", "a", "a"], REL, 3) == 1 / 3


def test_product_metrics():
    assert product_precision(["a", "x"], REL) == 0.5
    assert product_precision([], REL) == 0.0
    assert product_recall(["a"], REL) == 0.5


def test_refusal_accuracy():
    assert refusal_accuracy(True, False) == 1.0
    assert refusal_accuracy(False, False) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all metric checks passed")
