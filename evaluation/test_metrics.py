"""Offline checks with hand-made examples. Run: python -m evaluation.test_metrics (or pytest)."""

import math

from evaluation.metrics import (
    GAINS,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
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
    assert hit_rate_at_k([], REL, 5) == 0.0


def test_duplicates_count_once():
    assert precision_at_k(["a", "a", "a"], REL, 3) == 1 / 3


def test_hit_rate_and_mrr():
    assert hit_rate_at_k(["x", "b"], REL, 5) == 1.0
    assert mrr(["x", "y", "b"], REL) == 1 / 3
    assert mrr(["x", "y"], REL) == 0.0


def test_ndcg_perfect_and_worse():
    gains = {"a": GAINS["E"], "b": GAINS["S"]}
    assert ndcg_at_k(["a", "b"], gains, 2) == 1.0
    worse = ndcg_at_k(["b", "a"], gains, 2)
    assert 0 < worse < 1
    assert math.isclose(ndcg_at_k(["x"], gains, 2), 0.0)
    assert ndcg_at_k(["a"], {}, 5) == 0.0  # nothing labeled


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
