"""The four evaluation metrics. Plain functions first, the Opik wrapper at the end.

- recall@10          does the search find the products? (retrieved ids, ESCI label E)
- product_precision  are the recommended products right? (answer ids, label E or S)
- product_recall     did the answer include the right products? (answer ids, label E)
- refusal_accuracy   does it refuse exactly when it should?
"""

NAMES = ["recall@10", "product_precision", "product_recall", "refusal_accuracy"]


def recall_at_k(retrieved, expected, k=10):
    """Share of the expected products found in the first k retrieved (a product counts once)."""
    if not expected:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    return sum(p in expected for p in top) / len(expected)


def product_precision(answer_ids, good):
    """Share of the recommended products that are good (expected or acceptable)."""
    answer_ids = set(answer_ids)
    return sum(p in good for p in answer_ids) / len(answer_ids) if answer_ids else 0.0


def product_recall(answer_ids, expected):
    """Share of the expected products that the answer recommends."""
    return sum(p in expected for p in set(answer_ids)) / len(expected) if expected else 0.0


def refusal_accuracy(refused: bool, answerable: bool) -> float:
    return float(refused != answerable)


def opik_metrics(names=NAMES):
    """Wrap precomputed scores as Opik metrics (the task returns a dict of scores)."""
    from opik.evaluation.metrics import base_metric, score_result

    def make(name):
        class Metric(base_metric.BaseMetric):
            def __init__(self):
                super().__init__(name=name)

            def score(self, **task_output):
                value = task_output.get(name)
                if value is None:
                    return score_result.ScoreResult(name=name, value=0.0, scoring_failed=True)
                return score_result.ScoreResult(name=name, value=float(value))

        return Metric()

    return [make(n) for n in names]
