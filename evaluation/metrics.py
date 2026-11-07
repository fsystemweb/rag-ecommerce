"""Retrieval and answer metrics. Plain functions first, Opik wrappers at the end.

`retrieved` is a list of product ids in rank order (a product counts once).
`relevant` is the set of expected product ids (ESCI label E).
"""

import math

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}  # nDCG gains by ESCI label


def dedupe(ids):
    return list(dict.fromkeys(ids))


def precision_at_k(retrieved, relevant, k):
    top = dedupe(retrieved)[:k]
    return sum(p in relevant for p in top) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return sum(p in relevant for p in dedupe(retrieved)[:k]) / len(relevant)


def hit_rate_at_k(retrieved, relevant, k):
    return float(any(p in relevant for p in dedupe(retrieved)[:k]))


def mrr(retrieved, relevant):
    for rank, p in enumerate(dedupe(retrieved), start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, gains, k):
    """gains: {product_id: gain}, e.g. E=1.0, S=0.1, C=0.01 (unknown = 0)."""
    top = dedupe(retrieved)[:k]
    dcg = sum(gains.get(p, 0.0) / math.log2(i + 2) for i, p in enumerate(top))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def product_precision(answer_ids, expected):
    answer_ids = dedupe(answer_ids)
    return sum(p in expected for p in answer_ids) / len(answer_ids) if answer_ids else 0.0


def product_recall(answer_ids, expected):
    return sum(p in expected for p in set(answer_ids)) / len(expected) if expected else 0.0


def refusal_accuracy(refused: bool, answerable: bool) -> float:
    return float(refused != answerable)


def retrieval_scores(retrieved, relevant, gains, ks=(5, 10)):
    out = {"mrr": mrr(retrieved, relevant)}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gains, k)
    return out


def opik_metrics(names):
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
