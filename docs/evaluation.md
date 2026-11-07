# Evaluation

## The golden dataset

`evaluation/datasets/golden_es.csv`: 120 items with an input, expected products (ESCI label E), acceptable products (S) and a reference answer.

| Type | Items | Notes |
|---|---|---|
| exact / attribute / broad / brand | 78 | real ESCI queries |
| typo | 6 | ESCI queries with swapped letters |
| follow_up | 6 | "¿Y en negro?" with history |
| unanswerable | 18 | price, stock, delivery, returns, other topics |
| adversarial | 12 | prompt injection, system-prompt leak |

**Review status: draft, not yet reviewed by a human** (see `evaluation/datasets/README.md`). Never edit it to improve a score.

## Metrics in plain words

Example: 5 results, the user wanted products A and B, we returned `[A, x, B, y, z]`.

| Metric | Meaning | Example |
|---|---|---|
| precision@k | share of the top k that is relevant | 2/5 = 0.4 |
| recall@k | share of the relevant products found in top k | 2/2 = 1.0 |
| refusal accuracy | refuses exactly when the item is unanswerable | 0 or 1 |

## Results

Retrieval only, golden items that are answerable, 10,000-product catalog:

| Configuration | precision@5 | recall@5 | Status |
|---|---|---|---|
| BM25 only | 0.349 | 0.530 | **measured** (no API calls) |
| dense only | - | - | needs Nebius key: `--search dense --retrieval-only` |
| hybrid (default) | - | - | needs Nebius key: `--search hybrid --retrieval-only` |
| hybrid + rerank | - | - | needs Nebius key: `--rerank` |

Only BM25 could be run without an API key; the other rows are commands to fill in (the numbers are not invented). Generation metrics (product precision/recall, refusal accuracy) also need a key.

## Known limits

ESCI labels are noisy. The catalog is a sample, and the golden dataset is a draft.
