# Evaluation

## The golden dataset

`src/evaluation/datasets/golden_es.csv`: 120 items with an input, expected products (ESCI label E), acceptable products (S) and a reference answer.

| Type | Items | Notes |
|---|---|---|
| exact / attribute / broad / brand | 78 | real ESCI queries |
| typo | 6 | ESCI queries with swapped letters |
| follow_up | 6 | "¿Y en negro?" with history |
| unanswerable | 18 | price, stock, delivery, returns, other topics |
| adversarial | 12 | prompt injection, system-prompt leak |

**Review status: draft, not yet reviewed by a human** (see `src/evaluation/datasets/README.md`). Never edit it to improve a score.

## Metrics in plain words

Example: 5 results, the user wanted products A and B, we returned `[A, x, B, y, z]`.

| Metric | Meaning | Example |
|---|---|---|
| precision@k | share of the top k that is relevant | 2/5 = 0.4 |
| recall@k | share of the relevant products found in top k | 2/2 = 1.0 |
| refusal accuracy | refuses exactly when the item is unanswerable | 0 or 1 |

## Results

Last full run (120 items, hybrid search, k=5, generator `deepseek-ai/DeepSeek-V4-Flash-0731`, classifier `nvidia/Nemotron-3_5-Lightning`):

| precision@5 | recall@5 | recall@10 | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|---|
| 0.39 | 0.55 | 0.73 | 0.60 | 0.44 | 0.81 |

The full report, with charts, a description of every metric, token usage and recommendations, is in [Evaluation report](report.md).

## Known limits

ESCI labels are noisy. The catalog is a sample, and the golden dataset is a draft.
