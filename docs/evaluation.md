# Evaluation

## The golden dataset

`src/evaluation/datasets/golden_es_v2.csv`: 117 items with an input, expected products (ESCI label E), acceptable products (S) and a reference answer.

[DEPRECATED] - `src/evaluation/datasets/golden_es.csv`: 120 items with an input, expected products (ESCI label E), acceptable products (S) and a reference answer.


| Type | Items | Notes |
|---|---|---|
| exact / attribute / broad / brand | 76 | real ESCI queries |
| typo | 14 | ESCI queries with swapped letters |
| unanswerable | 21 | price, stock, delivery, returns, other topics |
| adversarial | 6 | prompt injection, system-prompt leak |

**Review status: draft, not yet reviewed by a human** (see `src/evaluation/datasets/README.md`). Never edit it to improve a score.

## Metrics in plain words

Example: 5 results, the user wanted products A and B, we returned `[A, x, B, y, z]`.

| Metric | Meaning | Example |
|---|---|---|
| precision@k | share of the top k that is relevant | 2/5 = 0.4 |
| recall@k | share of the relevant products found in top k | 2/2 = 1.0 |
| refusal accuracy | refuses exactly when the item is unanswerable | 0 or 1 |

### How items are scored

`python -m src.evaluation.evaluator --threads 4` runs 4 items in parallel. In full mode the retrieval metrics are computed on the chunks the pipeline itself retrieved, with its rewritten query (no second search). `@10` metrics are reported only when `k >= 10`.

- An answerable item stopped before retrieval (input guardrail or topic classifier) is left out of recall and precision and counted in `blocked_answerable`.
- An answerable item that is refused (`REFUSAL` or `NO_MATCH`) is left out of `precision@k`, `product_precision` and `product_recall`; it counts in `refusal_accuracy`.

Because of these rules, results are not directly comparable with runs made before this change.

## Results

Last full run (120 items, hybrid search, k=5, generator `deepseek-ai/DeepSeek-V4-Flash-0731`, classifier `nvidia/Nemotron-3_5-Lightning`):

| precision@5 | recall@5 | recall@10 | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|---|
| 0.39 | 0.55 | 0.73 | 0.60 | 0.44 | 0.81 |

After the fixes (classifier model, usage capture, fewer refusals, k=10 with 5 recommendations) refusal_accuracy went from 0.81 to 0.93 and product_recall from 0.44 to 0.62, see [experiment 2 in the evaluation report](report.md#experiment-2-after-the-fixes).

The full evaluation report, with every experiment, charts, a description of every metric, token usage and recommendations, is in [Evaluation report](report.md).

## Known limits

ESCI labels are noisy. The catalog is a sample, and the golden dataset is a draft.
