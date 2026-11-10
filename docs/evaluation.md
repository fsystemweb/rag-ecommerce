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

Four metrics, one question each. *Expected* = ESCI label E, *acceptable* = label S.

| Metric | Question | How |
|---|---|---|
| recall@10 | Does the search find the products? | expected products in the top 10 retrieved / all expected |
| product_precision | Are the recommended products right? | recommended that are expected or acceptable / all recommended |
| product_recall | Did the answer include the right products? | expected products recommended / all expected |
| refusal_accuracy | Does it refuse correctly? | 1 if it refuses exactly when the item is unanswerable, else 0 |

Example: the user wanted A and B, the search returned `[A, x, B, y, z]` and the answer recommended A and S (an acceptable product): recall@10 = 2/2 = 1.0, product_precision = 2/2 = 1.0, product_recall = 1/2 = 0.5.

### How items are scored

`python -m src.evaluation.evaluator --threads 4` runs 4 items in parallel. In full mode recall@10 is computed on the chunks the pipeline itself retrieved, with its rewritten query (no second search). `--retrieval-only` reports only recall@10.

- Unanswerable items only get `refusal_accuracy`.
- An answerable item stopped before retrieval (input guardrail or topic classifier) only gets `refusal_accuracy`.
- An answerable item that is refused (`REFUSAL` or `NO_MATCH`) gets `refusal_accuracy` and `recall@10`, not the product metrics.

## Results

Last full run (120 items, hybrid search, k=5, generator `deepseek-ai/DeepSeek-V4-Flash-0731`, classifier `nvidia/Nemotron-3_5-Lightning`):

| precision@5 | recall@5 | recall@10 | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|---|
| 0.39 | 0.55 | 0.73 | 0.60 | 0.44 | 0.81 |

After the fixes (classifier model, usage capture, fewer refusals, k=10 with 5 recommendations) refusal_accuracy went from 0.81 to 0.93 and product_recall from 0.44 to 0.62, see [experiment 2 in the evaluation report](report.md#experiment-2-after-the-fixes).

The full evaluation report, with every experiment, charts, a description of every metric, token usage and recommendations, is in [Evaluation report](report.md).

## Known limits

ESCI labels are noisy. The catalog is a sample, and the golden dataset is a draft.
