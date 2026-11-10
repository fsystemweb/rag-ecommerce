# Evaluation report

All the runs below use the same **120 golden items**, the full pipeline (not retrieval only), hybrid search (dense + BM25) and a catalog of 2,000 products. Each experiment changes one part of the system, so the runs can be compared. The newest experiment is first.

> The golden dataset is still a **draft** (see [Evaluation](evaluation.md)). Read the results as a baseline to improve on, not as a final grade. The golden set was never edited to improve a score.

## Experiments at a glance

| Experiment | Opik experiment | What changed | recall@5 | product_recall | refusal_accuracy |
|---|---|---|---|---|---|
| [3: Qwen as generator](#experiment-3-qwen-as-generator) (latest) | `hybrid-k10-Qwen/Qwen3-30B-A3B-Instruct-2507` | Generator is `Qwen/Qwen3-30B-A3B-Instruct-2507` (was DeepSeek) | 0.547 | 0.529 | 0.950 |
| [2: After the fixes](#experiment-2-after-the-fixes) | `hybrid-k10-deepseek-ai/DeepSeek-V4-Flash-0731` | Non-reasoning classifier, usage capture, looser prompts, k=10 with 5 recommendations | 0.547 | 0.624 | 0.925 |
| [1: Baseline](#experiment-1-baseline) | `hybrid-k5-deepseek-ai/DeepSeek-V4-Flash-0731` | First run, k=5 | 0.547 | 0.442 | 0.808 |

Experiment 1 is the baseline. Experiment 2 applies the first recommendations of experiment 1. Experiment 3 keeps the setup of experiment 2 and only swaps the generator.

## What each metric means

Example: the user wanted products A and B and the system returned `[A, x, B, y, z]`.

| Metric | Question it answers | How it is computed | Example |
|---|---|---|---|
| **precision@5** | Of the 5 products retrieved, how many are relevant? | relevant in top 5 / 5 | 2/5 = 0.40 |
| **recall@5** | Of the products the user wanted, how many are in the top 5? | relevant in top 5 / all relevant | 2/2 = 1.00 |
| **precision@10 / recall@10** | Same with the top 10: shows if the right product is found but ranked low | | |
| **product_precision** | Of the products the assistant **recommended in its answer**, how many are right? | right recommended / recommended | |
| **product_recall** | Of the right products, how many did the answer **recommend**? | right recommended / all right | |
| **refusal_accuracy** | Does it refuse exactly when it should? | 1 if (refused == question is unanswerable) | 0 or 1 |

*Relevant* = ESCI label E (exact) from the golden set. The first four measure the **retriever**, the next two the **whole answer** (retriever + generator), the last one the **guardrails**. Precision is low by design: many questions have only 1 or 2 right products, so precision@5 cannot be higher than about 0.2 to 0.4.

---

## Experiment 3: Qwen as generator

Run of **2026-11-08**, Opik experiment `hybrid-k10-Qwen/Qwen3-30B-A3B-Instruct-2507`. Same code and same setup as [experiment 2](#experiment-2-after-the-fixes) (hybrid search, k=10 products in the prompt, 5 recommendations, 120 golden items, catalog of 2,000 products). The only change is the **generator model**: `Qwen/Qwen3-30B-A3B-Instruct-2507` instead of `deepseek-ai/DeepSeek-V4-Flash-0731`.

### Results

| Metric | Experiment 1 | Experiment 2 (DeepSeek) | Experiment 3 (Qwen) | Change vs experiment 2 |
|---|---|---|---|---|
| precision@5 | 0.389 | 0.389 | 0.389 | same |
| recall@5 | 0.547 | 0.547 | 0.547 | same |
| precision@10 | 0.263 | 0.263 | 0.263 | same |
| recall@10 | 0.725 | 0.725 | 0.725 | same |
| product_precision | 0.597 | 0.542 | **0.433** | -0.109 |
| product_recall | 0.442 | 0.624 | **0.529** | -0.095 |
| refusal_accuracy | 0.808 | 0.925 | **0.950** | +0.025 |

- **Retrieval numbers are identical**, as expected. The generator does not change search.
- **Product metrics are lower than with DeepSeek.** Both product precision and product recall fell, so Qwen is not trading one for the other: it recommends more wrong products and also finds fewer of the expected ones.
- **Refusals are better.** 114 of 120 refusal decisions are correct (0.950). Answerable questions refused by mistake went from 8 to **5** (exact 2, broad 2, typo 1). One unanswerable question was answered, the same as before. All 12 adversarial questions were refused.
- **The run is about twice as fast.** Total time was **19 min 35 s** against 36 min 19 s for DeepSeek, an average of about **9.8 s per question** against about 18 s (the average includes the instant refusals).

### By question type

Each cell shows experiment 2 (DeepSeek) → experiment 3 (Qwen).

| Type | n | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|
| exact | 34 | 0.487 → 0.338 | 0.603 → 0.428 | 0.882 → 0.941 |
| attribute | 15 | 0.573 → 0.547 | 0.697 → 0.630 | 1.000 → 1.000 |
| brand | 15 | 0.698 → 0.493 | 0.740 → 0.656 | 1.000 → 1.000 |
| broad | 14 | 0.584 → 0.519 | 0.564 → 0.505 | 0.786 → 0.857 |
| typo | 6 | 0.428 → 0.433 | 0.611 → 0.611 | 0.833 → 0.833 |
| follow_up | 6 | 0.400 → 0.333 | 0.422 → 0.506 | 1.000 → 1.000 |
| unanswerable | 18 | - | - | 0.944 → 0.944 |
| adversarial | 12 | - | - | 1.000 → 1.000 |

The retrieval columns are left out because they did not change.

### How to read it

- **The product drop is largest on `brand` and `exact`.** Brand product precision fell from 0.698 to 0.493 and exact from 0.487 to 0.338. These are the types where the right answer is a specific product, so a generator that recommends looser matches loses the most.
- **The refusal gain is small.** It is 3 more correct decisions out of 120, and the per-type groups are small. Do not over-read it.
- **`follow_up` and `typo` have 6 items each.** One item moves a number by about 0.17.
- **The "30 failed" in the summary** are the 18 unanswerable and 12 adversarial items. They have no expected products, which is why their product and retrieval columns show `-` in the table.

### Speed and cost

The speed gain comes from the generator: Qwen3-30B-A3B is a smaller, non-reasoning model, so it answers faster than DeepSeek. The run did not measure per-step time or token usage, so the cost is not in this report. To check it, open the experiment in Opik and look at the token usage of the `generate` spans.

### Caveats

- The output does not print the classifier model. This run changed the generator only, according to the experiment name. Check `CLASSIFIER_LLM` in your `.env` if you need to be sure the classifier matches the previous run.
- The experiment name says `k10`, which matches the `RETRIEVAL_K` default in `src/core/config.py`. The retrieval numbers match the previous run, so the retrieval settings look the same, but the `.env` was not read to confirm it.

### Next steps

1. Open the 5 false refusals (exact 2, broad 2, typo 1) in Opik and see where each one stops.
2. Open a few `brand` and `exact` items in Opik and compare the DeepSeek and Qwen recommendations. Find out whether Qwen picks the right brand but the wrong product, or the wrong brand.
3. Decide the default generator. Qwen is about twice as fast with lower product scores; DeepSeek has better product scores but is slower. Compare token cost in Opik before choosing.

---

## Experiment 2: after the fixes

Run of **2026-11-08**, Opik experiment `hybrid-k10-deepseek-ai/DeepSeek-V4-Flash-0731`. Same setup as [experiment 1](#experiment-1-baseline) (full pipeline, 120 golden items, hybrid search, catalog of 2,000 products, same generator) with the changes below.

### What changed

| Issue | Change |
|---|---|
| #40 | The topic classifier uses `CLASSIFIER_LLM`, a small **non-reasoning** model (`Qwen/Qwen3-30B-A3B-Instruct-2507`), instead of a reasoning model. |
| #41 | `stream_usage=True` in `make_llm`, so the generator reports its token usage. |
| #42 | Classifier prompt: when unsure, answer `PRODUCT_SEARCH`; budget or quality searches count as product searches. Generator prompt: reply `NO_ANSWER` only if nothing retrieved is related; otherwise recommend the closest products and say what does not match. `RETRIEVAL_MIN_SCORE` stays at 0.2. |
| #43 | `RETRIEVAL_K` 5 → **10** products in the prompt, and the answer ends with the **5** best recommendations. |

### Results

| Metric | Experiment 1 | Experiment 2 | Change |
|---|---|---|---|
| precision@5 | 0.389 | 0.389 | same |
| recall@5 | 0.547 | 0.547 | same |
| precision@10 | 0.263 | 0.263 | same |
| recall@10 | 0.725 | 0.725 | same |
| product_precision | 0.597 | 0.542 | -0.055 |
| product_recall | 0.442 | **0.624** | +0.182 |
| refusal_accuracy | 0.808 | **0.925** | +0.117 |

- **Retrieval numbers are identical.** The evaluator always measures retrieval over the top 10 with the same retriever, and nothing in the retriever changed. These fixes act on the classifier and the generator, not on search.
- **False refusals dropped from 23 to 8** of the 90 answerable questions (26% → 9%). 9 of 120 refusal decisions are now wrong (0.925): 8 answerable questions still refused and 1 unanswerable question answered.
- **The assistant finds more of the right products, but is less precise.** Recommending up to 5 products raises product_recall (0.44 → 0.62) and lowers product_precision (0.60 → 0.54): more recommendations means more chances to include a wrong one. This is the expected trade-off of asking for 5.

### By question type

| Type | n | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|
| exact | 34 | 0.599 → 0.487 | 0.410 → 0.603 | 0.676 → 0.882 |
| attribute | 15 | 0.624 → 0.573 | 0.566 → 0.697 | 0.800 → 1.000 |
| brand | 15 | 0.778 → 0.698 | 0.487 → 0.740 | 0.933 → 1.000 |
| broad | 14 | 0.550 → 0.584 | 0.402 → 0.564 | 0.714 → 0.786 |
| typo | 6 | 0.292 → 0.428 | 0.236 → 0.611 | 0.500 → 0.833 |
| follow_up | 6 | 0.489 → 0.400 | 0.506 → 0.422 | 0.833 → 1.000 |
| unanswerable | 18 | - | - | 1.000 → **0.944** |
| adversarial | 12 | - | - | 1.000 → 1.000 |

Each cell shows experiment 1 → experiment 2. The retrieval columns are left out because they did not change. The follow-up retrieval numbers are still an evaluator artifact, see [experiment 1](#experiment-1-baseline).

### How to read it

- **The guardrails are a bit looser.** One unanswerable question (1 of 18) is now answered instead of refused. This is the cost of a less strict classifier and generator prompt. The 12 adversarial questions are all still refused.
- **Refusals still left:** `exact` (4), `broad` (3) and `typo` (1). The next step is to open those 8 items in Opik and see where they stop.
- **Small groups move a lot.** `typo` and `follow_up` have 6 items each: one item changes a number by 0.17. Do not over-read them.

### Speed

The run took **36 min 19 s** for 120 questions, an average of about **18 s per question** (the questions run one after another and the average includes the instant refusals). Experiment 1 had a **median of 7.4 s** and p90 of 13.1 s. These are different statistics, but the pipeline is clearly **slower**, not faster. Likely causes: the reasoning generator thinks longer, it receives 10 products instead of 5, and it writes a longer answer with 5 recommendations. This was not measured per step. Two things follow:

- The LLM timeout in `make_llm` went from 30 s to 120 s: with 30 s the first attempt of this run crashed on a generation timeout.
- A non-reasoning generator (`--generator <model>`) is the first thing to try to get the time back. [Experiment 3](#experiment-3-qwen-as-generator) tried this.

### Not measured here

This experiment does not include token or cost numbers. To check #40 and #41 (the classifier no longer spending about 580 reasoning tokens per call, and the generator usage now present), open the experiment in Opik and look at the token usage of the `classify` and `generate` spans.

### Next steps

1. Open the 8 remaining false refusals and the 1 wrongly answered question in Opik and find where each one stops.
2. Try a non-reasoning generator to recover the speed, and compare both experiments in Opik. (Done in experiment 3.)
3. Rerank or enlarge the candidate set to lift recall@5 (0.55 versus 0.73 at 10), and handle typos before BM25, as in experiment 1.

---

## Experiment 1: baseline

Run of **2026-11-08**, Opik experiment `hybrid-k5-deepseek-ai/DeepSeek-V4-Flash-0731`. Full pipeline over the 120 golden items, hybrid search (dense + BM25), `k=5`, catalog of 2,000 products. The numbers are also saved in [`assets/eval-report.json`](assets/eval-report.json).

> **Update:** recommendations 1, 2, 3 and 6 below were applied in [experiment 2](#experiment-2-after-the-fixes).

### Models used

| Role | Model (Nebius) | Where |
|---|---|---|
| Generator and rewriter | `deepseek-ai/DeepSeek-V4-Flash-0731` | `GENERATOR_LLM` |
| Topic classifier | `nvidia/Nemotron-3_5-Lightning` | `EVALUATOR_LLM` |
| Embeddings | `BAAI/bge-multilingual-gemma2` (default) | `EMBEDDINGS_MODEL` |

The names come from the Opik experiment config and spans (the defaults in `.env.example` are different Llama models). Both chat models are **reasoning models**: they spend tokens "thinking" before they answer. This matters for cost and speed, see below.

### Results

![Overall metrics](assets/eval-overall.png)

| Metric | Value |
|---|---|
| precision@5 | 0.389 |
| recall@5 | 0.547 |
| precision@10 | 0.263 |
| recall@10 | 0.725 |
| product_precision | 0.597 |
| product_recall | 0.442 |
| refusal_accuracy | 0.808 |

Latency per question: p50 7.4 s, p90 13.1 s, p99 32.3 s.

### By question type

![By type](assets/eval-by-type.png)

| Type | n | recall@5 | recall@10 | product_precision | product_recall | refusal_accuracy |
|---|---|---|---|---|---|---|
| exact | 34 | 0.626 | 0.808 | 0.599 | 0.410 | 0.676 |
| attribute | 15 | 0.617 | 0.770 | 0.624 | 0.566 | 0.800 |
| brand | 15 | 0.602 | 0.867 | 0.778 | 0.487 | 0.933 |
| broad | 14 | 0.468 | 0.648 | 0.550 | 0.402 | 0.714 |
| typo | 6 | 0.514 | 0.611 | 0.292 | 0.236 | 0.500 |
| follow_up | 6 | 0.000* | 0.083* | 0.489 | 0.506 | 0.833 |
| unanswerable | 18 | - | - | - | - | **1.000** |
| adversarial | 12 | - | - | - | - | **1.000** |

\* The follow-up retrieval numbers are an **evaluator artifact**: with the full pipeline the evaluator measures retrieval on the raw text ("¿Y en negro?") without the rewrite step. The end-to-end columns (product_precision/recall) are the real ones for follow-ups.

### How to read it

- **Guardrails work**: all 18 unanswerable and all 12 adversarial questions were refused (1.00).
- **The weak point is false refusals.** refusal_accuracy is 0.81 because 23 of the 90 answerable questions (26%) were refused or answered "no match": 10 stopped before the generator was called, 13 reached it and still ended without a recommendation.
- **Right products are found but ranked low**: recall goes from 0.55 (top 5) to 0.73 (top 10). A reranker or a larger candidate set would recover part of that.
- **Precision of the answer (0.60) is better than its recall (0.44)**: when the assistant recommends something it is usually right, but it recommends too few products.
- **Typos and broad queries are the hardest**: the lexical part (BM25) cannot match misspelled words, and "broad" queries have many acceptable products that are not labeled E.

### Token usage and cost

Opik records the usage of each LLM call. Average per question over the 120 items, from the experiment:

| | Per question | 120 questions |
|---|---|---|
| Prompt tokens | 196 | 23.5k |
| Completion tokens | 591 | 71.0k |
| of which reasoning tokens | 585 | 70.2k |
| **Total** | **788** | **94.5k** |

By model (spans with usage recorded):

| Model | Calls | Prompt | Completion | Median call |
|---|---|---|---|---|
| Nemotron-3_5-Lightning (classifier) | 110 | 21.1k | 64.4k (63.8k reasoning) | 2.1 s |
| DeepSeek-V4-Flash-0731 (generator, rewriter) | 7 | 0.5k | 0.6k | 3.4 s |

**Two findings:**

1. **The classifier is the most expensive step and it only returns one label.** About 99% of its completion tokens are reasoning (about 580 per call) for an answer of a few tokens.
2. **The generator cost is not measured.** The generator streams its answer and the stream does not report usage, so 86 of its 93 calls have no token data. The total above is therefore a **lower bound**. See the recommendations.

Indicative price: the classifier is about 0.0001 USD per question (about 0.13 USD per 1,000 questions) using 0.05 USD / 1M input and 0.20 USD / 1M output tokens. **These prices come from third-party price trackers, not from Nebius**: check the Nebius price page and set your own. The whole run of 120 questions cost a few cents; embedding the 2,000 products is cheaper (`EMBEDDINGS_PRICE_PER_1M`, shown before ingesting).

### Recommendations

Ordered by expected gain per effort. Change one thing at a time and re-run the evaluation; never edit the golden set to improve a score. Recommendations 1, 2, 3 and 6 were applied in [experiment 2](#experiment-2-after-the-fixes).

1. **Turn off thinking in the classifier (or use a non-reasoning model).** Saves about 580 tokens and about 2 s per question, with no quality loss for a one-word label. The Llama 3.1 8B in `.env.example` is enough for this step.
2. **Capture the generator usage.** Pass `stream_usage=True` to `ChatOpenAI` in `src/core/llm.py`. Without it the cost report is incomplete.
3. **Reduce false refusals.** Open the 23 refused answerable items in Opik and check where they stopped: classifier, `RETRIEVAL_MIN_SCORE` (0.2) or the generator. Lowering the threshold or making the classifier prompt less strict are the first things to try.
4. **Rerank.** Retrieve 20 candidates and rerank them (cross-encoder or a small LLM call) before passing 5 to the generator. Targets recall@5 (0.55 versus 0.73 at 10).
5. **Handle typos.** Add a spelling-correction step (or use the rewriter) before BM25. Typos score 0.51 recall@5 and 0.24 product_recall.
6. **Let the answer recommend more.** product_recall (0.44) trails product_precision (0.60): ask the generator for up to 3 products when several are relevant.
7. **Evaluate follow-ups end to end.** Make the evaluator run the rewrite before measuring retrieval, so the follow-up rows are meaningful. (Done: retrieval is now scored on the pipeline's own search.)
8. **Compare models.** Run `python -m src.evaluation.evaluator --generator <model>` with a non-reasoning generator and compare the experiments in Opik: quality, latency (p50 7.4 s now) and tokens side by side. (Done in [experiment 3](#experiment-3-qwen-as-generator).)
9. **Review the golden dataset** (labels are ESCI, answers are templates) before drawing final conclusions.
10. **Cache** repeated questions and the classifier result to avoid paying twice.
