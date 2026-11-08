# Evaluation report: after the fixes

Run of **2026-11-08**, Opik experiment `hybrid-k10-deepseek-ai/DeepSeek-V4-Flash-0731`. Same setup as the [first report](report.md) (full pipeline, 120 golden items, hybrid search, catalog of 2,000 products, same generator) with the changes below. The golden dataset was not touched.

> The golden dataset is still a **draft**. Read the numbers as a comparison between two runs, not as a final grade.

## What changed

| Issue | Change |
|---|---|
| #40 | The topic classifier uses `CLASSIFIER_LLM`, a small **non-reasoning** model (`Qwen/Qwen3-30B-A3B-Instruct-2507`), instead of a reasoning model. |
| #41 | `stream_usage=True` in `make_llm`, so the generator reports its token usage. |
| #42 | Classifier prompt: when unsure, answer `PRODUCT_SEARCH`; budget or quality searches count as product searches. Generator prompt: reply `NO_ANSWER` only if nothing retrieved is related; otherwise recommend the closest products and say what does not match. `RETRIEVAL_MIN_SCORE` stays at 0.2. |
| #43 | `RETRIEVAL_K` 5 → **10** products in the prompt, and the answer ends with the **5** best recommendations. |

## Results

| Metric | Before | After | Change |
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

The retrieval columns are left out because they did not change. The follow-up retrieval numbers are still an evaluator artifact, see the [first report](report.md).

### How to read it

- **The guardrails are a bit looser.** One unanswerable question (1 of 18) is now answered instead of refused. This is the cost of a less strict classifier and generator prompt. The 12 adversarial questions are all still refused.
- **Refusals still left:** `exact` (4), `broad` (3) and `typo` (1). The next step is to open those 8 items in Opik and see where they stop.
- **Small groups move a lot.** `typo` and `follow_up` have 6 items each: one item changes a number by 0.17. Do not over-read them.

## Speed

The run took **36 min 19 s** for 120 questions, an average of about **18 s per question** (the questions run one after another and the average includes the instant refusals). The first report had a **median of 7.4 s** and p90 of 13.1 s. These are different statistics, but the pipeline is clearly **slower**, not faster. Likely causes: the reasoning generator thinks longer, it receives 10 products instead of 5, and it writes a longer answer with 5 recommendations. This was not measured per step. Two things follow:

- The LLM timeout in `make_llm` went from 30 s to 120 s: with 30 s the first attempt of this run crashed on a generation timeout.
- A non-reasoning generator (`--generator <model>`) is the first thing to try to get the time back.

## Not measured here

This report does not include token or cost numbers. To check #40 and #41 (the classifier no longer spending about 580 reasoning tokens per call, and the generator usage now present), open the experiment in Opik and look at the token usage of the `classify` and `generate` spans.

## Next steps

1. Open the 8 remaining false refusals and the 1 wrongly answered question in Opik and find where each one stops.
2. Try a non-reasoning generator to recover the speed, and compare both experiments in Opik.
3. Rerank or enlarge the candidate set to lift recall@5 (0.55 versus 0.73 at 10), and handle typos before BM25, as in the first report.
