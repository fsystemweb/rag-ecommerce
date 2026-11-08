# Evaluation report: Qwen as generator

Run of **2026-11-08**, Opik experiment `hybrid-k10-Qwen/Qwen3-30B-A3B-Instruct-2507`. Same code and same setup as the [report after the fixes](report-after-fixes.md) (hybrid search, k=10 products in the prompt, 5 recommendations, 120 golden items, catalog of 2,000 products). The only change is the **generator model**: `Qwen/Qwen3-30B-A3B-Instruct-2507` instead of `deepseek-ai/DeepSeek-V4-Flash-0731`. The golden dataset was not touched.

> The golden dataset is still a **draft**. Read the numbers as a comparison between runs, not as a final grade.

## Results

| Metric | First report | After the fixes (DeepSeek) | This run (Qwen) | Change vs after the fixes |
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

Each cell shows after the fixes (DeepSeek) → this run (Qwen).

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

## Speed and cost

The speed gain comes from the generator: Qwen3-30B-A3B is a smaller, non-reasoning model, so it answers faster than DeepSeek. The run did not measure per-step time or token usage, so the cost is not in this report. To check it, open the experiment in Opik and look at the token usage of the `generate` spans.

## Caveats

- The output does not print the classifier model. This run changed the generator only, according to the experiment name. Check `CLASSIFIER_LLM` in your `.env` if you need to be sure the classifier matches the previous run.
- The experiment name says `k10`, which matches the `RETRIEVAL_K` default in `src/core/config.py`. The retrieval numbers match the previous run, so the retrieval settings look the same, but the `.env` was not read to confirm it.

## Next steps

1. Open the 5 false refusals (exact 2, broad 2, typo 1) in Opik and see where each one stops.
2. Open a few `brand` and `exact` items in Opik and compare the DeepSeek and Qwen recommendations. Find out whether Qwen picks the right brand but the wrong product, or the wrong brand.
3. Decide the default generator. Qwen is about twice as fast with lower product scores; DeepSeek has better product scores but is slower. Compare token cost in Opik before choosing.
