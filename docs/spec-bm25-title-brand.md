# Spec: BM25 on title and brand (hybrid retrieval)

**Status:** Implemented behind `BM25_FIELDS` (default `full`). Evaluator runs pending.

## Goal

Evaluate whether BM25 retrieval performs better when indexing only the short product fields (`title` and `brand`) instead of the full product text (`title`, `brand`, `description`, and bullets).

Dense retrieval continues to use the full product text.

The BM25 index must use only `title` and `brand`.

## Scope

- Only the BM25 index changes. The embedding model, Chroma store, chunker, and prompts remain unchanged.
- No re-ingestion is required. The Chroma metadata already contains `title` and `brand`, allowing BM25 to be built directly from metadata without triggering the paid embedding step.
- No changes to RRF, `POOL`, `RETRIEVAL_K`, or `RETRIEVAL_MIN_SCORE`.

## Proposed change

In `src/agent/retriever.py`, update `Retriever._load_bm25` to build the BM25 corpus from a short field string per product:

```python
bm25_text = f"{title} {brand}"  # from metadata
```

Use the existing `tokenize()` implementation without modifications.

**Important considerations:**

- **Title is essential.** The offline check showed that the product noun is typically present in the title. Using brand alone would remove a significant part of the matching signal.
  The proposed experiment evaluates `title + brand` only.
- **Keep the implementation configurable.** Read the field selection from `config.py`, for example, `BM25_FIELDS=full|short`, with `full` as the default. This allows switching between both strategies using a single environment variable and makes rollback straightforward.

## Evaluation

Run retrieval only, avoiding generation and classifier costs. The only expected paid operations are query embeddings.

```bash
python -m src.evaluation.evaluator --retrieval-only --k 10
```

Execute two runs using the same golden set:

1. `BM25_FIELDS=full` — baseline.
2. `BM25_FIELDS=short` — title + brand only.

Evaluate only the **84 answerable, non-follow-up items**. Follow-up retrieval metrics are excluded because the evaluator currently uses the raw input rather than the rewritten query.

| Metric | Acceptance criteria |
|---|---|
| recall@5 | Must not decrease by more than 0.02 |
| recall@10 | Must not decrease |
| Items with zero correct products in top 10 | Must not increase |

The offline check suggested a small improvement with title + brand. However, the expected difference is only a few percentage points, so this experiment should be treated as an initial screening rather than conclusive evidence.

If the variant passes these criteria, a subsequent full evaluation will be required to assess its impact on product metrics and refusals.

## Acceptance criteria

- Baseline results are consistent with previous retrieval measurements.
- The proposed `title + brand` configuration meets all three retrieval acceptance criteria.
- If the criteria pass, schedule a separate full evaluation to validate product metrics and refusals before changing the default configuration.

## Risks and considerations

- **Attribute-based queries:** Queries such as `"sin bomba"` or `"manga larga"` may depend on information available only in descriptions or bullets. Restricting BM25 to title and brand could reduce recall for these queries. The per-item evaluation should help identify regressions.
- **Hybrid retrieval masking:** Dense retrieval continues to use the full product text and may compensate for changes in BM25, making the overall hybrid retrieval impact smaller than the BM25-only impact.
- **Missing metadata:** Some products may have missing brand values. Empty fields are acceptable because the title is always available.
