# Golden dataset (`golden_es_v2.csv`)

117 items, Spanish. Fixed reference for every evaluation run. `golden_es.csv` is v1 (120 items), kept to compare old runs.

This file is the **dataset_final_117** version, which replaced the earlier 111-item v2 draft. It is the same Opik dataset, `rag-ecommerce-golden-v2`. In a later review, 5 items (`q007`, `q025`, `q068`, `q070`, `q076`) had products moved from `acceptable_product_ids` to `expected_product_ids`, and the reviewer metadata columns were dropped from the CSV. Compared with the draft, 12 items were added (`q085`-`q092`, `u019`-`u022`), 6 adversarial items were removed (`a007`-`a012`), and the inputs of 5 adversarial items (`a002`-`a006`) were rewritten under the same ids.

Mix: 90 answerable (`exact` 33, `attribute` 15, `brand` 15, `broad` 13, `typo` 14) and 27 unanswerable (21 `unanswerable`, 6 `adversarial`). There are no `follow_up` items.

| Column | Meaning |
|---|---|
| `id` | Unique item id (`q` ESCI query, `u` unanswerable, `a` adversarial) |
| `input` | The user question |
| `expected_product_ids` | ESCI label **E** products, separated by `\|` |
| `acceptable_product_ids` | ESCI label **S** products (also a good answer), separated by `\|` |
| `expected_output` | Reference answer. Answerable: a template with the product titles and ids. Others: the fixed refusal |
| `answerable` | `true` / `false` |
| `question_type` | `exact`, `attribute`, `broad`, `brand`, `typo`, `unanswerable`, `adversarial` (and `follow_up`, if a future version has one) |
| `source` | `esci` (query as in ESCI), `manual` (hand-written) or `derived` (made from another item) |
| `history` | JSON list of previous turns (follow-ups only) |

## Rules

- Add an item: append a row, run `python -m src.evaluation.dataset` (validates ids against the catalog).
- Sync Opik after editing the CSV: `python -m src.evaluation.dataset --upload`. It deletes Opik items that were removed from the CSV or whose content changed, and inserts the new and changed ones, so the evaluator's item ids match the file.
- Never edit the set to make a score better. Changes go in a PR with the reason.
- Never edit the set to make a score better. Changes go in a PR with the reason.
