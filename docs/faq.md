# FAQ

**Why not put the whole catalog in the prompt?** 10,000 products are millions of tokens: too slow, costly and above model limits.

**Why a second model as judge?** The generator tends to trust its own text. A separate, cheap check against the source products catches invented details.

**Why at least 5 chunks?** One hit is fragile; five distinct products allow comparison and make recall measurable.

**How do I change a prompt?** Edit it in Opik and restart the API (source `opik`), or edit `app/prompts/*.txt`. Re-run the evaluation afterwards.

**How do I use my own catalog?** Produce `data/catalog.parquet` with columns `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, run `python -m app.ingestion`, and replace the golden dataset.
