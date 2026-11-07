# Data

Source: [Amazon Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data), **Spanish (`es`) locale only**.
License: Apache 2.0 (the dataset is downloaded, never committed).

| File | Main columns |
|---|---|
| `shopping_queries_dataset_products.parquet` | `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale` |
| `shopping_queries_dataset_examples.parquet` | `query`, `product_id`, `esci_label`, `split`, `small_version`, `product_locale` |

Labels: **E** exact, **S** substitute, **C** complement, **I** irrelevant.

## Regenerate

```bash
python -m scripts.download_data                 # ~1 GB, once
python -m scripts.sample_catalog --size 10000   # same seed -> same catalog
```

Outputs `data/catalog.parquet` and `data/queries.parquet` (labels only for the sampled products).

## Known limits

- No prices and no store policies (returns, shipping). The assistant is a product *search and recommendation* assistant and refuses those questions.
- ESCI labels are human but noisy; the catalog is a sample, not the full store.
