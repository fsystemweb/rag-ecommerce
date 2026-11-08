# Data

**Amazon Shopping Queries Dataset (ESCI)**, Spanish (`es`) locale. License: Apache 2.0. Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data).

| File | Columns |
|---|---|
| products | `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale` |
| examples | `query`, `product_id`, `esci_label`, `split`, `small_version`, `product_locale` |

Labels: **E** exact (what the user wants), **S** substitute (acceptable alternative), **C** complement (goes with it), **I** irrelevant.

## Build the sample

```bash
python -m scripts.download_data                 # ~1 GB, once
python -m scripts.sample_catalog --size 2000   # same seed gives the same catalog
```

The sampler keeps only Spanish products with a title, always includes the E/S products of the golden-dataset queries, and fills the rest at random. Output: `data/catalog.parquet` and `data/queries.parquet`. **Nothing from the dataset is committed.**


## Golden dataset (versioned)

The only dataset in the repo is the evaluation set: [`src/evaluation/datasets/golden_es.csv`](https://github.com/fsystemweb/rag-ecommerce/blob/main/src/evaluation/datasets/golden_es.csv) (120 items), with its [README](https://github.com/fsystemweb/rag-ecommerce/blob/main/src/evaluation/datasets/README.md) and the list of ESCI queries it uses. The Amazon data and the vector store are generated locally and never committed.

## Ingestion: embeddings in batches

`python -m src.ingestion` does not embed the whole catalog in one request. It sends the chunks in **batches of 100** (`BATCH_SIZE` in `src/ingestion/index.py`) so a single call never exceeds the provider limits and Nebius does not answer **429 (too many requests)**. If a batch still fails (rate limit or network), it is retried up to 5 times, waiting 1, 2, 4 and 8 seconds. Finished products are skipped on the next run, so an interrupted ingestion just continues.
