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
python -m scripts.sample_catalog --size 10000   # same seed gives the same catalog
```

The sampler keeps only Spanish products with a title, always includes the E/S products of the golden-dataset queries, and fills the rest at random. Output: `data/catalog.parquet` and `data/queries.parquet`. **Nothing from the dataset is committed.**
