"""Build a small, repeatable Spanish catalog from the ESCI dataset.

Usage: python -m scripts.sample_catalog --size 2000 [--seed 42]

Outputs data/catalog.parquet and data/queries.parquet.
Products labeled E or S for the golden queries are always kept
(src/evaluation/datasets/golden_queries.txt, one query per line), then the
rest is filled with random products so retrieval is not trivial.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.core.config import DATA_DIR, ROOT

GOLDEN_QUERIES = ROOT / "src" / "evaluation" / "datasets" / "golden_queries.txt"


def build(products, examples, size, seed, golden_queries):
    products = products[products["product_locale"] == "es"]
    products = products[products["product_title"].fillna("").str.strip() != ""]
    examples = examples[examples["product_locale"] == "es"]

    # 1. products that must be there (E or S for golden queries)
    must = examples[examples["query"].isin(golden_queries) & examples["esci_label"].isin(["E", "S"])]
    must_ids = set(must["product_id"]) & set(products["product_id"])

    # 2. fill with random products (fixed seed -> same catalog every time)
    rest = products[~products["product_id"].isin(must_ids)]
    n_fill = max(size - len(must_ids), 0)
    fill = rest.sample(n=min(n_fill, len(rest)), random_state=seed)

    catalog = pd.concat([products[products["product_id"].isin(must_ids)], fill])
    catalog = catalog.sort_values("product_id").reset_index(drop=True)
    queries = examples[examples["product_id"].isin(catalog["product_id"])].reset_index(drop=True)
    return catalog, queries


def report(catalog, queries):
    print(f"products: {len(catalog)}  queries: {queries['query'].nunique()}  labels: {len(queries)}")
    print("label distribution:", queries["esci_label"].value_counts().to_dict())
    length = (catalog["product_title"].fillna("") + catalog["product_description"].fillna("")).str.len()
    print("text length (chars):", length.describe()[["mean", "50%", "max"]].round(0).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    products = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_products.parquet")
    examples = pd.read_parquet(DATA_DIR / "shopping_queries_dataset_examples.parquet")
    golden = []
    if GOLDEN_QUERIES.exists():
        golden = [q.strip() for q in Path(GOLDEN_QUERIES).read_text().splitlines() if q.strip()]
    catalog, queries = build(products, examples, args.size, args.seed, golden)
    catalog.to_parquet(DATA_DIR / "catalog.parquet", index=False)
    queries.to_parquet(DATA_DIR / "queries.parquet", index=False)
    report(catalog, queries)


if __name__ == "__main__":
    main()
