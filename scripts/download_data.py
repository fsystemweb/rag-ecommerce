"""Download the Amazon ESCI parquet files into DATA_DIR.

Usage: python -m scripts.download_data
"""

import requests

from src.core.config import DATA_DIR

BASE = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
FILES = [
    "shopping_queries_dataset_products.parquet",
    "shopping_queries_dataset_examples.parquet",
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Source: github.com/amazon-science/esci-data | License: Apache 2.0")
    print("The products file is large (about 1 GB, all locales).")
    for name in FILES:
        target = DATA_DIR / name
        if target.exists():
            print(f"skip (exists): {target}")
            continue
        print(f"downloading {name} ...")
        with requests.get(BASE + name, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        print(f"  saved {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
