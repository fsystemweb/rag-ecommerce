"""Read the product catalog from data/catalog.parquet."""

import pandas as pd

from src.core import config


def load_catalog(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "catalog.parquet")
    return df.head(limit) if limit else df
