"""Golden dataset loader with validation, and an idempotent upload to Opik.

Run: python -m src.evaluation.dataset [--upload]
"""

import csv
import json
import sys

import pandas as pd

from src.core.config import DATA_DIR, ROOT
from src.observability import tracer as observability

GOLDEN_CSV = ROOT / "src" / "evaluation" / "datasets" / "golden_es_v2.csv"
DATASET_NAME = "rag-ecommerce-golden-v2"
TYPES = {"exact", "attribute", "broad", "brand", "follow_up", "typo", "unanswerable", "adversarial"}


def _ids(value: str) -> list[str]:
    return [x for x in value.split("|") if x]


def load_golden(path=GOLDEN_CSV, catalog_ids: set[str] | None = None) -> list[dict]:
    """Read the CSV and check it. Raises ValueError with all problems found."""
    items, errors, seen = [], [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {
                **row,
                "expected_product_ids": _ids(row["expected_product_ids"]),
                "acceptable_product_ids": _ids(row["acceptable_product_ids"]),
                "answerable": row["answerable"].lower() == "true",
                "history": json.loads(row["history"]) if row.get("history") else [],
            }
            if item["id"] in seen:
                errors.append(f"{item['id']}: duplicated id")
            seen.add(item["id"])
            if item["question_type"] not in TYPES:
                errors.append(f"{item['id']}: unknown question_type {item['question_type']}")
            if item["answerable"] and not item["expected_product_ids"]:
                errors.append(f"{item['id']}: answerable but no expected_product_ids")
            if catalog_ids is not None:
                missing = [p for p in item["expected_product_ids"] + item["acceptable_product_ids"] if p not in catalog_ids]
                if missing:
                    errors.append(f"{item['id']}: ids not in catalog {missing}")
            items.append(item)
    if errors:
        raise ValueError("invalid golden dataset:\n" + "\n".join(errors))
    return items


def catalog_ids() -> set[str]:
    return set(pd.read_parquet(DATA_DIR / "catalog.parquet", columns=["product_id"])["product_id"])


def to_opik(item: dict) -> dict:
    """Rename the keys Opik reserves: its "id" becomes "item_id", and the CSV "source" (esci/manual) becomes "golden_source"."""
    row = {}
    for k, v in item.items():
        if k == "id":
            continue
        row["golden_source" if k == "source" else k] = v
    row["item_id"] = item["id"]
    return row


def sync(dataset, items: list[dict]) -> tuple[int, int]:
    """Make the Opik dataset match the local items (matched by item id). Safe to repeat.

    Remote items whose id is no longer in the CSV are deleted, so the evaluator never meets an item it cannot find.
    Returns (deleted, inserted).
    """
    local_ids = {i["id"] for i in items}
    remote = dataset.get_items()
    stale = [r["id"] for r in remote if r.get("item_id") not in local_ids]
    if stale:
        dataset.delete(stale)
    kept = {r.get("item_id") for r in remote if r.get("item_id") in local_ids}
    new = [to_opik(i) for i in items if i["id"] not in kept]
    if new:
        dataset.insert(new)
    return len(stale), len(new)


def upload(items: list[dict]) -> tuple[int, int]:
    """Sync the Opik dataset with the local items. Returns (deleted, inserted)."""
    client = observability.client()
    if client is None:
        raise RuntimeError("Opik is not configured (set OPIK_URL_OVERRIDE)")
    dataset = client.get_or_create_dataset(DATASET_NAME)
    return sync(dataset, items)


if __name__ == "__main__":
    data = load_golden(catalog_ids=catalog_ids())
    kinds = pd.Series([i["question_type"] for i in data]).value_counts().to_dict()
    print(f"{len(data)} items valid. mix: {kinds}")
    if "--upload" in sys.argv:
        deleted, inserted = upload(data)
        print(f"Opik dataset {DATASET_NAME}: deleted {deleted} stale items, inserted {inserted} new items")
