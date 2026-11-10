"""Run the golden dataset through the real pipeline and report the metrics.

Usage:
  python -m src.evaluation.evaluator --retrieval-only   # embeddings only (cheaper)
  python -m src.evaluation.evaluator [--k 10] [--limit N] [--generator MODEL] [--name NAME] [--floor recall@5=0.4]

Both modes make paid Nebius calls (without --retrieval-only, also classifier, rewrite and generation).
With OPIK_URL_OVERRIDE set, it runs over the items of the Opik dataset, whatever it holds (sync it with
`python -m src.evaluation.dataset --upload`), and logs the run as an Opik experiment with the generated answers.
Without Opik it uses the local golden CSV. Exit code 1 if a --floor is not met.
"""

import argparse
import json
import sys
from collections import defaultdict
from statistics import mean

from src.agent import guardrails
from src.agent.rag_agent import Pipeline
from src.agent.retriever import Retriever
from src.core import config
from src.core.llm import make_llm
from src.core.models import AskRequest, Turn
from src.evaluation import metrics
from src.evaluation.dataset import DATASET_NAME, load_golden
from src.ingestion.index import get_store
from src.observability import tracer as observability

RETRIEVAL_KEYS = ["precision@5", "recall@5", "precision@10", "recall@10"]
GENERATION_KEYS = ["product_precision", "product_recall", "refusal_accuracy"]


def retrieval_query(item, retrieval_only):
    """Follow-ups need a rewrite (an LLM call). In retrieval-only mode we join the turns instead."""
    if item["history"] and retrieval_only:
        first = next(t["content"] for t in item["history"] if t["role"] == "user")
        return f"{first} {item['input'].strip('¿?')}"
    return item["input"]


REQUIRED_FIELDS = ("input", "answerable", "question_type", "expected_product_ids")


def check_item(item):
    """Fail with a clear message if a dataset item lacks a field the evaluation needs. History is optional."""
    missing = [f for f in REQUIRED_FIELDS if item.get(f) is None]
    if missing:
        ident = item.get("item_id") or item.get("id")
        raise ValueError(f"dataset item {ident} is missing {missing}")
    item.setdefault("history", [])
    return item


def run_item(item, retriever, pipeline, retrieval_only, output=None):
    """Return {metric: value} for one golden item (metrics that do not apply are left out).

    If `output` is a dict, it is filled with the generated answer and products so they can be logged.
    """
    scores = {}
    expected = set(item["expected_product_ids"])
    if item["answerable"]:
        chunks = retriever.retrieve(retrieval_query(item, retrieval_only), k=10)
        ids = [c.metadata["product_id"] for c in chunks]
        scores.update(metrics.retrieval_scores(ids, expected))
    if retrieval_only:
        return scores

    history = [Turn(**t) for t in item["history"]]
    resp = pipeline.ask(AskRequest(question=item["input"], history=history))
    if output is not None:
        output["answer"] = resp.answer
        output["products"] = [f"{p.product_id}: {p.title}" for p in resp.products]
    refused = resp.answer in (guardrails.REFUSAL, guardrails.NO_MATCH)
    scores["refusal_accuracy"] = metrics.refusal_accuracy(refused, item["answerable"])
    if item["answerable"]:
        answer_ids = [p.product_id for p in resp.products]
        scores["product_precision"] = metrics.product_precision(answer_ids, expected)
        scores["product_recall"] = metrics.product_recall(answer_ids, expected)
    return scores


def aggregate(rows):
    """rows: [(question_type, scores)] -> {group: {metric: mean}} with an 'ALL' group."""
    groups = defaultdict(lambda: defaultdict(list))
    for qtype, scores in rows:
        for key, value in scores.items():
            groups[qtype][key].append(value)
            groups["ALL"][key].append(value)
    return {g: {k: mean(v) for k, v in d.items()} for g, d in groups.items()}


def check_floors(table, floors):
    """floors: ['recall@5=0.4']. Returns the list of failed checks."""
    failed = []
    for spec in floors:
        name, value = spec.split("=")
        got = table.get("ALL", {}).get(name)
        if got is None or got < float(value):
            failed.append(f"{name}: {got} < {value}")
    return failed


def print_table(table, keys):
    print("| type | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for group in sorted(table, key=lambda g: (g == "ALL", g)):
        cells = [f"{table[group][k]:.3f}" if k in table[group] else "-" for k in keys]
        print(f"| {group} | " + " | ".join(cells) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--generator", default=config.GENERATOR_LLM)
    ap.add_argument("--name", help="experiment name (default: built from the settings)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--floor", action="append", default=[], help="e.g. recall@5=0.4")
    args = ap.parse_args(argv)
    config.RETRIEVAL_K = args.k

    retriever = Retriever(get_store())
    pipeline = None if args.retrieval_only else Pipeline(retriever, make_llm(args.generator))
    keys = RETRIEVAL_KEYS if args.retrieval_only else RETRIEVAL_KEYS + GENERATION_KEYS
    name = args.name or f"hybrid-k{args.k}-{'retrieval' if args.retrieval_only else args.generator}"

    rows = []

    def task(dataset_item):
        """Run one item. It works on whatever the dataset holds, so no local file has to match it."""
        item = check_item(dict(dataset_item))
        generated = {}
        scores = run_item(item, retriever, pipeline, args.retrieval_only, generated)
        rows.append((item["question_type"], scores))
        # the Opik experiment item shows this dict as the output: answer first, then the scores
        return {**generated, **scores, "question_type": item["question_type"]}

    client = observability.client()
    if client is not None:
        from opik.evaluation import evaluate

        evaluate(
            dataset=client.get_dataset(DATASET_NAME),
            task=task,
            scoring_metrics=metrics.opik_metrics(keys),
            experiment_name=name,
            nb_samples=args.limit,
            task_threads=1,
            experiment_config={
                "search": "hybrid",
                "k": args.k,
                "generator": args.generator,
                "retrieval_only": args.retrieval_only,
            },
        )
    else:
        for item in load_golden()[: args.limit]:
            task(item)

    table = aggregate(rows)
    print(f"\nexperiment: {name}  items: {len(rows)}")
    print_table(table, keys)
    results = config.ROOT / "src" / "evaluation" / "results"
    results.mkdir(exist_ok=True)
    # model names contain "/" (e.g. deepseek-ai/DeepSeek-V4), which is not valid inside a file name
    (results / f"{name.replace('/', '_')}.json").write_text(json.dumps(table, indent=2))
    failed = check_floors(table, args.floor)
    if failed:
        print("FAILED floors:", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
