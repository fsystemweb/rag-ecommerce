"""Run the golden dataset through the real pipeline and report the metrics.

Usage:
  python -m evaluation.run --retrieval-only              # embeddings only (cheap)
  python -m evaluation.run --search bm25 --retrieval-only  # no API calls at all
  python -m evaluation.run [--k 5] [--limit N] [--search hybrid|dense|bm25] [--rerank]
                           [--generator MODEL] [--floor recall@5=0.4]

Without --retrieval-only this makes paid OpenAI calls. With OPIK_URL_OVERRIDE set,
the run is also logged as an Opik experiment. Exit code 1 if a --floor is not met.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from statistics import mean

from app import config, guardrails, observability
from app.chains import Pipeline, format_context, make_llm
from app.ingestion import get_store, load_catalog, split
from app.models import AskRequest, Turn
from app.prompt_loader import load_prompt, render
from app.retriever import Retriever
from evaluation import metrics
from evaluation.dataset import DATASET_NAME, load_golden

RETRIEVAL_KEYS = ["precision@5", "recall@5", "hit_rate@5", "ndcg@5", "precision@10", "recall@10", "hit_rate@10", "ndcg@10", "mrr"]
GENERATION_KEYS = ["product_precision", "product_recall", "refusal_accuracy", "faithfulness", "answer_correctness"]


class CatalogStore:
    """Minimal store for BM25-only runs: reads the chunks from the catalog, no embeddings."""

    def get(self, include=None):
        docs = [c for _, row in load_catalog().iterrows() for c in split(row)]
        return {"ids": [d.id for d in docs], "documents": [d.page_content for d in docs], "metadatas": [d.metadata for d in docs]}


def retrieval_query(item, retrieval_only):
    """Follow-ups need a rewrite (an LLM call). In retrieval-only mode we join the turns instead."""
    if item["history"] and retrieval_only:
        first = next(t["content"] for t in item["history"] if t["role"] == "user")
        return f"{first} {item['input'].strip('¿?')}"
    return item["input"]


def run_item(item, retriever, pipeline, retrieval_only):
    """Return {metric: value} for one golden item (metrics that do not apply are left out)."""
    scores = {}
    expected = set(item["expected_product_ids"])
    if item["answerable"]:
        chunks = retriever.retrieve(retrieval_query(item, retrieval_only), k=10)
        ids = [c.metadata["product_id"] for c in chunks]
        gains = {**{p: metrics.GAINS["S"] for p in item["acceptable_product_ids"]}, **{p: metrics.GAINS["E"] for p in expected}}
        scores.update(metrics.retrieval_scores(ids, expected, gains))
    if retrieval_only:
        return scores

    history = [Turn(**t) for t in item["history"]]
    resp = pipeline.ask(AskRequest(question=item["input"], history=history))
    refused = resp.answer == guardrails.REFUSAL
    scores["refusal_accuracy"] = metrics.refusal_accuracy(refused, item["answerable"])
    if item["answerable"]:
        answer_ids = [p.product_id for p in resp.products]
        scores["product_precision"] = metrics.product_precision(answer_ids, expected)
        scores["product_recall"] = metrics.product_recall(answer_ids, expected)
        if not refused:
            context = format_context(chunks[: config.RETRIEVAL_K])
            scores["faithfulness"] = pipeline.judge_answer(item["input"], context, resp.answer)["score"] / 5
            scores["answer_correctness"] = correctness(pipeline, item, resp.answer)
        else:
            scores["faithfulness"] = scores["answer_correctness"] = 0.0
    return scores


def correctness(pipeline, item, answer):
    template, _ = load_prompt("rag-ecommerce-correctness", "correctness_prompt.txt")
    reply = pipeline.judge.invoke(render(template, question=item["input"], expected=item["expected_output"], answer=answer)).content
    match = re.search(r'"score"\s*:\s*([01])', reply)
    return float(match.group(1)) if match else 0.0


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
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--search", choices=["hybrid", "dense", "bm25"], default="hybrid")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--generator", default=config.GENERATOR_LLM)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--floor", action="append", default=[], help="e.g. recall@5=0.4")
    args = ap.parse_args(argv)
    config.RETRIEVAL_K = args.k

    store = CatalogStore() if args.search == "bm25" else get_store()
    retriever = Retriever(store, rerank=args.rerank, mode=args.search)
    pipeline = None if args.retrieval_only else Pipeline(retriever, make_llm(args.generator), make_llm(config.EVALUATOR_LLM))
    items = load_golden()[: args.limit]
    keys = RETRIEVAL_KEYS if args.retrieval_only else RETRIEVAL_KEYS + GENERATION_KEYS
    name = f"{args.search}-k{args.k}{'-rerank' if args.rerank else ''}-{'retrieval' if args.retrieval_only else args.generator}"

    rows = []

    def task(dataset_item):
        item = next(i for i in items if i["id"] == dataset_item.get("item_id", dataset_item.get("id")))
        scores = run_item(item, retriever, pipeline, args.retrieval_only)
        rows.append((item["question_type"], scores))
        return {**scores, "question_type": item["question_type"]}

    client = observability.client()
    if client is not None:
        from opik.evaluation import evaluate

        evaluate(
            dataset=client.get_dataset(DATASET_NAME), task=task, scoring_metrics=metrics.opik_metrics(keys),
            experiment_name=name, task_threads=1,
            experiment_config={"search": args.search, "k": args.k, "rerank": args.rerank, "generator": args.generator, "retrieval_only": args.retrieval_only},
        )
    else:
        for item in items:
            task({"id": item["id"]})

    table = aggregate(rows)
    print(f"\nexperiment: {name}  items: {len(items)}")
    print_table(table, keys)
    (config.ROOT / "evaluation" / "results").mkdir(exist_ok=True)
    (config.ROOT / "evaluation" / "results" / f"{name}.json").write_text(json.dumps(table, indent=2))
    failed = check_floors(table, args.floor)
    if failed:
        print("FAILED floors:", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
