"""Hybrid retriever: dense (embeddings) + BM25, merged with Reciprocal Rank Fusion.

- always returns `k` >= 5 chunks, from distinct products when possible
- returns [] if even the best dense score is below RETRIEVAL_MIN_SCORE
"""

import re
import unicodedata
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from src.core import config
from src.observability.tracer import track

POOL = 20  # candidates taken from each side before fusing
RRF_C = 60  # standard RRF constant


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float  # dense cosine similarity (0 if only BM25 found it)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fused: float = field(default=0.0)


def tokenize(text: str) -> list[str]:
    """Lowercase, fold accents (camión ~ camion), keep numbers and units (500ml, 5.95)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text)


def rrf(rankings: list[list[str]]) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = sum(1 / (C + rank))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_C + rank)
    return scores


class Retriever:
    def __init__(self, store):
        self.store = store
        self._bm25 = None  # built lazily, in memory (10k products is fast)

    def _load_bm25(self):
        data = self.store.get(include=["documents", "metadatas"])
        self._ids, self._texts, self._metas = data["ids"], data["documents"], data["metadatas"]
        self._bm25 = BM25Okapi([tokenize(t) for t in self._texts])

    def _dense(self, query, where):
        res = self.store.similarity_search_with_score(query, k=POOL, filter=where)
        # Chroma returns cosine *distance*; similarity = 1 - distance
        return [(d.id, d.page_content, d.metadata, 1.0 - dist) for d, dist in res]

    def _bm25_search(self, query, where):
        if self._bm25 is None:
            self._load_bm25()
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= POOL:
                break
            if where and any(self._metas[i].get(k) != v for k, v in where.items()):
                continue
            out.append((self._ids[i], self._texts[i], self._metas[i]))
        return out

    @track("retrieve", type="tool")
    def retrieve(self, query: str, k: int | None = None, where: dict | None = None) -> list[RetrievedChunk]:
        k = max(k or config.RETRIEVAL_K, 5)
        dense = self._dense(query, where)
        if not dense or max(d[3] for d in dense) < config.RETRIEVAL_MIN_SCORE:
            return []  # nothing relevant in the catalog

        chunks: dict[str, RetrievedChunk] = {}
        for rank, (cid, text, meta, sim) in enumerate(dense, start=1):
            chunks[cid] = RetrievedChunk(text, meta, sim, dense_rank=rank)
        bm25 = self._bm25_search(query, where)
        for rank, (cid, text, meta) in enumerate(bm25, start=1):
            if cid in chunks:
                chunks[cid].bm25_rank = rank
            else:
                chunks[cid] = RetrievedChunk(text, meta, 0.0, bm25_rank=rank)

        fused = rrf([[d[0] for d in dense], [b[0] for b in bm25]])
        ordered = sorted(chunks, key=lambda cid: -fused[cid])
        result, seen = [], set()
        for cid in ordered:  # one chunk per product
            chunks[cid].fused = fused[cid]
            pid = chunks[cid].metadata["product_id"]
            if pid not in seen:
                seen.add(pid)
                result.append(chunks[cid])
        return result[:k]
