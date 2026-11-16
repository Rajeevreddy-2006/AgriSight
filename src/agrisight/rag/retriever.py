"""Hybrid retriever: BM25 + dense embeddings fused with RRF, optional cross-encoder rerank.

The `mode` argument makes ablations easy: bm25 | dense | hybrid | hybrid+rerank.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fusion import reciprocal_rank_fusion, tokenize

# BGE embedding models expect this instruction in front of queries (not documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HybridRetriever:
    def __init__(self, index_dir: str | Path, reranker: str | None = None, candidates: int = 30, default_mode: str = "hybrid+rerank"):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        index_dir = Path(index_dir)
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run: python -m agrisight.rag.index")
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.emb = np.load(index_dir / "embeddings.npy")
        meta = json.loads((index_dir / "meta.json").read_text())
        self.embed_model_name = meta["embed_model"]
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.candidates = candidates
        self.default_mode = default_mode
        self.reranker = None
        if reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker)

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[: self.candidates]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_rank(self, query: str) -> list[int]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.embed_model_name.lower() else ""
        q = self.embedder.encode([prefix + query], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(self.emb @ q))[: self.candidates]]

    def search(self, query: str, k: int = 5, mode: str | None = None) -> list[dict]:
        mode = mode or self.default_mode
        if mode == "bm25":
            ids = self._bm25_rank(query)
        elif mode == "dense":
            ids = self._dense_rank(query)
        elif mode in ("hybrid", "hybrid+rerank"):
            ids = [i for i, _ in reciprocal_rank_fusion([self._bm25_rank(query), self._dense_rank(query)])][: self.candidates]
            if mode == "hybrid+rerank" and self.reranker is not None and ids:
                scores = self.reranker.predict([(query, self.chunks[i]["text"]) for i in ids])
                ids = [ids[j] for j in np.argsort(-np.asarray(scores))]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        return [{**self.chunks[i], "rank": r + 1} for r, i in enumerate(ids[:k])]
