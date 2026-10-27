"""Hybrid retrieval: BM25 (lexical) + dense (semantic), fused with Reciprocal Rank Fusion.

* BM25 catches exact scientific tokens (model names, dataset names, numbers).
* Dense catches paraphrases. Default dense encoder is TF-IDF + truncated SVD
  (LSA) so the system runs fully offline; if `sentence-transformers` is
  installed and SCIAGENT_EMBED_MODEL is set, a neural encoder is used instead.
* RRF is rank-based, so the two score scales never need calibrating.
* A per-paper cap keeps one long paper from flooding the top-k, which matters
  when the question needs evidence from several (possibly conflicting) papers.
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = set("""a an the of and or in on for to with by from is are was were be been as at that this these those it its we our
their which who whom what when where how why do does did not no than then so such into over under between about can
could may might also both each more most other some only same very""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
    return [t for t in toks if t not in STOP]


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    kind: str
    section: str
    text: str
    score: float
    meta: dict


class _LSA:
    def __init__(self, dim=256):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                   token_pattern=None)
        self.dim = dim

    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(k, random_state=0)
        return self._norm(self.svd.fit_transform(X))

    def encode(self, texts):
        return self._norm(self.svd.transform(self.vec.transform(texts)))

    @staticmethod
    def _norm(Z):
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


class _ST:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit(self, texts):
        return self.encode(texts)

    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))


class HybridIndex:
    def __init__(self, chunks: list[dict], papers: list[dict]):
        self.chunks = chunks
        self.papers = {p["id"]: p for p in papers}
        self.by_id = {c["chunk_id"]: c for c in chunks}
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        name = os.environ.get("SCIAGENT_EMBED_MODEL")
        self.encoder = _ST(name) if name else _LSA()
        self.dense = self.encoder.fit(texts)

    # ------------------------------------------------------------- search
    def _rank_bm25(self, q):
        return np.asarray(self.bm25.get_scores(tokenize(q)))

    def _rank_dense(self, q):
        return self.dense @ self.encoder.encode([q])[0]

    def search(self, query: str, k: int = 8, mode: str = "hybrid", kinds: list[str] | None = None,
               paper_ids: list[str] | None = None, per_paper: int = 3, rrf_k: int = 60) -> list[Hit]:
        mask = np.ones(len(self.chunks), bool)
        if kinds:
            mask &= np.array([c["kind"] in kinds for c in self.chunks])
        if paper_ids:
            mask &= np.array([c["paper_id"] in paper_ids for c in self.chunks])
        if not mask.any() or not query.strip():
            return []
        if mode == "bm25":
            score = self._rank_bm25(query)
        elif mode == "dense":
            score = self._rank_dense(query)
        else:
            score = np.zeros(len(self.chunks))
            for s in (self._rank_bm25(query), self._rank_dense(query)):
                order = np.argsort(-s)
                ranks = np.empty(len(s), int)
                ranks[order] = np.arange(len(s))
                score += 1.0 / (rrf_k + ranks + 1)
        score = np.where(mask, score, -np.inf)
        hits, count = [], {}
        for i in np.argsort(-score):
            if not np.isfinite(score[i]):
                break
            c = self.chunks[i]
            if count.get(c["paper_id"], 0) >= per_paper:
                continue
            count[c["paper_id"]] = count.get(c["paper_id"], 0) + 1
            hits.append(Hit(c["chunk_id"], c["paper_id"], c["kind"], c["section"], c["text"], float(score[i]), c.get("meta", {})))
            if len(hits) >= k:
                break
        return hits

    # ------------------------------------------------------------- io
    @classmethod
    def build(cls, index_dir: Path) -> "HybridIndex":
        chunks = [json.loads(l) for l in open(index_dir / "chunks.jsonl")]
        papers = [json.loads(l) for l in open(index_dir / "papers.jsonl")]
        idx = cls(chunks, papers)
        with open(index_dir / "index.pkl", "wb") as f:
            pickle.dump(idx, f)
        return idx

    @classmethod
    def load(cls, index_dir: Path) -> "HybridIndex":
        p = index_dir / "index.pkl"
        if not p.exists() or p.stat().st_mtime < (index_dir / "chunks.jsonl").stat().st_mtime:
            return cls.build(index_dir)
        with open(p, "rb") as f:
            return pickle.load(f)
