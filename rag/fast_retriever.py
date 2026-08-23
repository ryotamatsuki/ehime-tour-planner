from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag.retriever import EhimeRetriever, RetrievalItem, _tokenize_for_bm25


@dataclass
class _CorpusIndex:
    chunks: list
    bm25_tokens: list[list[str]]
    doc_vecs: np.ndarray


class CachedSpotRetriever(EhimeRetriever):
    """EhimeRetriever with an in-process LRU cache for document embeddings.

    Streamlit keeps this retriever in st.cache_resource, so repeated plan runs
    against the same collected pages reuse the expensive CPU document vectors.
    Only the query vector is recomputed for each new user request.
    """

    def __init__(self, api_key: str | None = None, *, max_cached_corpora: int = 8):
        super().__init__(api_key=api_key)
        self.max_cached_corpora = max(1, max_cached_corpora)
        self._corpus_cache: OrderedDict[str, _CorpusIndex] = OrderedDict()
        self.last_metrics: dict[str, float | int | bool] = {}

    @staticmethod
    def _corpus_key(items: list[RetrievalItem]) -> str:
        digest = hashlib.sha256()
        for item in items:
            digest.update(item.url.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(item.content.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _get_or_build_index(self, items: list[RetrievalItem]) -> tuple[_CorpusIndex, bool, float]:
        key = self._corpus_key(items)
        cached = self._corpus_cache.get(key)
        if cached is not None:
            self._corpus_cache.move_to_end(key)
            return cached, True, 0.0

        chunks = self._make_chunks(items)
        if not chunks:
            empty = _CorpusIndex(chunks=[], bm25_tokens=[], doc_vecs=np.empty((0, 0)))
            return empty, False, 0.0

        bm25_tokens = [_tokenize_for_bm25(chunk.text) for chunk in chunks]
        doc_texts = [f"検索文書: {chunk.text}" for chunk in chunks]
        started = time.perf_counter()
        doc_vecs = self.embedding_model.encode(
            doc_texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embedding_ms = round((time.perf_counter() - started) * 1000, 1)
        index = _CorpusIndex(chunks=chunks, bm25_tokens=bm25_tokens, doc_vecs=doc_vecs)
        self._corpus_cache[key] = index
        self._corpus_cache.move_to_end(key)
        while len(self._corpus_cache) > self.max_cached_corpora:
            self._corpus_cache.popitem(last=False)
        return index, False, embedding_ms

    def retrieve_spot_candidates(
        self,
        *,
        items: list[RetrievalItem],
        user_query: str,
        candidate_limit: int = 8,
    ) -> tuple[list[str], list[dict]]:
        """Return compact, ID-addressable candidates for itinerary generation."""
        total_started = time.perf_counter()
        index, cache_hit, doc_embedding_ms = self._get_or_build_index(items)
        if not index.chunks:
            self.last_metrics = {
                "cache_hit": cache_hit,
                "chunk_count": 0,
                "candidate_count": 0,
                "doc_embedding_ms": doc_embedding_ms,
                "query_embedding_ms": 0.0,
                "retrieval_total_ms": round((time.perf_counter() - total_started) * 1000, 1),
            }
            return [], []

        bm25 = BM25Okapi(index.bm25_tokens)
        bm25_scores = np.asarray(bm25.get_scores(_tokenize_for_bm25(user_query)))
        bm25_order = np.argsort(-bm25_scores).tolist()

        query_started = time.perf_counter()
        q_vec = self.embedding_model.encode(
            [f"検索クエリ: {user_query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        query_embedding_ms = round((time.perf_counter() - query_started) * 1000, 1)

        dense_scores = index.doc_vecs @ q_vec
        dense_order = np.argsort(-dense_scores).tolist()

        fused: dict[int, float] = {}
        for order in (bm25_order, dense_order):
            for rank, idx in enumerate(order, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(fused, key=fused.get, reverse=True)
        candidates: list[dict] = []
        seen_urls: set[str] = set()
        for idx in ranked:
            chunk = index.chunks[idx]
            if chunk.url in seen_urls:
                continue
            if not re.match(r"^https?://", chunk.url):
                continue
            seen_urls.add(chunk.url)
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()[:420]
            candidates.append(
                {
                    "spot_id": f"S{len(candidates) + 1:03d}",
                    "title": chunk.title,
                    "url": chunk.url,
                    "site": chunk.site,
                    "excerpt": excerpt,
                }
            )
            if len(candidates) >= max(1, candidate_limit):
                break

        context = [
            f"{c['spot_id']} | {c['title']} | {c['site']} | {c['excerpt']}"
            for c in candidates
        ]
        self.last_metrics = {
            "cache_hit": cache_hit,
            "chunk_count": len(index.chunks),
            "candidate_count": len(candidates),
            "doc_embedding_ms": doc_embedding_ms,
            "query_embedding_ms": query_embedding_ms,
            "retrieval_total_ms": round((time.perf_counter() - total_started) * 1000, 1),
        }
        return context, candidates
