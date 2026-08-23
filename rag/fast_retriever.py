from __future__ import annotations

import hashlib
import re
import time
import unicodedata
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


def canonical_spot_key(title: str) -> str:
    """Normalize search-result titles so duplicate facility pages collapse.

    Tavily can return multiple iyokannet pages for one physical place, e.g.
    "松山城観光" and "松山城 - 愛媛県". URL-only dedupe cannot catch this,
    so remove common portal/suffix noise before candidate IDs are assigned.
    """
    text = unicodedata.normalize("NFKC", str(title or "")).lower().strip()
    text = re.sub(r"【[^】]*】|\[[^\]]*\]|（[^）]*）|\([^)]*\)", "", text)
    # Portal/site suffixes usually follow a separator; keep only the facility side.
    text = re.split(r"\s+(?:[-–—|｜])\s+", text, maxsplit=1)[0]
    for noise in (
        "愛媛県公式観光サイト",
        "愛媛県観光サイト",
        "いよ観ネット",
        "公式ホームページ",
        "公式サイト",
    ):
        text = text.replace(noise, "")
    text = re.sub(r"(?:観光案内|観光情報|観光|愛媛県)$", "", text)
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", text)


class CachedSpotRetriever(EhimeRetriever):
    """EhimeRetriever with an in-process LRU cache for document embeddings."""

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

    def _get_or_build_index(
        self, items: list[RetrievalItem]
    ) -> tuple[_CorpusIndex, bool, float]:
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
        index = _CorpusIndex(
            chunks=chunks,
            bm25_tokens=bm25_tokens,
            doc_vecs=doc_vecs,
        )
        self._corpus_cache[key] = index
        self._corpus_cache.move_to_end(key)
        while len(self._corpus_cache) > self.max_cached_corpora:
            self._corpus_cache.popitem(last=False)
        return index, False, embedding_ms

    def _rank(
        self,
        index: _CorpusIndex,
        user_query: str,
    ) -> tuple[list[int], float]:
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
        return sorted(fused, key=fused.get, reverse=True), query_embedding_ms

    def _set_metrics(
        self,
        *,
        total_started: float,
        cache_hit: bool,
        chunk_count: int,
        candidate_count: int,
        doc_embedding_ms: float,
        query_embedding_ms: float,
    ) -> None:
        self.last_metrics = {
            "cache_hit": cache_hit,
            "chunk_count": chunk_count,
            "candidate_count": candidate_count,
            "doc_embedding_ms": doc_embedding_ms,
            "query_embedding_ms": query_embedding_ms,
            "retrieval_total_ms": round(
                (time.perf_counter() - total_started) * 1000, 1
            ),
        }

    def retrieve_for_plan(
        self,
        items: list[RetrievalItem],
        user_query: str,
        k: int = 8,
    ) -> tuple[list[str], list[dict]]:
        """Compatibility path for refine/repair, backed by the same cache."""
        total_started = time.perf_counter()
        index, cache_hit, doc_embedding_ms = self._get_or_build_index(items)
        if not index.chunks:
            self._set_metrics(
                total_started=total_started,
                cache_hit=cache_hit,
                chunk_count=0,
                candidate_count=0,
                doc_embedding_ms=doc_embedding_ms,
                query_embedding_ms=0.0,
            )
            return [], []

        ranked, query_embedding_ms = self._rank(index, user_query)
        selected = []
        per_url: dict[str, int] = {}
        for idx in ranked:
            chunk = index.chunks[idx]
            if per_url.get(chunk.url, 0) >= 2:
                continue
            selected.append(chunk)
            per_url[chunk.url] = per_url.get(chunk.url, 0) + 1
            if len(selected) >= k:
                break

        context = [
            f"出典: {chunk.title}\nURL: {chunk.url}\nサイト: {chunk.site}\n内容:\n{chunk.text}"
            for chunk in selected
        ]
        sources: list[dict] = []
        seen: set[str] = set()
        for chunk in selected:
            if chunk.url not in seen:
                sources.append(
                    {"title": chunk.title, "url": chunk.url, "site": chunk.site}
                )
                seen.add(chunk.url)

        self._set_metrics(
            total_started=total_started,
            cache_hit=cache_hit,
            chunk_count=len(index.chunks),
            candidate_count=len(sources),
            doc_embedding_ms=doc_embedding_ms,
            query_embedding_ms=query_embedding_ms,
        )
        return context, sources

    def retrieve_spot_candidates(
        self,
        *,
        items: list[RetrievalItem],
        user_query: str,
        candidate_limit: int = 8,
    ) -> tuple[list[str], list[dict]]:
        """Return compact, ID-addressable, facility-deduplicated candidates."""
        total_started = time.perf_counter()
        index, cache_hit, doc_embedding_ms = self._get_or_build_index(items)
        if not index.chunks:
            self._set_metrics(
                total_started=total_started,
                cache_hit=cache_hit,
                chunk_count=0,
                candidate_count=0,
                doc_embedding_ms=doc_embedding_ms,
                query_embedding_ms=0.0,
            )
            return [], []

        ranked, query_embedding_ms = self._rank(index, user_query)
        candidates: list[dict] = []
        seen_urls: set[str] = set()
        seen_spot_keys: set[str] = set()
        for idx in ranked:
            chunk = index.chunks[idx]
            if chunk.url in seen_urls:
                continue
            if not re.match(r"^https?://", chunk.url) or "${" in chunk.url:
                continue

            spot_key = canonical_spot_key(chunk.title)
            if spot_key and spot_key in seen_spot_keys:
                continue

            seen_urls.add(chunk.url)
            if spot_key:
                seen_spot_keys.add(spot_key)
            excerpt = re.sub(r"\s+", " ", chunk.text).strip()[:420]
            candidates.append(
                {
                    "spot_id": f"S{len(candidates) + 1:03d}",
                    "title": chunk.title,
                    "url": chunk.url,
                    "site": chunk.site,
                    "excerpt": excerpt,
                    "canonical_key": spot_key,
                }
            )
            if len(candidates) >= max(1, candidate_limit):
                break

        context = [
            f"{candidate['spot_id']} | {candidate['title']} | "
            f"{candidate['site']} | {candidate['excerpt']}"
            for candidate in candidates
        ]
        self._set_metrics(
            total_started=total_started,
            cache_hit=cache_hit,
            chunk_count=len(index.chunks),
            candidate_count=len(candidates),
            doc_embedding_ms=doc_embedding_ms,
            query_embedding_ms=query_embedding_ms,
        )
        return context, candidates
