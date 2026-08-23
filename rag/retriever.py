from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from bs4 import BeautifulSoup
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from tavily import TavilyClient


RURI_MODEL_ID = os.getenv("RURI_MODEL_ID", "cl-nagoya/ruri-v3-30m")
RRF_K = 60


class RetrievalItem(BaseModel):
    title: str
    url: str
    site: str
    content: str
    content_chars: int


@dataclass
class Chunk:
    text: str
    title: str
    url: str
    site: str


def _tokenize_for_bm25(text: str) -> list[str]:
    """Dependency-free Japanese-friendly tokenizer.

    Uses ASCII/alphanumeric words plus Japanese character bi-grams.
    Dense retrieval handles semantics; BM25 mainly contributes exact names.
    """
    normalized = re.sub(r"\s+", "", text.lower())
    ascii_tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", normalized)
    jp = "".join(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", normalized))
    bigrams = [jp[i : i + 2] for i in range(max(0, len(jp) - 1))]
    return ascii_tokens + bigrams or [normalized[:64]]


class EhimeRetriever:
    def __init__(self, api_key: str | None = None):
        self.client = TavilyClient(api_key) if api_key else None
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(RURI_MODEL_ID, device="cpu")
        return self._embedding_model

    def search_and_prepare(
        self,
        query: str,
        max_results: int = 8,
        add_web_search: bool = False,
    ) -> list[RetrievalItem]:
        if self.client is None:
            raise RuntimeError("TAVILY_API_KEY が設定されていません。")

        items: list[RetrievalItem] = []
        seen_urls: set[str] = set()
        iyokan_count = max_results if not add_web_search else max(1, max_results // 2)
        web_count = max_results - iyokan_count
        search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic")

        def process(results: Iterable[dict], is_iyokan: bool) -> None:
            for row in results:
                url = str(row.get("url", "") or "")
                if not url or url in seen_urls:
                    continue
                title = str(row.get("title", "") or url)
                raw = row.get("raw_content") or row.get("content") or ""
                if isinstance(raw, list):
                    raw = "\n".join(map(str, raw))
                cleaned = self._clean_text(str(raw))
                if not cleaned:
                    continue
                site = "いよ観ネット" if is_iyokan else self._site_from_url(url)
                items.append(
                    RetrievalItem(
                        title=title[:180],
                        url=url,
                        site=site,
                        content=cleaned[:12000],
                        content_chars=len(cleaned),
                    )
                )
                seen_urls.add(url)

        iyokan = self.client.search(
            query=query,
            search_depth=search_depth,
            include_raw_content="markdown",
            include_answer=False,
            include_domains=["iyokannet.jp"],
            max_results=iyokan_count,
        )
        process(iyokan.get("results", []), True)

        if add_web_search and web_count > 0:
            web = self.client.search(
                query=query,
                search_depth=search_depth,
                include_raw_content="markdown",
                include_answer=False,
                max_results=web_count,
            )
            process(web.get("results", []), False)

        return items

    def retrieve_for_plan(
        self,
        items: list[RetrievalItem],
        user_query: str,
        k: int = 8,
    ) -> tuple[list[str], list[dict]]:
        chunks = self._make_chunks(items)
        if not chunks:
            return [], []

        bm25_tokens = [_tokenize_for_bm25(c.text) for c in chunks]
        bm25 = BM25Okapi(bm25_tokens)
        bm25_scores = np.asarray(bm25.get_scores(_tokenize_for_bm25(user_query)))
        bm25_order = np.argsort(-bm25_scores).tolist()

        doc_texts = [f"検索文書: {c.text}" for c in chunks]
        query_text = [f"検索クエリ: {user_query}"]
        doc_vecs = self.embedding_model.encode(
            doc_texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q_vec = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        dense_scores = doc_vecs @ q_vec
        dense_order = np.argsort(-dense_scores).tolist()

        fused: dict[int, float] = {}
        for order in (bm25_order, dense_order):
            for rank, idx in enumerate(order, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(fused, key=fused.get, reverse=True)
        selected: list[Chunk] = []
        per_url: dict[str, int] = {}
        for idx in ranked:
            ch = chunks[idx]
            if per_url.get(ch.url, 0) >= 2:
                continue
            selected.append(ch)
            per_url[ch.url] = per_url.get(ch.url, 0) + 1
            if len(selected) >= k:
                break

        context = [
            f"出典: {c.title}\nURL: {c.url}\nサイト: {c.site}\n内容:\n{c.text}"
            for c in selected
        ]
        sources: list[dict] = []
        seen: set[str] = set()
        for c in selected:
            if c.url not in seen:
                sources.append({"title": c.title, "url": c.url, "site": c.site})
                seen.add(c.url)
        return context, sources

    def _make_chunks(self, items: list[RetrievalItem], size: int = 1000, overlap: int = 120) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = max(1, size - overlap)
        for item in items:
            text = item.content
            for start in range(0, len(text), step):
                part = text[start : start + size].strip()
                if len(part) < 80:
                    continue
                chunks.append(
                    Chunk(
                        text=part,
                        title=item.title,
                        url=item.url,
                        site=item.site,
                    )
                )
        return chunks

    @staticmethod
    def _site_from_url(url: str) -> str:
        match = re.match(r"https?://([^/]+)", url)
        return (match.group(1) if match else url).replace("www.", "")

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text(separator="\n")
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()
