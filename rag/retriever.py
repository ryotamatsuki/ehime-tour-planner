from __future__ import annotations
import re
import time
import html
import hashlib
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from tavily import TavilyClient
from bs4 import BeautifulSoup

from google import genai
from google.genai import types

# --- 切替: FAISS を使わず Numpy 類似度のみでも動かせる ---
USE_FAISS = True
try:
    if USE_FAISS:
        import faiss  # type: ignore
except Exception:
    USE_FAISS = False

class RetrievalItem(BaseModel):
    title: str
    url: str
    site: str
    content: str
    content_chars: int

class EhimeRetriever:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key)
        self.gclient = genai.Client()  # GEMINI_API_KEY は環境/Secrets から

    # --- 1) 検索→抽出→要約/クリーニング ---
    def search_and_prepare(self, query: str, max_results: int = 8) -> List[RetrievalItem]:
        resp = self.client.search(
            query=query,
            search_depth="advanced",
            include_raw_content="markdown",
            include_answer=False,
            include_domains=["iyokannet.jp"],
            max_results=max_results,
            chunks_per_source=3,
        )
        items: List[RetrievalItem] = []
        for r in resp.get("results", []):
            url = r.get("url", "")
            title = r.get("title", "") or url
            raw_md = r.get("raw_content", "") or "n".join(r.get("content", []))
            cleaned = self._clean_text(raw_md)
            if not cleaned:
                # 最低限 HTML 抽出を試行
                ext = self.client.extract(url)
                cleaned = self._clean_text(ext.get("text", ""))
            if cleaned:
                items.append(
                    RetrievalItem(
                        title=title[:180],
                        url=url,
                        site="いよ観ネット",
                        content=cleaned[:10000],  # 入力長の安全確保
                        content_chars=len(cleaned),
                    )
                )
        return items

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Markdown/HTML → テキスト
        soup = BeautifulSoup(text, "html.parser")
        txt = soup.get_text(separator="n")
        txt = html.unescape(txt)
        txt = re.sub(r"n{3,}", "nn", txt)
        txt = re.sub(r"s+", " ", txt)
        # いよ観ネットの原文転載を避けるため、チャンク化前に短縮
        return txt.strip()

    # --- 2) 埋め込みユーティリティ ---
    def _embed(self, texts: List[str], task_type: str, dim: int = 768) -> np.ndarray:
        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
        res = self.gclient.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=cfg,
        )
        vecs = [np.array(e.values, dtype="float32") for e in res.embeddings]
        return np.vstack(vecs)

    # --- 3) ベクトル化 → 検索 ---
    def _build_index(self, chunks: List[str]):
        X = self._embed(chunks, task_type="RETRIEVAL_DOCUMENT")
        if USE_FAISS:
            index = faiss.IndexFlatIP(X.shape[1])
            faiss.normalize_L2(X)
            index.add(X)
            return index, X
        # 代替: 生行列を返す
        return None, X

    def _search_index(self, index, X: np.ndarray, q: np.ndarray, topk: int = 8) -> Tuple[List[int], List[float]]:
        if USE_FAISS and index is not None:
            faiss.normalize_L2(q)
            D, I = index.search(q, topk)
            return I[0].tolist(), D[0].tolist()
        # 代替: NumPy コサイン類似度
        # 正規化
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = Xn @ qn.T
        order = np.argsort(-sims[:, 0])[:topk]
        return order.tolist(), sims[order, 0].tolist()

    def _chunk(self, text: str, size: int = 800, overlap: int = 120) -> List[str]:
        # 文字ベース簡易チャンク（和文前提）
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + size])
            i += size - overlap
        return chunks

    def retrieve_for_plan(self, items: List[RetrievalItem], user_query: str, k: int = 8):
        # すべての候補をチャンク化
        chunk_texts, chunk_meta = [], []
        for it in items:
            chunks = self._chunk(it.content)
            for ch in chunks:
                chunk_texts.append(ch)
                chunk_meta.append({"title": it.title, "url": it.url, "site": it.site})
        # インデックス作成
        index, X = self._build_index(chunk_texts)
        q = self._embed([user_query], task_type="RETRIEVAL_QUERY")
        ids, scores = self._search_index(index, X, q, topk=k)
        selected = []
        used_sources = []
        seen = set()
        for idx in ids:
            meta = chunk_meta[idx]
            # 同一URLの過度な重複を避ける
            key = meta["url"]
            if key not in seen:
                seen.add(key)
                used_sources.append(meta)
            # 原文貼付ではなく "要点要約" を Gemini に一度かけて圧縮
            summary = self._summarize_for_context(chunk_texts[idx])
            selected.append(f"出典: {meta['title']} | {meta['url']}n要点:n{summary}")
        return selected, used_sources

    def _summarize_for_context(self, text: str) -> str:
        # いよ観ネットのポリシーに配慮: 引用ではなく短い要点箇条書き
        prompt = (
            "以下の観光記事テキストから、固有名詞と実用情報（場所・体験・時期・所要時間・注意点）を日本語で5点以内に簡潔要約してください。n"
            "**原文の連続した引用は禁止。必ず言い換え・要約で**。最大400字。nn" + text[:4000]
        )
        resp = self.gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text.strip()