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
    def search_and_prepare(self, query: str, max_results: int = 8, add_web_search: bool = False) -> List[RetrievalItem]:
        items: List[RetrievalItem] = []
        seen_urls = set()

        # 配分を決定
        if add_web_search:
            iyokan_results_count = max_results // 2
            web_results_count = max_results - iyokan_results_count
        else:
            iyokan_results_count = max_results
            web_results_count = 0

        def _process_results(results, is_iyokan: bool):
            for r in results:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue

                title = r.get("title", "") or url
                raw_md = r.get("raw_content", "") or "\n".join(r.get("content", []))
                
                site_name = "いよ観ネット" if is_iyokan else url.split('/')[2].replace("www.", "")

                cleaned = self._clean_text(raw_md)
                if not cleaned:
                    try:
                        ext = self.client.extract(url)
                        cleaned = self._clean_text(ext.get("text", ""))
                    except Exception:
                        continue
                
                if cleaned:
                    items.append(RetrievalItem(
                        title=title[:180],
                        url=url,
                        site=site_name,
                        content=cleaned[:10000],
                        content_chars=len(cleaned),
                    ))
                    seen_urls.add(url)

        # 1. いよ観ネットを検索
        if iyokan_results_count > 0:
            try:
                resp_iyokan = self.client.search(
                    query=query, search_depth="advanced", include_raw_content="markdown",
                    include_answer=False, include_domains=["iyokannet.jp"],
                    max_results=iyokan_results_count, chunks_per_source=3, timeout=120
                )
                _process_results(resp_iyokan.get("results", []), is_iyokan=True)
            except Exception as e:
                print(f"Error searching iyokannet.jp: {e}")

        # 2. ウェブ全体を検索 (追加が有効な場合)
        if web_results_count > 0:
            try:
                resp_web = self.client.search(
                    query=query, search_depth="advanced", include_raw_content="markdown",
                    include_answer=False, max_results=web_results_count, 
                    chunks_per_source=3, timeout=120
                )
                _process_results(resp_web.get("results", []), is_iyokan=False)
            except Exception as e:
                print(f"Error searching web: {e}")

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
        all_vecs = []
        # Process in batches to respect API limits (e.g., 100 texts per batch)
        # and add a delay to respect rate limits (e.g., 60 requests per minute)
        batch_size = 100 
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            if not batch_texts:
                continue
            
            cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim)
            try:
                res = self.gclient.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts,
                    config=cfg,
                )
                
                batch_vecs = [np.array(e.values, dtype="float32") for e in res.embeddings]
                all_vecs.extend(batch_vecs)
                
                # If there are more batches to process, wait to avoid hitting rate limits.
                if i + batch_size < len(texts):
                    print(f"Embedding batch {i//batch_size + 1} complete. Waiting for 5 seconds...")
                    time.sleep(5)

            except Exception as e:
                print(f"An error occurred during embedding batch {i//batch_size + 1}: {e}")
                # Continue to the next batch if one fails
                continue

        if not all_vecs:
            # Handle case where all embedding calls failed
            return np.array([], dtype="float32").reshape(0, dim)

        return np.vstack(all_vecs)

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
        
        if not chunk_texts:
            return [], []

        # インデックス作成
        index, X = self._build_index(chunk_texts)
        if X.shape[0] == 0:
            return [], []
            
        q = self._embed([user_query], task_type="RETRIEVAL_QUERY")
        ids, scores = self._search_index(index, X, q, topk=k)
        
        selected = []
        used_sources = []
        seen_urls = set()

        for i, idx in enumerate(ids):
            meta = chunk_meta[idx]
            # 同一URLの過度な重複を避ける
            key = meta["url"]
            if key not in seen_urls:
                seen_urls.add(key)
                used_sources.append(meta)
            
            # 原文貼付ではなく "要点要約" を Gemini に一度かけて圧縮
            try:
                print(f"Summarizing chunk {i+1}/{len(ids)}: {meta['title']}")
                summary = self._summarize_for_context(chunk_texts[idx])
                selected.append(f"出典: {meta['title']} | {meta['url']}\n要点:\n{summary}")
                
                # 最後の1回以外は待機する
                if i < len(ids) - 1:
                    print(f"Waiting for 6 seconds to respect API rate limits...")
                    time.sleep(6)

            except Exception as e:
                print(f"Could not summarize chunk {i+1}: {e}")
                continue

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