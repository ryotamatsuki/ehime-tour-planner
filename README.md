# Ehime Tour Planner (RAG)

- **データ方針**: いよ観ネットのページ本文をそのまま掲載せず、要約とパラフレーズを行い、**元URLへの導線**を示す。
- **検索**: Tavily Search `include_domains=['iyokannet.jp']` でドメイン限定。
- **埋め込み**: `gemini-embedding-001` (output_dimensionality=768, task_type=RETRIEVAL_*)
- **生成**: `gemini-2.5-flash`（Structured Output JSON）。

## 開発のヒント
- FAISS が使えない場合は `retriever.py` の `USE_FAISS=False` に設定。
- 高頻度利用時は Tavily の `chunks_per_source` を 1〜2 に抑えて API クレジット消費を節約。
- 検索クエリは「エリア + テーマ + 季節」(例: `松山 温泉 家族 春 モデルコース`) が有効。