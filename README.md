# Ehime Tour Planner — Hybrid RAG × Sarashina

愛媛県内の観光情報を検索し、根拠URLを保持したまま旅程を作るPoCです。

## v2 architecture

Gemini APIへの依存を外し、小型日本語LLMでも安定するように処理を明示的なWorkflowへ分解しています。

```text
Streamlit
  ↓
LangGraph workflow
  ├─ 必要時のみ Tavily
  ├─ BM25
  ├─ Ruri-v3-30m dense retrieval
  ├─ Reciprocal Rank Fusion
  ↓
Modal / vLLM
  ↓
Sarashina2.2-3B-Instruct
  ├─ JSON Schema constrained output
  └─ plan patch generation
  ↓
Pydantic + semantic validation
```

### Design principles

- **Retrieval と Generation を分離**: RetrievalはBM25 + Ruri、GenerationはSarashina。
- **LLM要約を検索前に挟まない**: Web本文をチャンク化し、Hybrid RAGで必要部分だけLLMへ渡す。
- **Multi-Agent化しない**: LangGraphは決定的Workflowとして使用し、3Bモデルにツール選択を丸投げしない。
- **長期旅程を分割**: 1〜3日は一括、4日以上は最大3日単位で生成。
- **修正はPatch方式**: 「2日目をゆったり」なら2日目だけ再生成。
- **JSONを後処理で祈らない**: vLLM Structured OutputsでJSON Schemaを制約し、Pydanticで再検証。
- **URL hallucinationを抑制**: 検索で得たURL以外はPython側で除去。
- **無料枠を意識**: Tavilyは既定でbasic search。Ruri/BM25はStreamlit側CPUで実行。Modalはscale-to-zero。

## Main components

| Component | Role |
|---|---|
| Streamlit | UI |
| LangGraph | Workflow / state management |
| Tavily | いよ観ネット中心のWeb retrieval |
| Ruri-v3-30m | Japanese dense embedding |
| BM25 | Exact-name / keyword retrieval |
| RRF | Dense + sparse rank fusion |
| Modal | GPU serving |
| vLLM | OpenAI-compatible Sarashina serving / Structured Outputs |
| Sarashina2.2-3B-Instruct | Itinerary generation |
| Pydantic | Structural + semantic validation |

## Local / Streamlit setup

Install:

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` locally (this file is gitignored):

```toml
TAVILY_API_KEY = "tvly-..."
SARASHINA_BASE_URL = "https://<workspace>--ehime-tour-planner-sarashina-server.modal.direct"
SARASHINA_API_KEY = "<VLLM_API_KEY>"
SARASHINA_MODEL = "sarashina"
```

Then:

```bash
streamlit run app.py
```

## Modal deployment

Install Modal CLI:

```bash
pip install -r requirements-modal.txt
modal setup
```

Create a Modal secret named `ehime-tour-planner-vllm` containing:

```text
VLLM_API_KEY=<random-long-secret>
```

Deploy:

```bash
modal deploy modal_backend.py
```

The deployment outputs a `modal.direct` URL. Set that URL as `SARASHINA_BASE_URL` in Streamlit Secrets and use the same `VLLM_API_KEY` as `SARASHINA_API_KEY`.

A live smoke test can be run with:

```bash
VLLM_API_KEY=<same-secret> modal run modal_backend.py
```

## Cost controls

- `TAVILY_SEARCH_DEPTH=basic` is the default.
- The embedding model is local CPU inference (`cl-nagoya/ruri-v3-30m`).
- Modal uses one T4 and `scaledown_window=60`.
- No LangSmith service is required.
- No Gemini API key is required.

## Safety / grounding

The planner is a recommendation aid. Opening hours, closure days, fares, transport disruptions, weather and other fresh facts should be checked at the cited source before travel.
