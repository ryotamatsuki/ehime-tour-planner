from __future__ import annotations

import numpy as np

from rag.fast_retriever import CachedSpotRetriever
from rag.retriever import RetrievalItem
from rag.spot_models import CompactDayBundle, compact_day_bundle_schema
from rag.spot_prompts import build_spot_plan_prompt
from workflow.fast_planner import FastPlannerWorkflow, hydrate_spot_bundle


class FakeEmbeddingModel:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        del kwargs
        rows = list(texts)
        self.calls.append(rows)
        return np.asarray(
            [[1.0, float((sum(map(ord, text)) % 17) + 1)] for text in rows],
            dtype=float,
        )


def _items():
    return [
        RetrievalItem(
            title="道後温泉",
            url="https://example.com/dogo",
            site="example.com",
            content="道後温泉の観光情報です。" * 40,
            content_chars=400,
        ),
        RetrievalItem(
            title="松山城",
            url="https://example.com/castle",
            site="example.com",
            content="松山城の観光情報です。" * 40,
            content_chars=400,
        ),
    ]


def test_document_embeddings_are_reused_for_same_corpus():
    retriever = CachedSpotRetriever(api_key=None)
    fake = FakeEmbeddingModel()
    retriever._embedding_model = fake

    retriever.retrieve_spot_candidates(
        items=_items(), user_query="温泉", candidate_limit=2
    )
    first_metrics = dict(retriever.last_metrics)
    retriever.retrieve_spot_candidates(
        items=_items(), user_query="城", candidate_limit=2
    )
    second_metrics = dict(retriever.last_metrics)

    document_calls = [call for call in fake.calls if call and call[0].startswith("検索文書:")]
    query_calls = [call for call in fake.calls if call and call[0].startswith("検索クエリ:")]
    assert len(document_calls) == 1
    assert len(query_calls) == 2
    assert first_metrics["cache_hit"] is False
    assert second_metrics["cache_hit"] is True
    assert second_metrics["doc_embedding_ms"] == 0.0


def test_compact_schema_restricts_days_and_spot_ids():
    schema = compact_day_bundle_schema(
        expected_days=2, allowed_spot_ids=["S001", "S002"]
    )
    assert schema["properties"]["days"]["minItems"] == 2
    assert schema["properties"]["days"]["maxItems"] == 2
    assert schema["$defs"]["SpotChoice"]["properties"]["spot_id"]["enum"] == [
        "S001",
        "S002",
    ]


def test_hydration_restores_url_without_llm_generated_metadata():
    candidates = [
        {
            "spot_id": "S001",
            "title": "道後温泉",
            "url": "https://example.com/dogo",
            "site": "example.com",
            "excerpt": "温泉",
        }
    ]
    bundle = CompactDayBundle.model_validate(
        {
            "days": [
                {
                    "day": 1,
                    "theme": "温泉",
                    "area": "松山",
                    "spots": [
                        {
                            "spot_id": "S001",
                            "time": "10:00-11:00",
                            "activity": "温泉街を散策",
                            "tip": "歩きやすい靴",
                        }
                    ],
                    "notes": "",
                }
            ]
        }
    )
    days, sources = hydrate_spot_bundle(bundle, candidates)
    stop = days[0]["schedule"][0]
    assert stop["spot"] == "道後温泉"
    assert stop["url"] == "https://example.com/dogo"
    assert stop["address"] == ""
    assert days[0]["source_urls"] == ["https://example.com/dogo"]
    assert sources[0]["url"] == "https://example.com/dogo"


def test_spot_prompt_does_not_require_urls_or_addresses():
    prompt = build_spot_plan_prompt(
        trip_days=1,
        start_date="2026-08-24",
        party="大人2",
        transport="自家用車",
        interests=["温泉"],
        start_area="松山",
        with_kids=False,
        pace="標準",
        start_end_point="松山空港",
        candidate_context=["S001 | 道後温泉 | example.com | 温泉街を楽しめる"],
    )
    assert "https://" not in prompt
    assert "URL、住所" in prompt


class FakeSpotRetriever:
    def __init__(self):
        self.last_metrics = {"cache_hit": True}

    def retrieve_spot_candidates(self, *, items, user_query, candidate_limit):
        del items, user_query, candidate_limit
        candidate = {
            "spot_id": "S001",
            "title": "道後温泉",
            "url": "https://example.com/dogo",
            "site": "example.com",
            "excerpt": "温泉街",
        }
        return ["S001 | 道後温泉 | example.com | 温泉街"], [candidate]


class FakeSpotLLM:
    def __init__(self):
        self.last_request_metrics = {"elapsed_ms": 10.0}

    def generate_json(self, *, prompt, schema, schema_name, max_tokens, temperature):
        del prompt, schema, schema_name, max_tokens, temperature
        return {
            "days": [
                {
                    "day": 1,
                    "theme": "道後",
                    "area": "松山",
                    "spots": [
                        {
                            "spot_id": "S001",
                            "time": "10:00-11:00",
                            "activity": "温泉街を散策",
                            "tip": "",
                        }
                    ],
                    "notes": "",
                },
                {
                    "day": 2,
                    "theme": "道後",
                    "area": "松山",
                    "spots": [
                        {
                            "spot_id": "S001",
                            "time": "10:00-11:00",
                            "activity": "周辺を散策",
                            "tip": "",
                        }
                    ],
                    "notes": "",
                },
            ]
        }


def test_fast_workflow_returns_full_itinerary_from_spot_ids():
    workflow = FastPlannerWorkflow(retriever=FakeSpotRetriever(), llm=FakeSpotLLM())
    state = workflow.run_plan(
        query="温泉",
        trip_days=2,
        start_date="2026-08-24",
        party="大人2",
        transport="自家用車",
        interests=["温泉"],
        start_area="松山",
        with_kids=False,
        pace="標準",
        start_end_point="指定なし",
        add_web_search=False,
        max_results=8,
        items=[
            {
                "title": "道後温泉",
                "url": "https://example.com/dogo",
                "site": "example.com",
                "content": "温泉街の情報" * 20,
                "content_chars": 120,
            }
        ],
    )
    assert state["generation_strategy"] == "spot_id_single_pass_2d"
    assert state["result"]["days"][0]["schedule"][0]["url"] == "https://example.com/dogo"
    assert state["result"]["sources"][0]["url"] == "https://example.com/dogo"
    assert "total_ms" in state["timings"]
