from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rag.fast_retriever import CachedSpotRetriever, _CorpusIndex, canonical_spot_key
from rag.retriever import Chunk, RetrievalItem
from rag.spot_models import CompactDayBundle, CompactDayPlan, SpotChoice
from utils.formatting import plan_json_to_markdown
from workflow.fast_planner import hydrate_spot_bundle, normalize_compact_bundle


def test_canonical_spot_key_collapses_matsuyama_castle_title_variants():
    assert canonical_spot_key("松山城観光") == "松山城"
    assert canonical_spot_key("松山城 - 愛媛県") == "松山城"
    assert canonical_spot_key("松山城観光") == canonical_spot_key("松山城 - 愛媛県")


def test_spot_candidate_retrieval_deduplicates_equivalent_facility_pages(monkeypatch):
    retriever = CachedSpotRetriever(api_key=None)
    chunks = [
        Chunk(text="松山城の観光案内", title="松山城観光", url="https://example.test/a", site="test"),
        Chunk(text="松山城の施設情報", title="松山城 - 愛媛県", url="https://example.test/b", site="test"),
        Chunk(text="道後温泉の観光案内", title="道後温泉", url="https://example.test/c", site="test"),
    ]
    index = _CorpusIndex(
        chunks=chunks,
        bm25_tokens=[["x"], ["x"], ["x"]],
        doc_vecs=np.zeros((3, 1), dtype=float),
    )
    monkeypatch.setattr(
        retriever,
        "_get_or_build_index",
        lambda items: (index, True, 0.0),
    )
    monkeypatch.setattr(retriever, "_rank", lambda index, query: ([0, 1, 2], 0.0))

    context, candidates = retriever.retrieve_spot_candidates(
        items=[
            RetrievalItem(
                title="dummy",
                url="https://example.test/source",
                site="test",
                content="dummy content",
                content_chars=13,
            )
        ],
        user_query="松山",
        candidate_limit=3,
    )

    assert len(context) == 2
    assert [candidate["title"] for candidate in candidates] == ["松山城観光", "道後温泉"]
    assert len({candidate["canonical_key"] for candidate in candidates}) == 2


def test_compact_day_area_must_not_be_empty():
    with pytest.raises(ValidationError):
        CompactDayPlan(
            day=1,
            theme="城と温泉",
            area="",
            spots=[
                SpotChoice(
                    spot_id="S001",
                    time="09:00-10:00",
                    activity="見学",
                    tip="",
                )
            ],
            notes="",
        )


def test_markdown_never_renders_empty_area_parentheses():
    markdown = plan_json_to_markdown(
        {
            "title": "test",
            "days": [
                {
                    "day": 1,
                    "theme": "城めぐり",
                    "area": "",
                    "schedule": [],
                    "source_urls": [],
                }
            ],
            "sources": [],
        }
    )
    assert "#### Day 1: 城めぐり" in markdown
    assert "城めぐり ()" not in markdown


def test_modal_backend_defaults_to_scale_to_zero_fast_boot_with_rollback():
    source = Path("modal_backend.py").read_text(encoding="utf-8")
    assert "scaledown_window=20 * MINUTES" in source
    assert '_env_flag("VLLM_FAST_BOOT", True)' in source
    assert 'os.getenv("VLLM_ENFORCE_EAGER") is not None' in source
    assert 'cmd.append("--enforce-eager")' in source
    assert 'cmd.append("--no-enforce-eager")' in source


def test_compact_bundle_normalizes_blank_area_and_empty_theme_parentheses():
    normalized = normalize_compact_bundle(
        {
            "days": [
                {
                    "day": 1,
                    "theme": "城めぐり ()",
                    "area": "   ",
                    "spots": [],
                }
            ]
        }
    )
    assert normalized["days"][0]["area"] == "愛媛県内"
    assert normalized["days"][0]["theme"] == "城めぐり"


def test_hydrate_spot_bundle_suppresses_cross_day_facility_repeats():
    bundle = CompactDayBundle.model_validate(
        {
            "days": [
                {
                    "day": 1,
                    "theme": "城めぐり",
                    "area": "松山",
                    "spots": [
                        {
                            "spot_id": "S001",
                            "time": "09:00-10:00",
                            "activity": "松山城を見学",
                            "tip": "",
                        }
                    ],
                    "notes": "",
                },
                {
                    "day": 2,
                    "theme": "別の地域",
                    "area": "愛媛県内",
                    "spots": [
                        {
                            "spot_id": "S002",
                            "time": "09:00-10:00",
                            "activity": "松山城の紅葉を楽しむ",
                            "tip": "",
                        }
                    ],
                    "notes": "",
                },
            ]
        }
    )
    candidates = [
        {
            "spot_id": "S001",
            "title": "松山城｜公式情報",
            "url": "https://example.test/castle",
            "site": "test",
            "excerpt": "松山城の案内",
            "canonical_key": "松山城",
        },
        {
            "spot_id": "S002",
            "title": "秋の愛媛モデルコース",
            "url": "https://example.test/autumn",
            "site": "test",
            "excerpt": "松山城の紅葉",
            "canonical_key": "秋愛媛モデルコース",
        },
        {
            "spot_id": "S003",
            "title": "道後温泉",
            "url": "https://example.test/dogo",
            "site": "test",
            "excerpt": "道後温泉の案内",
            "canonical_key": "道後温泉",
        },
    ]

    days, _ = hydrate_spot_bundle(bundle, candidates)

    assert days[0]["schedule"][0]["spot"].startswith("松山城")
    assert days[1]["schedule"][0]["url"] == "https://example.test/dogo"
    assert days[1]["schedule"][0]["activity"] == "候補スポットを訪問"
