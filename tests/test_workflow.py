from rag.models import DayPlan, PlanPatch, ScheduleItem
from workflow.planner import (
    apply_patch,
    itinerary_schema_for_days,
    needs_fresh_search,
    semantic_violations,
)


def _plan():
    return {
        "title": "test",
        "summary": "",
        "audience": "大人2",
        "transport": "自家用車",
        "days": [
            {
                "day": 1,
                "theme": "松山",
                "area": "松山",
                "schedule": [
                    {
                        "time": "09:00-10:00",
                        "activity": "観光",
                        "spot": "A",
                        "address": "",
                        "url": "https://example.com/a",
                        "tip": "",
                    }
                ],
                "notes": "",
                "source_urls": ["https://example.com/a"],
            },
            {
                "day": 2,
                "theme": "道後",
                "area": "松山",
                "schedule": [
                    {
                        "time": "09:00-10:00",
                        "activity": "観光",
                        "spot": "B",
                        "address": "",
                        "url": "https://example.com/b",
                        "tip": "",
                    }
                ],
                "notes": "",
                "source_urls": ["https://example.com/b"],
            },
        ],
        "sources": [
            {"title": "A", "url": "https://example.com/a", "site": "x"},
            {"title": "B", "url": "https://example.com/b", "site": "x"},
        ],
    }


def test_freshness_router():
    assert needs_fresh_search("今日の営業時間を確認して")
    assert not needs_fresh_search("道後をゆったり回りたい")


def test_semantic_validator_rejects_unknown_url():
    plan = _plan()
    plan["days"][0]["schedule"][0]["url"] = "https://hallucinated.example"
    errors = semantic_violations(
        plan,
        expected_days=2,
        allowed_urls={"https://example.com/a", "https://example.com/b"},
    )
    assert any("根拠外URL" in e for e in errors)


def test_patch_replaces_only_target_day():
    plan = _plan()
    replacement = DayPlan(
        day=2,
        theme="ゆったり道後",
        area="松山",
        schedule=[
            ScheduleItem(
                time="10:00-11:00",
                activity="散策",
                spot="C",
                address="",
                url="",
                tip="",
            )
        ],
        notes="",
        source_urls=[],
    )
    patch = PlanPatch(days=[replacement])
    updated = apply_patch(plan, patch)
    assert updated["days"][0]["theme"] == "松山"
    assert updated["days"][1]["theme"] == "ゆったり道後"



def test_day_schema_limits_schedule_size():
    assert DayPlan.model_json_schema()["properties"]["schedule"]["maxItems"] == 2



def test_itinerary_schema_limits_requested_day_count():
    schema = itinerary_schema_for_days(1)
    assert schema["properties"]["days"]["minItems"] == 1
    assert schema["properties"]["days"]["maxItems"] == 1
