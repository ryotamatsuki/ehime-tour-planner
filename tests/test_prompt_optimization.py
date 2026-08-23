from rag.prompts import build_plan_prompt, build_segment_prompt


def _kwargs():
    return {
        "trip_days": 4,
        "start_date": "2026-08-24",
        "party": "大人2",
        "transport": "自家用車",
        "interests": ["温泉"],
        "start_area": "中予",
        "with_kids": False,
        "pace": "標準",
        "start_end_point": "松山空港",
        "context": ["出典: A\nURL: https://example.com/a\n内容:\n道後温泉"],
    }


def test_plan_prompt_matches_day_bundle_schema():
    prompt = build_plan_prompt(**_kwargs())
    assert "DayBundle JSONだけを返し" in prompt
    assert "title/summary/audience/transport/sources" not in prompt


def test_segment_prompt_places_shared_context_before_variable_range():
    kwargs = _kwargs()
    prompt = build_segment_prompt(
        **kwargs,
        start_day=1,
        end_day=3,
        previous_day=None,
    )
    assert prompt.index("【検索コンテキスト】") < prompt.index("【今回の区間】")


def test_segment_prompt_compacts_previous_day_json():
    kwargs = _kwargs()
    prompt = build_segment_prompt(
        **kwargs,
        start_day=4,
        end_day=4,
        previous_day={"day": 3, "theme": "道後"},
    )
    assert '{"day":3,"theme":"道後"}' in prompt
