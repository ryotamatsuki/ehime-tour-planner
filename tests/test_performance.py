from rag.models import DayPlan
from workflow.planner import (
    estimate_prompt_tokens,
    should_generate_single_pass,
    single_pass_output_budget,
)


def test_day_schema_limits_source_urls():
    schema = DayPlan.model_json_schema()
    assert schema["properties"]["source_urls"]["maxItems"] == 2


def test_prompt_token_estimate_counts_japanese_conservatively():
    assert estimate_prompt_tokens("愛媛abcde") == 4


def test_four_day_plan_can_use_single_pass_when_prompt_fits():
    assert should_generate_single_pass(4, "愛媛旅行" * 100)
    assert single_pass_output_budget(4) == 1620


def test_four_day_plan_falls_back_when_prompt_is_too_large():
    assert not should_generate_single_pass(4, "愛" * 7000)


def test_five_day_plan_remains_segmented():
    assert not should_generate_single_pass(5, "短いプロンプト")
