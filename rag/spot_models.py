from __future__ import annotations

from copy import deepcopy

from pydantic import Field

from rag.models import StrictModel


class SpotChoice(StrictModel):
    spot_id: str = Field(max_length=8)
    time: str = Field(max_length=20)
    activity: str = Field(max_length=48)
    tip: str = Field(default="", max_length=60)


class CompactDayPlan(StrictModel):
    day: int = Field(ge=1)
    theme: str = Field(max_length=48)
    area: str = Field(min_length=1, max_length=40)
    spots: list[SpotChoice] = Field(min_length=1, max_length=2)
    notes: str = Field(default="", max_length=80)


class CompactDayBundle(StrictModel):
    days: list[CompactDayPlan]


COMPACT_DAY_BUNDLE_SCHEMA = CompactDayBundle.model_json_schema()


def compact_day_bundle_schema(*, expected_days: int, allowed_spot_ids: list[str]) -> dict:
    schema = deepcopy(COMPACT_DAY_BUNDLE_SCHEMA)
    schema["properties"]["days"]["minItems"] = expected_days
    schema["properties"]["days"]["maxItems"] = expected_days
    schema["$defs"]["SpotChoice"]["properties"]["spot_id"]["enum"] = allowed_spot_ids
    return schema
