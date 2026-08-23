from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceItem(StrictModel):
    title: str
    url: str
    site: str = ""


class ScheduleItem(StrictModel):
    time: str = Field(
        max_length=20, description="開始時刻または時間帯。例: 09:00-10:00"
    )
    activity: str = Field(max_length=48)
    spot: str = Field(max_length=48)
    address: str = Field(default="", max_length=80)
    url: str = Field(default="", max_length=500)
    tip: str = Field(default="", max_length=60)


class DayPlan(StrictModel):
    day: int = Field(ge=1)
    theme: str = Field(max_length=48)
    area: str = Field(default="", max_length=40)
    # Bound both itinerary stops and source references so small-model
    # structured output cannot grow without limit.
    schedule: list[ScheduleItem] = Field(min_length=1, max_length=2)
    notes: str = Field(default="", max_length=80)
    source_urls: list[str] = Field(default_factory=list, max_length=2)


class Itinerary(StrictModel):
    title: str
    summary: str = ""
    audience: str
    transport: str
    days: list[DayPlan]
    sources: list[SourceItem]


class DayBundle(StrictModel):
    days: list[DayPlan]


class PlanPatch(StrictModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    days: list[DayPlan] = Field(default_factory=list)


ITINERARY_SCHEMA = Itinerary.model_json_schema()
DAY_BUNDLE_SCHEMA = DayBundle.model_json_schema()
PLAN_PATCH_SCHEMA = PlanPatch.model_json_schema()
