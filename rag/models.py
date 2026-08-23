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
    time: str = Field(description="開始時刻または時間帯。例: 09:00-10:00")
    activity: str
    spot: str
    address: str = ""
    url: str = ""
    tip: str = ""


class DayPlan(StrictModel):
    day: int = Field(ge=1)
    theme: str
    area: str = ""
    # 8,192-token contextで小型モデルのJSONが途中切れしないよう、
    # 1日あたりの候補数をSchemaでも上限固定する。
    schedule: list[ScheduleItem] = Field(min_length=1, max_length=2)
    notes: str = ""
    source_urls: list[str] = Field(default_factory=list)


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
