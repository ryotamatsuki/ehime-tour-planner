from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from llm.sarashina_client import SarashinaClient
from rag.models import (
    DAY_BUNDLE_SCHEMA,
    ITINERARY_SCHEMA,
    PLAN_PATCH_SCHEMA,
    DayBundle,
    Itinerary,
    PlanPatch,
    SourceItem,
)
from rag.prompts import (
    build_plan_prompt,
    build_refine_patch_prompt,
    build_repair_prompt,
    build_segment_prompt,
)
from rag.retriever import EhimeRetriever, RetrievalItem


FRESHNESS_TERMS = (
    "今日",
    "明日",
    "今週",
    "現在",
    "最新",
    "営業時間",
    "休館",
    "休業",
    "運休",
    "天気",
    "見頃",
    "開催中",
    "イベント",
    "混雑",
)


class PlannerState(TypedDict, total=False):
    mode: Literal["plan", "refine"]
    query: str
    trip_days: int
    start_date: str
    party: str
    transport: str
    interests: list[str]
    start_area: str
    with_kids: bool
    pace: str
    start_end_point: str
    add_web_search: bool
    max_results: int
    items: list[dict[str, Any]]
    existing_plan: dict[str, Any]
    context: list[str]
    sources: list[dict[str, Any]]
    result: dict[str, Any]
    violations: list[str]
    repair_count: int


def needs_fresh_search(text: str) -> bool:
    return any(term in text for term in FRESHNESS_TERMS)


def semantic_violations(plan: dict[str, Any], expected_days: int, allowed_urls: set[str]) -> list[str]:
    violations: list[str] = []
    days = plan.get("days", [])
    actual = [d.get("day") for d in days]
    expected = list(range(1, expected_days + 1))
    if actual != expected:
        violations.append(f"day番号は {expected} である必要があります。現在: {actual}")

    for day in days:
        day_no = day.get("day")
        schedule = day.get("schedule") or []
        if not schedule:
            violations.append(f"Day {day_no} の schedule が空です。")
        for url in day.get("source_urls", []):
            if url and url not in allowed_urls:
                violations.append(f"Day {day_no} に根拠外URLがあります: {url}")
        for entry in schedule:
            url = entry.get("url", "")
            if url and url not in allowed_urls:
                violations.append(f"Day {day_no} の schedule に根拠外URLがあります: {url}")

    return violations


def sanitize_urls(plan: dict[str, Any], allowed_urls: set[str]) -> dict[str, Any]:
    cleaned = deepcopy(plan)
    cleaned["sources"] = [
        s for s in cleaned.get("sources", []) if s.get("url") in allowed_urls
    ]
    for day in cleaned.get("days", []):
        day["source_urls"] = [
            u for u in day.get("source_urls", []) if u in allowed_urls
        ]
        for entry in day.get("schedule", []):
            if entry.get("url") and entry["url"] not in allowed_urls:
                entry["url"] = ""
    return cleaned


def apply_patch(plan: dict[str, Any], patch: PlanPatch) -> dict[str, Any]:
    updated = deepcopy(plan)
    if patch.title:
        updated["title"] = patch.title
    if patch.summary:
        updated["summary"] = patch.summary

    day_map = {int(d["day"]): d for d in updated.get("days", [])}
    for day in patch.days:
        day_map[day.day] = day.model_dump()
    updated["days"] = [day_map[k] for k in sorted(day_map)]
    return updated


def itinerary_schema_for_days(expected_days: int) -> dict[str, Any]:
    schema = deepcopy(ITINERARY_SCHEMA)
    schema["properties"]["days"]["minItems"] = expected_days
    schema["properties"]["days"]["maxItems"] = expected_days
    return schema


def day_bundle_schema_for_days(expected_days: int) -> dict[str, Any]:
    """Return a compact schema for short plans.

    Metadata and the full source list are deterministic application data, so
    asking the small model to emit them wastes output tokens.  Keeping only
    the requested days in the structured response prevents 2- and 3-day
    plans from being truncated at vLLM's context/output boundary.
    """
    schema = deepcopy(DAY_BUNDLE_SCHEMA)
    schema["properties"]["days"]["minItems"] = expected_days
    schema["properties"]["days"]["maxItems"] = expected_days
    return schema


def day_bundle_schema_for_range(start_day: int, end_day: int) -> dict[str, Any]:
    """Return a DayBundle schema constrained to one generated segment."""
    return day_bundle_schema_for_days(end_day - start_day + 1)


def trim_extra_days(plan: dict[str, Any], expected_days: int) -> dict[str, Any]:
    cleaned = deepcopy(plan)
    days = [d for d in cleaned.get("days", []) if d.get("day", 0) <= expected_days]
    cleaned["days"] = sorted(days, key=lambda d: d.get("day", 0))[:expected_days]
    return cleaned


class PlannerWorkflow:
    def __init__(self, retriever: EhimeRetriever, llm: SarashinaClient):
        self.retriever = retriever
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PlannerState)
        graph.add_node("research", self._research)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_node("repair", self._repair)

        graph.add_edge(START, "research")
        graph.add_edge("research", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"done": END, "repair": "repair"},
        )
        graph.add_edge("repair", "validate")
        return graph.compile()

    def run_plan(self, **kwargs) -> PlannerState:
        state: PlannerState = {"mode": "plan", "repair_count": 0, **kwargs}
        return self.graph.invoke(state)

    def run_refine(self, **kwargs) -> PlannerState:
        state: PlannerState = {"mode": "refine", "repair_count": 0, **kwargs}
        return self.graph.invoke(state)

    def _research(self, state: PlannerState) -> dict[str, Any]:
        items = state.get("items", [])
        must_search = not items
        if state["mode"] == "refine":
            must_search = must_search or needs_fresh_search(state.get("query", ""))

        if must_search:
            found = self.retriever.search_and_prepare(
                query=state["query"],
                max_results=state.get("max_results", 8),
                add_web_search=state.get("add_web_search", False),
            )
            items = [i.model_dump() for i in found]
        return {"items": items}

    def _retrieve(self, state: PlannerState) -> dict[str, Any]:
        items = [RetrievalItem(**i) for i in state.get("items", [])]
        context, sources = self.retriever.retrieve_for_plan(
            items=items,
            user_query=state["query"],
            # Keep enough evidence for grounding while leaving headroom under
            # Sarashina/vLLM's 8,192-token context limit.
            k=6,
        )
        if not context:
            raise RuntimeError("関連情報を取得できませんでした。検索条件を変えてください。")
        return {"context": context, "sources": sources}

    def _generate(self, state: PlannerState) -> dict[str, Any]:
        if state["mode"] == "refine":
            return self._generate_refine(state)
        return self._generate_initial(state)

    def _generate_initial(self, state: PlannerState) -> dict[str, Any]:
        trip_days = state["trip_days"]
        if trip_days <= 3:
            prompt = build_plan_prompt(
                trip_days=trip_days,
                start_date=state["start_date"],
                party=state["party"],
                transport=state["transport"],
                interests=state["interests"],
                start_area=state["start_area"],
                with_kids=state["with_kids"],
                pace=state["pace"],
                start_end_point=state["start_end_point"],
                context=state["context"],
            )
            raw = self.llm.generate_json(
                prompt=prompt,
                schema=day_bundle_schema_for_days(trip_days),
                schema_name=f"days_1_{trip_days}",
                # vLLMの8192上限に対し、RAGコンテキストが最大5793 tokens
                # になるため、出力は日程部分だけにして余裕を確保する。
                max_tokens=500 + 300 * trip_days,
            )
            bundle = DayBundle.model_validate(raw)
            expected = list(range(1, trip_days + 1))
            actual = [day.day for day in bundle.days]
            if actual != expected:
                raise ValueError(
                    f"旅程のday番号が不正です。expected={expected}, actual={actual}"
                )
            plan = {
                "title": f"愛媛 {trip_days}日間プラン",
                "summary": f"{state['party']}向けの{state['pace']}な愛媛旅行プランです。",
                "audience": state["party"],
                "transport": state["transport"],
                "days": [day.model_dump() for day in bundle.days],
                "sources": state["sources"],
            }
        else:
            plan = self._generate_segmented(state)

        allowed = {s["url"] for s in state["sources"]}
        plan = sanitize_urls(plan, allowed)
        plan["sources"] = [SourceItem(**s).model_dump() for s in state["sources"]]
        return {"result": plan}

    def _generate_segmented(self, state: PlannerState) -> dict[str, Any]:
        trip_days = state["trip_days"]
        all_days: list[dict[str, Any]] = []
        previous_day: dict[str, Any] | None = None
        for start_day in range(1, trip_days + 1, 3):
            end_day = min(trip_days, start_day + 2)
            prompt = build_segment_prompt(
                start_day=start_day,
                end_day=end_day,
                trip_days=trip_days,
                start_date=state["start_date"],
                party=state["party"],
                transport=state["transport"],
                interests=state["interests"],
                start_area=state["start_area"],
                with_kids=state["with_kids"],
                pace=state["pace"],
                start_end_point=state["start_end_point"],
                previous_day=previous_day,
                context=state["context"],
            )
            raw = self.llm.generate_json(
                prompt=prompt,
                schema=day_bundle_schema_for_range(start_day, end_day),
                schema_name=f"days_{start_day}_{end_day}",
                # Keep each segment below the point where a T4 request is
                # likely to be drained or duplicated during cold-start retry.
                max_tokens=500 + 350 * (end_day - start_day + 1),
            )
            bundle = DayBundle.model_validate(raw)
            expected = list(range(start_day, end_day + 1))
            actual = [d.day for d in bundle.days]
            if actual != expected:
                raise ValueError(f"分割旅程のday番号が不正です。expected={expected}, actual={actual}")
            dumped = [d.model_dump() for d in bundle.days]
            all_days.extend(dumped)
            previous_day = dumped[-1]

        return {
            "title": f"愛媛 {trip_days}日間プラン",
            "summary": f"{state['party']}向けの{state['pace']}な愛媛旅行プランです。",
            "audience": state["party"],
            "transport": state["transport"],
            "days": all_days,
            "sources": state["sources"],
        }

    def _generate_refine(self, state: PlannerState) -> dict[str, Any]:
        prompt = build_refine_patch_prompt(
            existing_plan=state["existing_plan"],
            user_request=state["query"],
            context=state["context"],
        )
        raw = self.llm.generate_json(
            prompt=prompt,
            schema=PLAN_PATCH_SCHEMA,
            schema_name="plan_patch",
            max_tokens=1800,
        )
        patch = PlanPatch.model_validate(raw)
        merged = apply_patch(state["existing_plan"], patch)
        allowed = {s["url"] for s in state["sources"]}
        allowed.update(s.get("url", "") for s in state["existing_plan"].get("sources", []))
        merged = sanitize_urls(merged, allowed)

        source_by_url = {
            s.get("url", ""): s for s in state["existing_plan"].get("sources", [])
            if s.get("url")
        }
        for source in state["sources"]:
            source_by_url[source["url"]] = source
        merged["sources"] = list(source_by_url.values())
        return {"result": merged}

    def _validate(self, state: PlannerState) -> dict[str, Any]:
        result = state["result"]
        try:
            validated = Itinerary.model_validate(result).model_dump()
        except ValidationError as exc:
            return {"violations": [str(exc)]}

        allowed = {s["url"] for s in state.get("sources", [])}
        allowed.update(s.get("url", "") for s in validated.get("sources", []))
        violations = semantic_violations(validated, state["trip_days"], allowed)
        return {"result": validated, "violations": violations}

    def _route_after_validate(self, state: PlannerState) -> Literal["done", "repair"]:
        if not state.get("violations"):
            return "done"
        if state.get("repair_count", 0) >= 1:
            raise ValueError("旅程の自動修復後も検証エラーが残りました: " + " / ".join(state["violations"]))
        return "repair"

    def _repair(self, state: PlannerState) -> dict[str, Any]:
        prompt = build_repair_prompt(
            invalid_plan=state["result"],
            violations=state.get("violations", []),
            context=state["context"],
        )
        raw = self.llm.generate_json(
            prompt=prompt,
            schema=itinerary_schema_for_days(state["trip_days"]),
            schema_name="itinerary_repair",
            max_tokens=2000,
        )
        plan = trim_extra_days(
            Itinerary.model_validate(raw).model_dump(), state["trip_days"]
        )
        allowed = {s["url"] for s in state["sources"]}
        plan = sanitize_urls(plan, allowed)
        plan["sources"] = [SourceItem(**s).model_dump() for s in state["sources"]]
        return {"result": plan, "repair_count": state.get("repair_count", 0) + 1}
