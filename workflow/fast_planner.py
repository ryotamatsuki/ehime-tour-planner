from __future__ import annotations

from typing import Any

from rag.models import SourceItem
from rag.retriever import RetrievalItem
from rag.spot_models import CompactDayBundle, compact_day_bundle_schema
from rag.spot_prompts import build_spot_plan_prompt, build_spot_segment_prompt
from workflow.planner import (
    SEGMENT_DAYS,
    PlannerState,
    PlannerWorkflow,
    _timings_with,
    should_generate_single_pass,
)


def compact_output_budget(days: int) -> int:
    return 260 + 170 * days


def hydrate_spot_bundle(
    bundle: CompactDayBundle,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore deterministic display/source fields from selected spot IDs."""
    by_id = {candidate["spot_id"]: candidate for candidate in candidates}
    used_ids: list[str] = []
    days: list[dict[str, Any]] = []

    for compact_day in bundle.days:
        schedule: list[dict[str, Any]] = []
        source_urls: list[str] = []
        for choice in compact_day.spots:
            candidate = by_id.get(choice.spot_id)
            if candidate is None:
                raise ValueError(f"未知のspot_idです: {choice.spot_id}")
            url = str(candidate["url"])
            title = str(candidate["title"]).strip() or choice.spot_id
            schedule.append(
                {
                    "time": choice.time,
                    "activity": choice.activity,
                    "spot": title[:48],
                    "address": "",
                    "url": url,
                    "tip": choice.tip,
                }
            )
            if url not in source_urls:
                source_urls.append(url)
            if choice.spot_id not in used_ids:
                used_ids.append(choice.spot_id)

        days.append(
            {
                "day": compact_day.day,
                "theme": compact_day.theme,
                "area": compact_day.area,
                "schedule": schedule,
                "notes": compact_day.notes,
                "source_urls": source_urls[:2],
            }
        )

    sources = [
        SourceItem(
            title=str(by_id[spot_id]["title"]),
            url=str(by_id[spot_id]["url"]),
            site=str(by_id[spot_id].get("site", "")),
        ).model_dump()
        for spot_id in used_ids
    ]
    return days, sources


class FastPlannerWorkflow(PlannerWorkflow):
    """PlannerWorkflow whose initial generation selects compact RAG spot IDs."""

    def _retrieve(self, state: PlannerState) -> dict[str, Any]:
        if state["mode"] == "refine":
            return super()._retrieve(state)

        import time

        started = time.perf_counter()
        items = [RetrievalItem(**item) for item in state.get("items", [])]
        candidate_limit = min(10, max(6, state["trip_days"] * 2))
        context, candidates = self.retriever.retrieve_spot_candidates(
            items=items,
            user_query=state["query"],
            candidate_limit=candidate_limit,
        )
        if not context or not candidates:
            raise RuntimeError("関連情報を取得できませんでした。検索条件を変えてください。")
        return {
            "context": context,
            # PlannerState already carries sources as arbitrary dictionaries.
            # During initial generation we temporarily store candidate metadata
            # here; the final Itinerary sources are rebuilt from selected IDs.
            "sources": candidates,
            "timings": _timings_with(state, "retrieval_ms", started),
        }

    def _generate_initial(self, state: PlannerState) -> dict[str, Any]:
        candidates = [source for source in state["sources"] if source.get("spot_id")]
        if not candidates:
            return super()._generate_initial(state)

        trip_days = state["trip_days"]
        prompt = build_spot_plan_prompt(
            trip_days=trip_days,
            start_date=state["start_date"],
            party=state["party"],
            transport=state["transport"],
            interests=state["interests"],
            start_area=state["start_area"],
            with_kids=state["with_kids"],
            pace=state["pace"],
            start_end_point=state["start_end_point"],
            candidate_context=state["context"],
        )

        if should_generate_single_pass(trip_days, prompt):
            raw = self.llm.generate_json(
                prompt=prompt,
                schema=compact_day_bundle_schema(
                    expected_days=trip_days,
                    allowed_spot_ids=[c["spot_id"] for c in candidates],
                ),
                schema_name=f"spot_days_1_{trip_days}",
                max_tokens=compact_output_budget(trip_days),
                temperature=0,
            )
            bundle = CompactDayBundle.model_validate(raw)
            expected = list(range(1, trip_days + 1))
            actual = [day.day for day in bundle.days]
            if actual != expected:
                raise ValueError(
                    f"旅程のday番号が不正です。expected={expected}, actual={actual}"
                )
            days, sources = hydrate_spot_bundle(bundle, candidates)
            strategy = f"spot_id_single_pass_{trip_days}d"
        else:
            days, sources = self._generate_spot_segmented(state, candidates)
            strategy = f"spot_id_segmented_{trip_days}d"

        return {
            "result": {
                "title": f"愛媛 {trip_days}日間プラン",
                "summary": f"{state['party']}向けの{state['pace']}な愛媛旅行プランです。",
                "audience": state["party"],
                "transport": state["transport"],
                "days": days,
                "sources": sources,
            },
            "generation_strategy": strategy,
        }

    def _generate_spot_segmented(
        self,
        state: PlannerState,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_days: list[dict[str, Any]] = []
        source_by_url: dict[str, dict[str, Any]] = {}
        previous_spot_id: str | None = None
        trip_days = state["trip_days"]
        allowed_ids = [candidate["spot_id"] for candidate in candidates]

        for start_day in range(1, trip_days + 1, SEGMENT_DAYS):
            end_day = min(trip_days, start_day + SEGMENT_DAYS - 1)
            segment_days = end_day - start_day + 1
            prompt = build_spot_segment_prompt(
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
                candidate_context=state["context"],
                previous_spot_id=previous_spot_id,
            )
            raw = self.llm.generate_json(
                prompt=prompt,
                schema=compact_day_bundle_schema(
                    expected_days=segment_days,
                    allowed_spot_ids=allowed_ids,
                ),
                schema_name=f"spot_days_{start_day}_{end_day}",
                max_tokens=compact_output_budget(segment_days),
                temperature=0,
            )
            bundle = CompactDayBundle.model_validate(raw)
            expected = list(range(start_day, end_day + 1))
            actual = [day.day for day in bundle.days]
            if actual != expected:
                raise ValueError(
                    f"分割旅程のday番号が不正です。expected={expected}, actual={actual}"
                )
            segment_full_days, segment_sources = hydrate_spot_bundle(bundle, candidates)
            all_days.extend(segment_full_days)
            for source in segment_sources:
                source_by_url[source["url"]] = source
            if bundle.days and bundle.days[-1].spots:
                previous_spot_id = bundle.days[-1].spots[-1].spot_id

        return all_days, list(source_by_url.values())
