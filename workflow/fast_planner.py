from __future__ import annotations

import re
import time
from copy import deepcopy
from typing import Any

from rag.models import SourceItem
from rag.retriever import RetrievalItem
from rag.spot_models import CompactDayBundle, compact_day_bundle_schema
from rag.spot_prompts import build_spot_plan_prompt, build_spot_segment_prompt
from workflow.planner import PlannerState, PlannerWorkflow


# Keep the fast planner self-contained at import time. Streamlit can hot-reload
# this module while workflow.planner is still the older in-memory module from a
# previous deploy, so importing helpers/constants newly added to planner creates
# a brittle runtime import contract even when the repository contents match.
MODEL_CONTEXT_TOKENS = 8192
SINGLE_PASS_MAX_DAYS = 4
SINGLE_PASS_SAFETY_TOKENS = 700
STRUCTURED_OUTPUT_OVERHEAD_TOKENS = 400
SEGMENT_DAYS = 3


def _timings_with(
    state: PlannerState, key: str, started: float
) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[key] = round((time.perf_counter() - started) * 1000, 1)
    return timings


def _estimate_prompt_tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return non_ascii_chars + (ascii_chars + 3) // 4


def _should_generate_single_pass(trip_days: int, prompt: str) -> bool:
    if trip_days <= 3:
        return True
    if trip_days > SINGLE_PASS_MAX_DAYS:
        return False
    # Preserve PR #15's conservative decision rule so this hotfix changes only
    # the import contract, not the one-pass safety margin.
    legacy_output_budget = 500 + 280 * trip_days
    estimated_total = (
        _estimate_prompt_tokens(prompt)
        + STRUCTURED_OUTPUT_OVERHEAD_TOKENS
        + legacy_output_budget
        + SINGLE_PASS_SAFETY_TOKENS
    )
    return estimated_total <= MODEL_CONTEXT_TOKENS


def compact_output_budget(days: int) -> int:
    return 260 + 170 * days


_EMPTY_THEME_PARENS = re.compile(r"\s*(?:\(\s*\)|（\s*）)\s*$")


def _normalize_area(value: Any) -> str:
    area = str(value or "").strip()
    return area or "愛媛県内"


def _normalize_theme(value: Any) -> str:
    theme = str(value or "").strip()
    theme = _EMPTY_THEME_PARENS.sub("", theme).strip()
    return theme or "愛媛観光"


def normalize_compact_bundle(raw: Any) -> Any:
    """Defensively normalize model output before strict schema validation."""
    if not isinstance(raw, dict):
        return raw
    cleaned = deepcopy(raw)
    days = cleaned.get("days")
    if not isinstance(days, list):
        return cleaned
    for day in days:
        if not isinstance(day, dict):
            continue
        day["area"] = _normalize_area(day.get("area"))
        day["theme"] = _normalize_theme(day.get("theme"))
    return cleaned


def _facility_keys(candidates: list[dict[str, Any]]) -> list[str]:
    generic_keys = {"愛媛", "松山", "観光", "温泉", "グルメ", "歴史"}
    keys = {
        str(candidate.get("canonical_key") or "").strip()
        for candidate in candidates
    }
    return sorted(
        (key for key in keys if 3 <= len(key) <= 12 and key not in generic_keys),
        key=len,
        reverse=True,
    )


def _mentioned_facilities(text: Any, facility_keys: list[str]) -> set[str]:
    normalized = re.sub(r"\\s+", "", str(text or "")).lower()
    return {key for key in facility_keys if key and key in normalized}


def hydrate_spot_bundle(
    bundle: CompactDayBundle,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore deterministic display/source fields from selected spot IDs."""
    by_id = {candidate["spot_id"]: candidate for candidate in candidates}
    facility_keys = _facility_keys(candidates)
    used_ids: list[str] = []
    used_facilities: set[str] = set()
    days: list[dict[str, Any]] = []

    for compact_day in bundle.days:
        schedule: list[dict[str, Any]] = []
        source_urls: list[str] = []
        for choice in compact_day.spots:
            candidate = by_id.get(choice.spot_id)
            if candidate is None:
                raise ValueError(f"未知のspot_idです: {choice.spot_id}")

            candidate_text = " ".join(
                [
                    str(candidate.get("title", "")),
                    str(candidate.get("excerpt", "")),
                    choice.activity,
                    choice.tip,
                ]
            )
            mentioned = _mentioned_facilities(candidate_text, facility_keys)
            if choice.spot_id in used_ids or mentioned & used_facilities:
                continue

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
            used_facilities.update(mentioned)

        if not schedule:
            fallback = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["spot_id"] not in used_ids
                    and not (
                        _mentioned_facilities(
                            " ".join(
                                [
                                    str(candidate.get("title", "")),
                                    str(candidate.get("excerpt", "")),
                                ]
                            ),
                            facility_keys,
                        )
                        & used_facilities
                    )
                ),
                None,
            )
            if fallback is not None:
                fallback_id = str(fallback["spot_id"])
                fallback_url = str(fallback["url"])
                schedule.append(
                    {
                        "time": "09:00-10:00",
                        "activity": "候補スポットを訪問",
                        "spot": str(fallback["title"]).strip() or fallback_id,
                        "address": "",
                        "url": fallback_url,
                        "tip": "",
                    }
                )
                source_urls.append(fallback_url)
                used_ids.append(fallback_id)
                used_facilities.update(
                    _mentioned_facilities(
                        " ".join(
                            [
                                str(fallback.get("title", "")),
                                str(fallback.get("excerpt", "")),
                            ]
                        ),
                        facility_keys,
                    )
                )

        days.append(
            {
                "day": compact_day.day,
                "theme": _normalize_theme(compact_day.theme),
                "area": _normalize_area(compact_day.area),
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

        if _should_generate_single_pass(trip_days, prompt):
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
            bundle = CompactDayBundle.model_validate(normalize_compact_bundle(raw))
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
            bundle = CompactDayBundle.model_validate(normalize_compact_bundle(raw))
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
