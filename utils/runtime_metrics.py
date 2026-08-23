from __future__ import annotations

from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    """Return a defensive dict copy for telemetry values."""
    try:
        return dict(value or {})
    except (TypeError, ValueError):
        return {}


def ensure_workflow_metrics_compatibility(workflow: Any) -> None:
    """Make metrics collection safe across Streamlit hot-reload generations.

    Streamlit can keep a cached workflow/client instance alive while reloading
    newer application code. Older SarashinaClient instances did not expose
    ``last_request_metrics``. Add the field lazily so measurement code can never
    turn a successful itinerary generation into a user-visible failure.
    """
    llm = getattr(workflow, "llm", None)
    if llm is not None and not hasattr(llm, "last_request_metrics"):
        llm.last_request_metrics = {}

    retriever = getattr(workflow, "retriever", None)
    if retriever is not None and not hasattr(retriever, "last_metrics"):
        retriever.last_metrics = {}


def snapshot_workflow_metrics(workflow: Any, final_state: dict[str, Any]) -> dict[str, Any]:
    """Collect best-effort metrics without raising for missing telemetry."""
    ensure_workflow_metrics_compatibility(workflow)
    return {
        "strategy": final_state.get("generation_strategy", "unknown"),
        "planner": _mapping(final_state.get("timings", {})),
        "retrieval": _mapping(
            getattr(getattr(workflow, "retriever", None), "last_metrics", {})
        ),
        "llm": _mapping(
            getattr(getattr(workflow, "llm", None), "last_request_metrics", {})
        ),
    }
