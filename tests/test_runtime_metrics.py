from pathlib import Path

from workflow.planner import PlannerWorkflow

from utils.runtime_metrics import (
    ensure_workflow_metrics_compatibility,
    snapshot_workflow_metrics,
)


class LegacyClient:
    pass


class LegacyRetriever:
    pass


class LegacyWorkflow:
    def __init__(self):
        self.llm = LegacyClient()
        self.retriever = LegacyRetriever()


def test_legacy_cached_workflow_gets_safe_metrics_fields():
    workflow = LegacyWorkflow()

    ensure_workflow_metrics_compatibility(workflow)

    assert workflow.llm.last_request_metrics == {}
    assert workflow.retriever.last_metrics == {}


def test_snapshot_never_fails_when_legacy_metrics_are_missing():
    workflow = LegacyWorkflow()
    final_state = {
        "generation_strategy": "spot_id_single_pass_2d",
        "timings": {"total_ms": 22600.0, "generation_ms": 18000.0},
    }

    metrics = snapshot_workflow_metrics(workflow, final_state)

    assert metrics["strategy"] == "spot_id_single_pass_2d"
    assert metrics["planner"]["total_ms"] == 22600.0
    assert metrics["retrieval"] == {}
    assert metrics["llm"] == {}


def test_snapshot_preserves_new_client_metrics():
    workflow = LegacyWorkflow()
    workflow.llm.last_request_metrics = {
        "elapsed_ms": 1234.5,
        "prompt_tokens": 321,
        "completion_tokens": 98,
        "retries": 0,
    }
    workflow.retriever.last_metrics = {
        "cache_hit": True,
        "doc_embedding_ms": 0.0,
        "query_embedding_ms": 42.0,
    }

    metrics = snapshot_workflow_metrics(workflow, {"timings": {}})

    assert metrics["llm"]["elapsed_ms"] == 1234.5
    assert metrics["llm"]["completion_tokens"] == 98
    assert metrics["retrieval"]["cache_hit"] is True


def test_streamlit_entrypoint_busts_versioned_resources_and_uses_safe_snapshot():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'WORKFLOW_CONFIG_VERSION = "spot-id-cache-v3-quality"' in source
    assert 'RETRIEVER_CONFIG_VERSION = "canonical-spot-dedupe-v1"' in source
    assert 'RESOURCE_CACHE_EPOCH = "pr20-quality-runtime-reverify-1"' in source
    assert "resource_cache_epoch" in source
    assert "WORKFLOW_CONFIG_VERSION," in source
    assert "RETRIEVER_CONFIG_VERSION," in source
    assert "ensure_workflow_metrics_compatibility(workflow)" in source
    assert "snapshot_workflow_metrics(workflow, final_state)" in source
    assert "dict(workflow.llm.last_request_metrics)" not in source


class FakeGraph:
    def invoke(self, state):
        return {"generation_strategy": "spot_id_single_pass_2d", "timings": {}}


def test_planner_logging_is_safe_for_legacy_cached_client():
    workflow = PlannerWorkflow.__new__(PlannerWorkflow)
    workflow.llm = LegacyClient()
    workflow.graph = FakeGraph()

    result = workflow.run_plan()

    assert result["generation_strategy"] == "spot_id_single_pass_2d"
