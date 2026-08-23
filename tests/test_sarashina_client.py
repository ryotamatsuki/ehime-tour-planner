from __future__ import annotations

from unittest.mock import patch

from llm.sarashina_client import SarashinaClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _payload() -> dict:
    return {
        "choices": [
            {"message": {"content": '{"city":"松山"}'}},
        ]
    }


def test_retries_modal_cold_start_503_then_succeeds():
    responses = [FakeResponse(503, {}), FakeResponse(200, _payload())]
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        max_retries=2,
        retry_backoff_seconds=0,
        cold_start_poll_seconds=0,
    )

    with patch("llm.sarashina_client.requests.post", side_effect=responses) as post:
        with patch("llm.sarashina_client.time.sleep") as sleep:
            result = client.generate_json(
                prompt="test",
                schema={"type": "object"},
                schema_name="test",
            )

    assert result == {"city": "松山"}
    assert post.call_count == 2
    sleep.assert_called_once_with(0)
    assert client.last_request_metrics["retry_wait_ms"] == 0.0


def test_raises_after_retry_budget_is_exhausted():
    responses = [FakeResponse(503, {}), FakeResponse(503, {})]
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        max_retries=1,
        retry_backoff_seconds=0,
        cold_start_poll_seconds=0,
    )

    with patch("llm.sarashina_client.requests.post", side_effect=responses) as post:
        with patch("llm.sarashina_client.time.sleep"):
            try:
                client.generate_json(
                    prompt="test",
                    schema={"type": "object"},
                    schema_name="test",
                )
            except RuntimeError as exc:
                assert str(exc) == "HTTP 503"
            else:
                raise AssertionError("503 should be raised after retries are exhausted")

    assert post.call_count == 2


def test_parses_json_code_fence_and_surrounding_text():
    data = {
        "choices": [
            {
                "message": {
                    "content": '説明です。\n\x60\x60\x60json\n{"city":"松山"}\n\x60\x60\x60'
                }
            }
        ]
    }
    assert SarashinaClient._decode_json_content(data) == {"city": "松山"}


def test_retries_once_when_json_is_truncated():
    responses = [
        FakeResponse(200, {"choices": [{"message": {"content": '{"city":'}}]}),
        FakeResponse(200, _payload()),
    ]
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        max_retries=0,
    )

    with patch("llm.sarashina_client.requests.post", side_effect=responses) as post:
        result = client.generate_json(
            prompt="test",
            schema={"type": "object"},
            schema_name="test",
        )

    assert result == {"city": "松山"}
    assert post.call_count == 2
