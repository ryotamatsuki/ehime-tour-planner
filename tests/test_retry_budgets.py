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


def _payload(*, with_usage: bool = False) -> dict:
    data = {"choices": [{"message": {"content": '{"city":"松山"}'}}]}
    if with_usage:
        data["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "total_tokens": 112,
        }
    return data


def test_warm_request_uses_short_retry_budget():
    responses = [
        FakeResponse(200, _payload()),
        FakeResponse(503, {}),
        FakeResponse(503, {}),
    ]
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        cold_start_retries=5,
        generation_retries=1,
        retry_backoff_seconds=0,
    )

    with patch("llm.sarashina_client.requests.post", side_effect=responses) as post:
        client.generate_json(
            prompt="first",
            schema={"type": "object"},
            schema_name="first",
        )
        try:
            client.generate_json(
                prompt="second",
                schema={"type": "object"},
                schema_name="second",
            )
        except RuntimeError as exc:
            assert str(exc) == "HTTP 503"
        else:
            raise AssertionError("warm retry budget should be exhausted quickly")

    assert post.call_count == 3


def test_cold_start_uses_constant_poll_interval_and_records_wait():
    responses = [
        FakeResponse(503, {}),
        FakeResponse(503, {}),
        FakeResponse(200, _payload(with_usage=True)),
    ]
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        cold_start_retries=5,
        cold_start_poll_seconds=2.0,
    )

    with (
        patch("llm.sarashina_client.requests.post", side_effect=responses) as post,
        patch("llm.sarashina_client.time.sleep") as sleep,
    ):
        client.generate_json(
            prompt="cold",
            schema={"type": "object"},
            schema_name="cold",
        )

    assert post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2.0, 2.0]
    assert client.last_request_metrics["phase"] == "cold_start"
    assert client.last_request_metrics["retries"] == 2
    assert client.last_request_metrics["retry_wait_ms"] == 4000.0


def test_request_metrics_capture_usage():
    client = SarashinaClient(
        base_url="https://example.modal.direct",
        api_key="test-key",
        max_retries=0,
    )
    with patch(
        "llm.sarashina_client.requests.post",
        return_value=FakeResponse(200, _payload(with_usage=True)),
    ):
        client.generate_json(
            prompt="test",
            schema={"type": "object"},
            schema_name="test",
        )

    assert client.last_request_metrics["phase"] == "cold_start"
    assert client.last_request_metrics["attempts"] == 1
    assert client.last_request_metrics["retry_wait_ms"] == 0.0
    assert client.last_request_metrics["prompt_tokens"] == 100
    assert client.last_request_metrics["completion_tokens"] == 12
