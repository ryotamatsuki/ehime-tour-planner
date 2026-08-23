from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
LOGGER = logging.getLogger(__name__)


class SarashinaClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "sarashina",
        timeout: int = 300,
        max_retries: int | None = None,
        cold_start_retries: int = 120,
        generation_retries: int = 2,
        retry_backoff_seconds: float = 3.0,
        cold_start_poll_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        if max_retries is not None:
            # Backward-compatible override used by tests and callers that want
            # one retry budget for both phases.
            cold_start_retries = max_retries
            generation_retries = max_retries
        self.cold_start_retries = max(0, cold_start_retries)
        self.generation_retries = max(0, generation_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.cold_start_poll_seconds = max(0.0, cold_start_poll_seconds)
        self._ready_seen = False
        self.last_request_metrics: dict[str, Any] = {}

    def generate_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 2200,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        # 小型モデルでは、長い旅程JSONが出力上限で途中終了することがある。
        # 1回だけ「省スペース」を強調して再試行し、ユーザー操作を失敗にしない。
        parse_prompts = [
            prompt,
            prompt
            + "\n\n【出力上限への最終注意】各日2件まで、各文字列は短くし、"
            + "必ず閉じた完全なJSONだけを返してください。Markdownは禁止です。",
        ]
        last_error: Exception | None = None
        for parse_prompt in parse_prompts:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "指示と根拠に忠実に従う日本語アシスタントです。",
                    },
                    {"role": "user", "content": parse_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            try:
                data = self._post_with_retries(payload)
                return self._decode_json_content(data)
            except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
                last_error = exc

        raise ValueError("Sarashina APIが完全なJSONを返しませんでした。") from last_error

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_kwargs = {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
            "timeout": self.timeout,
        }
        retry_budget = (
            self.generation_retries if self._ready_seen else self.cold_start_retries
        )
        phase = "warm" if self._ready_seen else "cold_start"
        started = time.perf_counter()
        response = None
        attempts = 0
        retry_wait_ms = 0.0

        for attempt in range(retry_budget + 1):
            attempts = attempt + 1
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions", **request_kwargs
                )
            except requests.RequestException:
                if attempt >= retry_budget:
                    self._record_metrics(
                        started=started,
                        phase=phase,
                        attempts=attempts,
                        status_code=None,
                        usage=None,
                        retry_wait_ms=retry_wait_ms,
                    )
                    raise
                retry_wait_ms += self._sleep_before_retry(attempt, phase)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < retry_budget:
                retry_wait_ms += self._sleep_before_retry(attempt, phase)
                continue

            if response.status_code >= 400:
                self._record_metrics(
                    started=started,
                    phase=phase,
                    attempts=attempts,
                    status_code=response.status_code,
                    usage=None,
                    retry_wait_ms=retry_wait_ms,
                )
                response.raise_for_status()
            break

        if response is None:
            raise RuntimeError("Sarashina APIへの接続に失敗しました。")

        data = response.json()
        self._ready_seen = True
        self._record_metrics(
            started=started,
            phase=phase,
            attempts=attempts,
            status_code=response.status_code,
            usage=data.get("usage") if isinstance(data, dict) else None,
            retry_wait_ms=retry_wait_ms,
        )
        return data

    def _record_metrics(
        self,
        *,
        started: float,
        phase: str,
        attempts: int,
        status_code: int | None,
        usage: Any,
        retry_wait_ms: float = 0.0,
    ) -> None:
        metrics: dict[str, Any] = {
            "phase": phase,
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "retry_wait_ms": round(retry_wait_ms, 1),
            "status_code": status_code,
        }
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    metrics[key] = value
        self.last_request_metrics = metrics
        LOGGER.info("sarashina_request_metrics=%s", metrics)

    @staticmethod
    def _decode_json_content(data: dict[str, Any]) -> dict[str, Any]:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        text = str(content).strip()
        if text.startswith("\x60\x60\x60json"):
            text = text[7:]
        elif text.startswith("\x60\x60\x60"):
            text = text[3:]
        if text.endswith("\x60\x60\x60"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 説明文が前後に混ざった場合だけ、JSON本体を再抽出する。
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _sleep_before_retry(self, attempt: int, phase: str) -> float:
        if phase == "cold_start":
            # A 5xx during scale-from-zero is a readiness signal, not a reason
            # for increasingly long blind sleeps. Poll frequently so the first
            # ready replica is used promptly. The larger cold retry budget keeps
            # the same multi-minute startup tolerance.
            delay = self.cold_start_poll_seconds
        else:
            delay = min(self.retry_backoff_seconds * (2**attempt), 15.0)
        time.sleep(delay)
        return delay * 1000.0
