from __future__ import annotations

import json
import time
from typing import Any

import requests


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class SarashinaClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "sarashina",
        timeout: int = 300,
        max_retries: int = 5,
        retry_backoff_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def generate_json(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 2200,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "指示と根拠に忠実に従う日本語アシスタントです。",
                },
                {"role": "user", "content": prompt},
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
        request_kwargs = {
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
            "timeout": self.timeout,
        }

        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions", **request_kwargs
                )
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt)
                continue

            if (
                response.status_code not in RETRYABLE_STATUS_CODES
                or attempt >= self.max_retries
            ):
                response.raise_for_status()
                break

            self._sleep_before_retry(attempt)

        if response is None:
            raise RuntimeError("Sarashina APIへの接続に失敗しました。")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        return json.loads(content)

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        time.sleep(min(delay, 15.0))
