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
        max_retries: int = 14,
        retry_backoff_seconds: float = 3.0,
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
        return response.json()

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

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        time.sleep(min(delay, 15.0))
