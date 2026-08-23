import json
import os
import socket
import subprocess
import time

import modal


MODEL_NAME = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
SERVED_MODEL_NAME = "sarashina"
VLLM_PORT = 8000
MINUTES = 60
READY_TIMEOUT = 10 * MINUTES

app = modal.App("ehime-tour-planner-sarashina")

hf_cache = modal.Volume.from_name("ehime-tour-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ehime-tour-vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)


@app.server(
    image=vllm_image,
    gpu="T4",
    scaledown_window=60,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("ehime-tour-planner-vllm")],
    port=VLLM_PORT,
    target_concurrency=4,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        api_key = os.environ["VLLM_API_KEY"]
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            api_key,
            "--dtype",
            "half",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-seqs",
            "4",
            "--enable-prefix-caching",
            "--enforce-eager",
            "--disable-log-requests",
            "--generation-config",
            "vllm",
        ]
        print("Starting vLLM for", MODEL_NAME)
        self.process = subprocess.Popen(cmd)
        self._wait_until_ready()

    def _wait_until_ready(self):
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited before becoming ready: returncode={self.process.returncode}"
                )
            try:
                with socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=1):
                    print("vLLM is accepting connections")
                    return
            except OSError:
                time.sleep(1)
        raise TimeoutError(f"vLLM did not become ready within {READY_TIMEOUT} seconds")

    @modal.exit()
    def stop(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@app.local_entrypoint()
async def smoke_test():
    import aiohttp
    import asyncio
    import time

    url = await Server.get_url.aio()
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Local smoke test requires VLLM_API_KEY in the local environment."
        )

    deadline = time.time() + 9 * MINUTES
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    "/health",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("vLLM health check timed out.")

        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [
                {"role": "user", "content": "愛媛県の県庁所在地をJSONで答えて。"}
            ],
            "max_tokens": 64,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "smoke",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["city"],
                        "properties": {"city": {"type": "string"}},
                    },
                },
            },
        }
        async with session.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as response:
            response.raise_for_status()
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed.get("city"), str) or not parsed["city"]:
                raise RuntimeError(f"Structured output did not contain city: {parsed!r}")
            print(json.dumps(parsed, ensure_ascii=False))
