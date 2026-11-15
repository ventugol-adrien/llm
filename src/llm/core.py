import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import TracebackType

import httpx

from .schema import response_format


def _integer(value: int, name: str, minimum: int = -1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2147483647:
        raise ValueError(f"{name} must be an integer between {minimum} and 2147483647")
    return value


def _model_name(model: str | None) -> str:
    model = model if model is not None else os.environ.get("LLM_MODEL_NAME") or "qwen3.5-4bunc"
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty name")
    return model


def _content(data: object, field: str) -> str | None:
    if not isinstance(data, dict):
        raise ValueError("invalid completion response")
    if data.get("error"):
        raise RuntimeError(f"llama-server error: {data['error']}")
    choices = data.get("choices")
    if not isinstance(choices, list):
        raise ValueError("response has no choices array")
    if not choices and field == "delta":
        return None
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no assistant choice")
    message = choices[0].get(field)
    if not isinstance(message, dict):
        raise ValueError(f"response has no assistant {field}")
    content = message.get("content")
    if content is None and field == "delta":
        return None
    if not isinstance(content, str):
        raise ValueError("response has no assistant text content")
    return content


class LLMEngine:
    """Async llama-server client with CLI-equivalent options over a Unix domain socket."""

    def __init__(self, parallel: int | None = None, sock: str | None = None) -> None:
        self.parallel = _integer(
            parallel if parallel is not None else int(os.environ.get("LLM_PARALLEL") or "1"),
            "parallel", 1,
        )
        run_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/llm-{os.getuid()}"
        self.sock = sock if sock is not None else os.environ.get("SOCK") or f"{run_dir}/llama-shared.sock"
        if not self.sock:
            raise ValueError("sock must be a nonempty socket path")

        self._sem = asyncio.Semaphore(self.parallel)
        self._startup_lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=self.parallel * 2,
                max_connections=self.parallel * 4,
            )
            transport = httpx.AsyncHTTPTransport(uds=self.sock, limits=limits, trust_env=False)
            self._client = httpx.AsyncClient(
                base_url=os.environ.get("LLM_BASE_URL") or "http://localhost",
                transport=transport,
                timeout=httpx.Timeout(150.0, connect=5.0),
                headers={"Accept-Encoding": "identity"},
            )
        return self._client

    async def __aenter__(self) -> "LLMEngine":
        started = False
        try:
            await self.ensure_started()
            started = True
        finally:
            if not started:
                await self.close()
        return self

    async def __aexit__(
        self, _exc_type: type[BaseException] | None, _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def ensure_started(self) -> None:
        async with self._startup_lock:
            client = self._get_client()
            if Path(self.sock).exists():
                try:
                    health = await client.get("/health", timeout=5.0)
                except httpx.TransportError as error:
                    print(f"[llm] failed: server health check: {error}", file=sys.stderr)
                else:
                    if health.status_code == 200:
                        print(f"[llm] reused: healthy server at {self.sock}", file=sys.stderr)
                        return
                    print(f"[llm] failed: server health check returned HTTP {health.status_code}", file=sys.stderr)

            candidates = (
                Path(__file__).resolve().parents[2] / "bin" / "llm",
                Path(sys.prefix) / "bin" / "llm",
            )
            executable = next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
            executable = executable or shutil.which("llm")
            if executable is None:
                raise FileNotFoundError("Cannot find the llm CLI; install the package or run from its source tree")
            env = os.environ | {"SOCK": self.sock, "LLM_PARALLEL": str(self.parallel)}
            process = await asyncio.create_subprocess_exec(
                executable, "--ensure-daemon", env=env, stdout=subprocess.DEVNULL,
            )
            try:
                status = await process.wait()
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
            if status != 0:
                raise subprocess.CalledProcessError(status, [executable, "--ensure-daemon"])
            health = await client.get("/health", timeout=5.0)
            health.raise_for_status()

    def _payload(
        self, prompt: str, system: str | None, max_tokens: int | None,
        temperature: float | None, stop_sequence: str | None,
        max_thinking_tokens: int | None, files: Sequence[str | Path] | None,
        schema: str | Path | None, stream: bool = False,
        model: str | None = None,
    ) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("a nonempty prompt is required")
        system = system if system is not None else os.environ.get("LLM_SYSTEM") or ""
        stop_sequence = stop_sequence if stop_sequence is not None else os.environ.get("LLM_STOP_SEQUENCE") or ""
        max_tokens = _integer(
            max_tokens if max_tokens is not None else int(os.environ.get("LLM_MAX_TOKENS") or "4096"),
            "max_tokens",
        )
        max_thinking_tokens = _integer(
            max_thinking_tokens if max_thinking_tokens is not None else int(os.environ.get("LLM_MAX_THINKING_TOKENS") or "1024"),
            "max_thinking_tokens",
        )
        temperature = temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE") or "0.8")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            raise ValueError("temperature must be a finite number")
        if not isinstance(system, str) or not isinstance(stop_sequence, str):
            raise ValueError("system and stop_sequence must be strings")
        if files is None:
            paths: object = json.loads(os.environ.get("LLM_FILES") or "[]")
            if not isinstance(paths, list) or not all(isinstance(path, str) and path for path in paths):
                raise ValueError("LLM_FILES must be a JSON array of nonempty paths")
            files = paths
        if isinstance(files, (str, bytes)):
            raise ValueError("files must be a sequence of paths, not one string")
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for name in files:
            if not isinstance(name, (str, Path)) or not str(name):
                raise ValueError("files must contain nonempty paths")
            path = Path(name).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Not a readable file: {path}")
            mime = subprocess.run(
                ["file", "--brief", "--mime-type", "--", str(path)],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if mime.startswith(("image/", "audio/", "video/")):
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                if mime.startswith("image/"):
                    part: dict[str, object] = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
                elif mime.startswith("audio/"):
                    part = {"type": "input_audio", "input_audio": {"data": encoded}}
                else:
                    part = {"type": "input_video", "input_video": {"data": encoded}}
            elif mime.startswith("text/") or mime in (
                "application/json", "application/xml", "application/javascript", "inode/x-empty",
            ):
                part = {"type": "text", "text": f"File: {path}\n{path.read_text(encoding='utf-8', errors='replace')}"}
            else:
                raise ValueError(f"Unsupported file type {mime}: {path}")
            content.append(part)
        schema = schema if schema is not None else os.environ.get("LLM_SCHEMA") or ""
        output_format = json.loads(response_format(str(Path(schema).resolve(strict=True)))) if schema else {"type": "text"}
        messages: list[dict[str, object]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content if files else prompt})
        payload: dict[str, object] = {
            "model": _model_name(model),
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "cache_prompt": True, "stream": stream, "response_format": output_format,
            "reasoning_budget_tokens": max_thinking_tokens,
        }
        if stop_sequence:
            payload["stop"] = [stop_sequence]
        return payload

    async def _generate(self, payload: dict[str, object]) -> str:
        async with self._sem:
            response = await self._get_client().post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            text = _content(response.json(), "message")
            if text is None:
                raise ValueError("response has no assistant text content")
            return text

    async def generate(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, stop_sequence: str | None = None,
        max_thinking_tokens: int | None = None, files: Sequence[str | Path] | None = None,
        schema: str | Path | None = None,
        *, model: str | None = None,
    ) -> str:
        payload = await asyncio.to_thread(
            self._payload, prompt, system, max_tokens, temperature, stop_sequence,
            max_thinking_tokens, files, schema, model=model,
        )
        await self.ensure_started()
        return await self._generate(payload)

    async def generate_parallel(
        self, prompts: Sequence[str], system: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, stop_sequence: str | None = None,
        max_thinking_tokens: int | None = None, files: Sequence[str | Path] | None = None,
        schema: str | Path | None = None,
        *, model: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, (str, bytes)):
            raise ValueError("prompts must be a sequence of strings")
        if not prompts:
            return []
        model = _model_name(model)
        payloads = [
            await asyncio.to_thread(
                self._payload, prompt, system, max_tokens, temperature, stop_sequence,
                max_thinking_tokens, files, schema, model=model,
            ) for prompt in prompts
        ]
        await self.ensure_started()
        shared_system = system if system is not None else os.environ.get("LLM_SYSTEM") or ""
        if shared_system:
            warmup = await asyncio.to_thread(
                self._payload, ".", shared_system, 1, temperature, "", max_thinking_tokens, [], "", model=model,
            )
            await self._generate(warmup)
        tasks = [asyncio.create_task(self._generate(payload)) for payload in payloads]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stream_raw(self, payload: dict[str, object]) -> AsyncIterator[bytes]:
        model = None
        if "model" in payload:
            value = payload["model"]
            if not isinstance(value, str):
                raise ValueError("model must be a nonempty name")
            model = value
        request = payload | {"model": _model_name(model), "stream": True}
        await self.ensure_started()
        async with self._sem:
            async with self._get_client().stream(
                "POST", "/v1/chat/completions", json=request,
            ) as response:
                if not response.is_success:
                    await response.aread()
                    response.raise_for_status()
                async for chunk in response.aiter_raw():
                    yield chunk

    async def stream(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, stop_sequence: str | None = None,
        max_thinking_tokens: int | None = None, files: Sequence[str | Path] | None = None,
        schema: str | Path | None = None,
        *, model: str | None = None,
    ) -> AsyncIterator[str]:
        payload = await asyncio.to_thread(
            self._payload, prompt, system, max_tokens, temperature, stop_sequence,
            max_thinking_tokens, files, schema, True, model=model,
        )
        await self.ensure_started()
        async with self._sem:
            async with self._get_client().stream("POST", "/v1/chat/completions", json=payload) as response:
                if not response.is_success:
                    await response.aread()
                    response.raise_for_status()
                data: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].removeprefix(" "))
                    elif line == "" and data:
                        event = "\n".join(data)
                        data.clear()
                        if event == "[DONE]":
                            return
                        text = _content(json.loads(event), "delta")
                        if text is not None:
                            yield text
                raise ValueError("stream ended without [DONE]")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
