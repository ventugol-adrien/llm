import asyncio
import base64
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import httpx

from llm import LLMEngine
from mock_server import Handler, UnixHTTPServer


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.socket = self.directory / "server.sock"
        self.server = UnixHTTPServer(str(self.socket), Handler)
        self.server.requests = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("LLM_") and key not in ("SOCK", "XDG_RUNTIME_DIR")}
        env.update(SOCK=str(self.socket), XDG_RUNTIME_DIR=str(self.directory))
        self.presets = self.directory / "models.ini"
        self.presets.write_text("version = 1\n[test-model]\nmodel = /unused.gguf\n")
        env["LLM_MODELS_PRESET"] = str(self.presets)
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.engine = LLMEngine(parallel=2)
        self.addAsyncCleanup(self.engine.close)

    async def test_defaults_and_cli_payload_parity(self):
        self.assertEqual(await self.engine.generate("hello"), "hello")
        expected = self.server.requests[-1]
        result = await asyncio.to_thread(
            subprocess.run, [str(ROOT / "bin/llm"), "hello"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(self.server.requests[-1], expected)
        self.assertEqual(expected["max_tokens"], 4096)
        self.assertEqual(expected["reasoning_budget_tokens"], 1024)
        self.assertEqual(expected["model"], "qwen3.5-4bunc")

    async def test_model_selection_and_warmup(self):
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "environment-model"}):
            await self.engine.generate("hello")
            self.assertEqual(self.server.requests[-1]["model"], "environment-model")
            await self.engine.generate("hello", model="selected-model")
            self.assertEqual(self.server.requests[-1]["model"], "selected-model")
            async for _ in self.engine.stream("hello", model="stream-model"):
                pass
            self.assertEqual(self.server.requests[-1]["model"], "stream-model")
            self.server.requests.clear()
            await self.engine.generate_parallel(["one", "two"], system="system", model="batch-model")
            self.assertEqual(len(self.server.requests), 3)
            self.assertTrue(all(body["model"] == "batch-model" for body in self.server.requests))

    async def test_raw_model_default_and_override(self):
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "environment-model"}):
            payload: dict[str, object] = {"messages": [{"role": "user", "content": "hello"}]}
            async for _ in self.engine.stream_raw(payload):
                pass
            self.assertEqual(self.server.requests[-1]["model"], "environment-model")
            self.assertNotIn("model", payload)
            payload["model"] = "raw-model"
            async for _ in self.engine.stream_raw(payload):
                pass
            self.assertEqual(self.server.requests[-1]["model"], "raw-model")
            for model in ("", " \t", None, 42):
                payload["model"] = model
                with self.assertRaisesRegex(ValueError, "model"):
                    async for _ in self.engine.stream_raw(payload):
                        pass

    async def test_option_precedence(self):
        with patch.dict(os.environ, {
            "LLM_SYSTEM": "ignored", "LLM_MAX_TOKENS": "999",
            "LLM_TEMPERATURE": "1", "LLM_STOP_SEQUENCE": "ignored",
            "LLM_MAX_THINKING_TOKENS": "999", "LLM_FILES": "invalid",
            "LLM_SCHEMA": "/missing.yaml",
        }):
            await self.engine.generate(
                "hello", system="", max_tokens=17, temperature=0,
                stop_sequence='quoted "stop"\n', max_thinking_tokens=0, files=[], schema="",
            )
        body = self.server.requests[-1]
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["max_tokens"], 17)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["stop"], ['quoted "stop"\n'])
        self.assertEqual(body["reasoning_budget_tokens"], 0)
        self.assertEqual(body["response_format"], {"type": "text"})

    async def test_environment_options(self):
        with patch.dict(os.environ, {
            "LLM_SYSTEM": "system", "LLM_MAX_TOKENS": "-1",
            "LLM_TEMPERATURE": "0.2", "LLM_STOP_SEQUENCE": "STOP",
            "LLM_MAX_THINKING_TOKENS": "32",
        }):
            await self.engine.generate("hello")
        body = self.server.requests[-1]
        self.assertEqual(body["messages"][0]["content"], "system")
        self.assertEqual(body["max_tokens"], -1)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["stop"], ["STOP"])
        self.assertEqual(body["reasoning_budget_tokens"], 32)

    async def test_files_and_schema_parity(self):
        note = self.directory / "notes with spaces.txt"
        note.write_text('A "quoted" note.\n')
        image = self.directory / "image.gif"
        image.write_bytes(base64.b64decode(
            "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        ))
        schema = self.directory / "schema.yaml"
        schema.write_text("type: object\nproperties:\n  answer: {type: string}\nrequired: [answer]\n")
        with patch.dict(os.environ, {"LLM_FILES": json.dumps([str(note), str(image)]), "LLM_SCHEMA": str(schema)}):
            await self.engine.generate("hello")
        expected = self.server.requests[-1]
        await asyncio.to_thread(
            subprocess.run, [str(ROOT / "bin/llm"), "--file", str(note),
                             "--file", str(image), "--schema", str(schema), "hello"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(self.server.requests[-1], expected)
        self.assertEqual(expected["response_format"]["type"], "json_schema")

    async def test_large_audio_and_video(self):
        audio = b"\x00\x01" * 100_000
        wav = (
            b"RIFF" + struct.pack("<I", 36 + len(audio)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) +
            b"data" + struct.pack("<I", len(audio)) + audio
        )
        path = self.directory / "audio.wav"
        path.write_bytes(wav)
        video = self.directory / "video.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        await self.engine.generate("hello", files=[path, video])
        content = self.server.requests[-1]["messages"][-1]["content"]
        self.assertEqual(base64.b64decode(content[1]["input_audio"]["data"]), wav)
        self.assertEqual(base64.b64decode(content[2]["input_video"]["data"]), video.read_bytes())

    async def test_stream_and_raw(self):
        text = "".join([part async for part in self.engine.stream("hello", max_tokens=9)])
        self.assertEqual(text, "Hello world")
        self.assertEqual(self.server.requests[-1]["max_tokens"], 9)
        payload = {"messages": [{"role": "user", "content": "hello"}], "stream": False}
        data = b"".join([part async for part in self.engine.stream_raw(payload)])
        self.assertIn(b"data: [DONE]", data)
        self.assertIn(b"Hello", data)
        self.assertFalse(payload["stream"])
        self.assertTrue(self.server.requests[-1]["stream"])

    async def test_split_multiline_sse_and_server_error(self):
        class Chunks(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in (
                    b": comment\r\ndata: {\r\ndata: \"choices\":[{\"delta\":{\"content\":\"caf",
                    b"\xc3", b"\xa9\"}}]}\r\n\r\ndata: {\"choices\":[]}\n\n",
                    b"data: [DONE]\r\n\r\n",
                ):
                    yield chunk

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            return httpx.Response(200, stream=Chunks())

        self.engine._client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(handler))
        result = "".join([part async for part in self.engine.stream("hello")])
        self.assertEqual(result, "caf\u00e9")
        await self.engine.close()

        def error_handler(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            return httpx.Response(200, content=b'data: {"error":{"message":"failed"}}\n\n')

        self.engine._client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(error_handler))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            async for _ in self.engine.stream("hello"):
                pass

    async def test_stream_redirect_is_an_http_error(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            return httpx.Response(302, headers={"Location": "/elsewhere"})

        self.engine._client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(handler))
        with self.assertRaises(httpx.HTTPStatusError):
            async for _ in self.engine.stream("hello"):
                pass
        with self.assertRaises(httpx.HTTPStatusError):
            async for _ in self.engine.stream_raw({"messages": []}):
                pass

    async def test_stream_errors_and_early_close(self):
        for prompt, error in (
            ("bad-stream", ValueError), ("truncated-stream", ValueError),
            ("http-error", httpx.HTTPStatusError),
        ):
            with self.subTest(prompt=prompt):
                with self.assertRaises(error):
                    async for _ in self.engine.stream(prompt):
                        pass
        async with LLMEngine(parallel=1) as engine:
            stream = engine.stream("hello")
            self.assertEqual(await anext(stream), "Hello")
            await stream.aclose()
            self.assertEqual(await asyncio.wait_for(engine.generate("after"), 2), "after")
            raw = engine.stream_raw({"messages": [{"role": "user", "content": "hello"}]})
            self.assertTrue(await anext(raw))
            await raw.aclose()
            self.assertEqual(await asyncio.wait_for(engine.generate("after"), 2), "after")

    async def test_completion_errors(self):
        for prompt, error in (
            ("http-error", httpx.HTTPStatusError),
            ("api-error", RuntimeError), ("bad-response", ValueError),
        ):
            with self.subTest(prompt=prompt):
                with self.assertRaises(error):
                    await self.engine.generate(prompt)
        with self.assertRaises(httpx.HTTPStatusError) as error:
            async for _ in self.engine.stream_raw({"messages": [{"role": "user", "content": "http-error"}]}):
                pass
        self.assertIn("mock failure", error.exception.response.text)

    async def test_parallel_warmup_and_options(self):
        schema = self.directory / "schema.yaml"
        schema.write_text("type: string\n")
        result = await self.engine.generate_parallel(
            ["one", "two"], system="system", temperature=0,
            stop_sequence="STOP", files=[], schema=schema,
        )
        self.assertEqual(result, ["one", "two"])
        warmup, *requests = self.server.requests
        self.assertEqual(warmup["max_tokens"], 1)
        self.assertEqual(warmup["response_format"], {"type": "text"})
        self.assertNotIn("stop", warmup)
        for body in requests:
            self.assertEqual(body["messages"][0]["content"], "system")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["stop"], ["STOP"])
            self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(await self.engine.generate_parallel([]), [])

    async def test_concurrency_bound_and_order(self):
        active = maximum = 0

        async def handler(request):
            nonlocal active, maximum
            if request.url.path == "/health":
                return httpx.Response(200)
            prompt = json.loads(request.content)["messages"][-1]["content"]
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.05 if prompt == "one" else 0.02)
                return httpx.Response(200, json={"choices": [{"message": {"content": prompt}}]})
            finally:
                active -= 1

        self.engine._client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(handler))
        result = await self.engine.generate_parallel(["one", "two", "three", "four"])
        self.assertEqual(result, ["one", "two", "three", "four"])
        self.assertEqual(maximum, 2)
        self.assertEqual(active, 0)

    async def test_parallel_failure_cancels_requests(self):
        active = 0

        async def handler(request):
            nonlocal active
            if request.url.path == "/health":
                return httpx.Response(200)
            prompt = json.loads(request.content)["messages"][-1]["content"]
            if prompt == "fail":
                return httpx.Response(500)
            active += 1
            try:
                await asyncio.sleep(60)
                return httpx.Response(200)
            finally:
                active -= 1

        self.engine._client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(handler))
        with self.assertRaises(httpx.HTTPStatusError):
            await asyncio.wait_for(self.engine.generate_parallel(["slow", "fail", "queued"]), 2)
        self.assertEqual(active, 0)

    async def test_invalid_inputs_before_network(self):
        for options in (
            {"max_tokens": -2}, {"max_tokens": True}, {"temperature": float("nan")},
            {"max_thinking_tokens": 2147483648}, {"files": [""]},
            {"files": [self.directory]}, {"schema": "/missing-schema.yaml"},
            {"model": ""}, {"model": " \t"}, {"model": 42},
        ):
            with self.subTest(options=options):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    await self.engine.generate("hello", **options)
        with self.assertRaises(ValueError):
            await self.engine.generate("")
        with patch.dict(os.environ, {"LLM_FILES": "{}"}):
            with self.assertRaises(ValueError):
                await self.engine.generate("hello")
        self.assertEqual(self.server.requests, [])
        self.assertIsNone(self.engine._client)

    async def test_constructor_and_context_cleanup(self):
        with self.assertRaises(ValueError):
            LLMEngine(parallel=0)
        with patch.dict(os.environ, {"LLM_PARALLEL": "3"}):
            engine = LLMEngine()
            self.assertEqual(engine.parallel, 3)
            self.assertEqual(engine.sock, str(self.socket))
            async with engine:
                client = engine._get_client()
                await engine.generate("hello")
            self.assertTrue(client.is_closed)
            self.assertIsNone(engine._client)
            await engine.close()
            self.assertEqual(await engine.generate("again"), "again")
            await engine.close()

    async def test_lifecycle_status(self):
        output = io.StringIO()
        with redirect_stderr(output):
            await self.engine.ensure_started()
        self.assertEqual(output.getvalue(), f"[llm] reused: healthy server at {self.socket}\n")

    async def test_supervisor_auto_start(self):
        launcher = self.directory / "mock-server"
        launcher.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} {ROOT / 'tests/mock_server.py'} \"$@\"\n")
        launcher.chmod(0o700)
        engine = LLMEngine(sock=str(self.directory / "managed.sock"))
        self.addAsyncCleanup(engine.close)
        with patch.dict(os.environ, {
            "LLM_SERVER_EXEC": str(launcher),
        }):
            try:
                with patch("llm.core.asyncio.create_subprocess_exec", wraps=asyncio.create_subprocess_exec) as start:
                    self.assertEqual(
                        await asyncio.gather(engine.generate("one"), engine.generate("two")),
                        ["one", "two"],
                    )
                    self.assertEqual(start.await_count, 1)
                self.assertTrue(Path(engine.sock).exists())
            finally:
                pid_file = self.directory / "llama-shared.pid"
                if pid_file.exists():
                    os.kill(int(pid_file.read_text()), 15)

    async def test_failed_context_entry_closes_client(self):
        engine = LLMEngine(sock=str(self.directory / "missing.sock"))
        with patch.dict(os.environ, {
            "LLM_SERVER_EXEC": "/bin/false",
        }):
            with self.assertRaises(subprocess.CalledProcessError):
                async with engine:
                    self.fail("Failed startup must not enter the context")
        self.assertIsNone(engine._client)


if __name__ == "__main__":
    unittest.main()
