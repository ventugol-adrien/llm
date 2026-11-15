import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from mock_server import Handler, UnixHTTPServer


ROOT = Path(__file__).resolve().parents[1]


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.socket = self.directory / "server.sock"
        self.server = UnixHTTPServer(str(self.socket), Handler)
        self.server.requests = []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("LLM_") and key not in ("SOCK", "XDG_RUNTIME_DIR")
        }
        self.config = self.directory / "config" / "llm"
        self.config.mkdir(parents=True)
        self.presets = self.config / "models.ini"
        self.presets.write_text("version = 1\n[test-model]\nmodel = /unused.gguf\n")
        self.env.update(SOCK=str(self.socket), XDG_RUNTIME_DIR=str(self.directory),
                        XDG_CONFIG_HOME=str(self.config.parent))

    def cli(self, *args, stdin=None, env=None):
        return subprocess.run(
            [os.environ.get("LLM_TEST_EXECUTABLE", str(ROOT / "bin/llm")), *args],
            input=stdin, text=True, capture_output=True, timeout=20,
            env=self.env | (env or {}), cwd=ROOT,
        )

    def library(self, command, *args, env=None, stdin=None):
        return subprocess.run(
            ["bash", "-euc", 'source lib/client.sh; ' + command, "_", *args],
            input=stdin, text=True, capture_output=True, timeout=20,
            env=self.env | (env or {}), cwd=ROOT,
        )

    def test_defaults(self):
        result = self.cli("hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "hello\n")
        body = self.server.requests[-1]
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["temperature"], 0.8)
        self.assertTrue(body["cache_prompt"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["reasoning_budget_tokens"], 1024)
        self.assertEqual(body["model"], "qwen3.5-4bunc")

    def test_model_selection_for_all_request_modes(self):
        for mode in ((), ("--stream",), ("--batch", "--system", "system")):
            with self.subTest(mode=mode):
                self.server.requests.clear()
                result = self.cli(*mode, "--model", "selected-model", "hello",
                                  env={"LLM_MODEL_NAME": "environment-model"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(self.server.requests)
                for body in self.server.requests:
                    self.assertEqual(body["model"], "selected-model")
        result = self.cli("hello", env={"LLM_MODEL_NAME": "environment-model"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[-1]["model"], "environment-model")
        for command in ('call_one hello', 'stream_one hello',
                        'run_batch system 1 0.8 "" 0 "[]" "" hello'):
            result = self.library(command, env={"LLM_MODEL_NAME": "library-model", "LLM_MAX_TOKENS": "4096"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.server.requests[-1]["model"], "library-model")

    def test_invalid_model_before_network(self):
        for model in ("", " \t"):
            result = self.cli("--model", model, "hello")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_flags_override_environment(self):
        result = self.cli(
            "--system", "", "--max-tokens", "17", "--temperature", "0",
            "--stop-sequence", 'a"\nb', "--max-thinking-tokens", "0",
            "--schema", "", "hello",
            env={"LLM_SYSTEM": "ignored", "LLM_MAX_TOKENS": "42",
                 "LLM_TEMPERATURE": "1", "LLM_STOP_SEQUENCE": "ignored",
                 "LLM_MAX_THINKING_TOKENS": "55", "LLM_SCHEMA": "/ignored.yaml"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self.server.requests[-1]
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["max_tokens"], 17)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["stop"], ['a"\nb'])
        self.assertEqual(body["reasoning_budget_tokens"], 0)
        self.assertEqual(body["response_format"], {"type": "text"})

    def test_library_environment_and_optional_arguments(self):
        result = self.library(
            'call_one "$1"', "hello",
            env={"LLM_SYSTEM": "system", "LLM_MAX_TOKENS": "-1",
                 "LLM_MAX_THINKING_TOKENS": "-1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self.server.requests[-1]
        self.assertEqual(body["messages"][0]["content"], "system")
        self.assertEqual(body["max_tokens"], -1)
        self.assertEqual(body["reasoning_budget_tokens"], -1)

    def test_stdin_and_large_prompt(self):
        prompt = "a" * 200_000
        result = self.cli(stdin=prompt)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, prompt + "\n")

    def test_zero_max_tokens(self):
        result = self.cli("--max-tokens", "0", "hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[-1]["max_tokens"], 0)

    def test_text_file_and_file_flag_override(self):
        path = self.directory / "notes with spaces.txt"
        path.write_text('Quoted "text"\nsecond line\n')
        result = self.cli("--file", str(path), "hello", env={"LLM_FILES": "invalid"})
        self.assertEqual(result.returncode, 0, result.stderr)
        content = self.server.requests[-1]["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "hello"})
        self.assertEqual(content[1]["text"], f"File: {path}\n{path.read_text()}")

    def test_large_media(self):
        # A WAV header is enough for MIME detection; the mock does not decode audio.
        import struct

        audio = b"\x00\x01" * 100_000
        data = (
            b"RIFF" + struct.pack("<I", 36 + len(audio)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) +
            b"data" + struct.pack("<I", len(audio)) + audio
        )
        path = self.directory / "audio.wav"
        path.write_bytes(data)
        result = self.cli("--file", str(path), "hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        part = self.server.requests[-1]["messages"][-1]["content"][1]
        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(base64.b64decode(part["input_audio"]["data"]), data)

    def test_image_and_video_payloads(self):
        image = self.directory / "image.gif"
        image.write_bytes(base64.b64decode(
            "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        ))
        video = self.directory / "video.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        result = self.cli("--file", str(image), "--file", str(video), "hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        content = self.server.requests[-1]["messages"][-1]["content"]
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/gif;base64,"))
        self.assertEqual(base64.b64decode(content[2]["input_video"]["data"]), video.read_bytes())

    def test_schema_format(self):
        path = self.directory / "schema with spaces.yaml"
        path.write_text(
            "type: object\nproperties:\n  answer:\n    type: integer\n"
            "required: [answer]\nadditionalProperties: false\n"
        )
        result = self.cli("--schema", str(path), "hello", env={"LLM_SCHEMA": "/ignored.yaml"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[-1]["response_format"], {
            "type": "json_schema", "json_schema": {"name": "response", "schema": {
                "type": "object", "properties": {"answer": {"type": "integer"}},
                "required": ["answer"], "additionalProperties": False,
            }},
        })

    def test_schema_environment_and_stream(self):
        path = self.directory / "schema.yaml"
        path.write_text("type: string\nminLength: 1\n")
        result = self.cli("--stream", "hello", env={"LLM_SCHEMA": str(path)})
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self.server.requests[-1]
        self.assertTrue(body["stream"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"], {
            "type": "string", "minLength": 1,
        })

    def test_schema_batch_and_library(self):
        path = self.directory / "schema.yaml"
        path.write_text("type: string\n")
        result = self.cli("--batch", "--system", "system", "--schema", str(path), "one", "two")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[0]["response_format"], {"type": "text"})
        for body in self.server.requests[1:]:
            self.assertEqual(body["response_format"]["json_schema"]["schema"], {"type": "string"})
        result = self.library('call_one "$1"', "hello",
                              env={"LLM_SCHEMA": str(path), "LLM_MAX_TOKENS": "4096"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[-1]["response_format"]["type"], "json_schema")

    def test_invalid_schema(self):
        path = self.directory / "schema.yaml"
        for content in (
            "", "type: [", "- object\n", "type: !!python/object:builtins.object {}",
            "type: object\nproperties:\n  42: {type: string}\n",
            "type: number\nminimum: .nan\n", "default: 2026-01-01\n",
            "type: object\ndefault: &loop [*loop]\n",
        ):
            with self.subTest(content=content):
                path.write_text(content)
                result = self.cli("--schema", str(path), "hello")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid schema file", result.stderr)
        result = self.cli("--schema", str(path) + ".missing", "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_json_schema_file(self):
        path = self.directory / "schema.json"
        path.write_text(json.dumps({"type": "string"}))
        result = self.cli("--schema", str(path), "hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.server.requests[-1]["response_format"]["json_schema"]["schema"],
                         {"type": "string"})

    def test_format_flag_removed(self):
        result = self.cli("--format", "json", "hello")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown option: --format", result.stderr)

    def test_stream(self):
        result = self.cli("--stream", "--max-tokens", "9", "hello")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Hello world\n")
        self.assertTrue(self.server.requests[-1]["stream"])
        self.assertEqual(self.server.requests[-1]["max_tokens"], 9)

    def test_errors_propagate(self):
        for args in (("http-error",), ("api-error",), ("bad-response",),
                     ("--stream", "http-error"), ("--stream", "bad-stream"),
                     ("--stream", "truncated-stream")):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
                self.assertNotEqual(result.stdout, "null\n")

    def test_invalid_options(self):
        for args in (("--max-tokens", "oops"), ("--temperature", "null"),
                     ("--max-thinking-tokens", "-2"), ("--schema", "/nonexistent"),
                     ("--max-tokens", "2147483648"),
                     ("--max-thinking-tokens", "2147483648"),
                     ("--file", "/nonexistent"), ("--stream", "--batch"),
                     ("--unknown",), ("--max-tokens",)):
            with self.subTest(args=args):
                result = self.cli(*args, "hello") if args[-1] != "--max-tokens" else self.cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_binary_file_rejected(self):
        path = self.directory / "binary"
        path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n")
        result = self.cli("--file", str(path), "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported file type", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_batch_quotes_and_warmup(self):
        marker = self.directory / "must-not-exist"
        system = f"""It's "$(touch {marker})" system"""
        prompts = ["first prompt", f'$(touch {marker})', 'a "quoted" prompt']
        result = self.cli("--batch", "--parallel", "2", "--system", system,
                          "--temperature", "0.2", *prompts)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())
        requests = self.server.requests
        self.assertEqual(requests[0]["messages"][-1]["content"], ".")
        self.assertEqual(requests[0]["max_tokens"], 1)
        self.assertCountEqual([r["messages"][-1]["content"] for r in requests[1:]], prompts)
        for body in requests:
            self.assertEqual(body["messages"][0]["content"], system)
            self.assertEqual(body["temperature"], 0.2)

    def test_batch_stdin_and_empty_input(self):
        result = self.cli("--batch", stdin="one\ntwo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertCountEqual(result.stdout.splitlines(), ["one", "two"])
        result = self.cli("--batch", stdin="")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_batch_failure(self):
        result = self.cli("--batch", "hello", "http-error")
        self.assertNotEqual(result.returncode, 0)
        result = self.cli("--batch", "--parallel", "0", "hello")
        self.assertNotEqual(result.returncode, 0)

    def test_daemon_startup_failure(self):
        result = self.cli("hello", env={
            "SOCK": str(self.directory / "missing.sock"),
            "LLM_SERVER_EXEC": "/bin/false",
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exited during startup", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            [line.split(": ", 1)[0] for line in result.stderr.splitlines()],
            ["[llm] starting", "[llm] loading", "[llm] failed"],
        )
        self.assertFalse((self.directory / "llama-shared.pid").exists())

    def test_daemon_launch_and_lock_release(self):
        launcher = self.directory / "mock-server"
        launcher.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"$XDG_RUNTIME_DIR/server-args"\nexec python3 ' +
            repr(str(ROOT / "tests/mock_server.py")) + ' "$@"\n'
        )
        launcher.chmod(0o700)
        env = {
            "SOCK": str(self.directory / "managed.sock"),
            "LLM_SERVER_EXEC": str(launcher),
        }
        try:
            result = self.cli("hello", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "hello\n")
            self.assertEqual(
                [line.split(": ", 1)[0] for line in result.stderr.splitlines()],
                ["[llm] starting", "[llm] loading", "[llm] ready"],
            )
            arguments = (self.directory / "server-args").read_text().splitlines()
            self.assertEqual(arguments[arguments.index("--models-preset") + 1], str(self.presets))
            self.assertEqual(arguments[arguments.index("--models-max") + 1], "1")
            self.assertNotIn("-m", arguments)
            self.assertNotIn("--mmproj", arguments)
            self.assertNotIn("--ctx-size", arguments)
            reused = self.cli("hello again", env=env)
            self.assertEqual(reused.returncode, 0, reused.stderr)
            self.assertEqual(reused.stdout, "hello again\n")
            self.assertEqual(reused.stderr, f"[llm] reused: healthy server at {env['SOCK']}\n")
            pid = int((self.directory / "llama-shared.pid").read_text())
            self.assertGreater(pid, 0)
            result = subprocess.run(
                ["flock", "-n", str(self.directory / "llama-shared.lock"), "true"],
                capture_output=True, timeout=2,
            )
            self.assertEqual(result.returncode, 0)
        finally:
            pid_file = self.directory / "llama-shared.pid"
            if pid_file.exists():
                os.kill(int(pid_file.read_text()), 15)

    def test_stop_managed_daemon(self):
        socket = self.directory / "managed.sock"
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tests/mock_server.py"), str(socket)]
        )
        try:
            for _ in range(200):
                if socket.exists():
                    break
                if process.poll() is not None:
                    self.fail("Mock server exited")
                time.sleep(0.01)
            self.assertTrue(socket.exists())
            pid_file = self.directory / "llama-shared.pid"
            pid_file.write_text(f"{process.pid}\n")
            result = subprocess.run(
                ["bash", "-euc", "source lib/daemon.sh; stop_llm_daemon"],
                text=True, capture_output=True, timeout=20, cwd=ROOT,
                env=self.env | {"SOCK": str(socket), "LLM_SERVER_EXEC": sys.executable},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                [line.split(": ", 1)[0] for line in result.stderr.splitlines()],
                ["[llm] stopping", "[llm] stopped"],
            )
            self.assertFalse(pid_file.exists())
            self.assertFalse(socket.exists())
            self.assertEqual(process.wait(timeout=2), -15)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=2)

    def test_stop_rejects_unrelated_pid(self):
        (self.directory / "llama-shared.pid").write_text(f"{os.getpid()}\n")
        result = subprocess.run(
            ["bash", "-euc", "source lib/daemon.sh; stop_llm_daemon"],
            text=True, capture_output=True, timeout=2, cwd=ROOT,
            env=self.env | {"LLM_SERVER_EXEC": "/bin/false"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not identify the configured server", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith("[llm] failed: "))

    def test_daemon_reused_status(self):
        for args, expected in (
            (("hello",), "hello\n"),
            (("--stream", "hello"), "Hello world\n"),
            (("--batch", "hello"), "hello\n"),
        ):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                self.assertEqual(result.stderr, f"[llm] reused: healthy server at {self.socket}\n")

    def test_ensure_daemon_without_generation(self):
        result = self.cli("--ensure-daemon", stdin="unused input")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.server.requests, [])
        self.assertIn("[llm] reused:", result.stderr)
        result = self.cli("--ensure-daemon", "not a prompt")
        self.assertEqual(result.returncode, 2)
        result = self.cli("--ensure-daemon", "--stream")
        self.assertEqual(result.returncode, 2)

    def test_stop_daemon_cli(self):
        socket = self.directory / "managed.sock"
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tests/mock_server.py"), str(socket)]
        )
        try:
            for _ in range(200):
                if socket.exists():
                    break
                if process.poll() is not None:
                    self.fail("Mock server exited")
                time.sleep(0.01)
            self.assertTrue(socket.exists())
            pid_file = self.directory / "llama-shared.pid"
            pid_file.write_text(f"{process.pid}\n")
            result = self.cli(
                "--stop-daemon", "--socket", str(socket), stdin="unused input",
                env={"LLM_SERVER_EXEC": sys.executable, "LLM_SCHEMA": "/missing.yaml"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(
                [line.split(": ", 1)[0] for line in result.stderr.splitlines()],
                ["[llm] stopping", "[llm] stopped"],
            )
            self.assertFalse(pid_file.exists())
            self.assertFalse(socket.exists())
            self.assertEqual(process.wait(timeout=2), -15)
            self.assertEqual(self.server.requests, [])
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=2)

    def test_stop_daemon_cli_errors(self):
        result = self.cli("--stop-daemon", stdin="unused input")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No managed LLM daemon PID file", result.stderr)
        self.assertNotIn("[llm] starting:", result.stderr)
        self.assertEqual(result.stdout, "")
        for args in (
            ("--stop-daemon", "prompt"), ("--stop-daemon", "--ensure-daemon"),
            ("--stop-daemon", "--stream"), ("--stop-daemon", "--batch"),
        ):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
        (self.directory / "llama-shared.pid").write_text(f"{os.getpid()}\n")
        result = self.cli("--stop-daemon", env={"LLM_SERVER_EXEC": "/bin/false"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not identify the configured server", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_daemon_reused_after_lock(self):
        result = subprocess.run(
            ["bash", "-euc", """
                source lib/daemon.sh
                checks=0
                is_llm_daemon_running() {
                    checks=$((checks + 1))
                    [[ $checks -gt 1 ]]
                }
                get_llm_daemon
            """],
            text=True, capture_output=True, timeout=2, cwd=ROOT, env=self.env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, f"[llm] reused: healthy server at {self.socket}\n")

    def test_daemon_validation_failure_status(self):
        result = self.cli("hello", env={
            "SOCK": str(self.directory / "missing.sock"),
            "LLM_SERVER_EXEC": "/bin/false", "LLM_MODELS_PRESET": "/missing-models.ini",
        })
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith("[llm] failed: Model presets not readable: "))
        self.assertNotIn("[llm] starting:", result.stderr)

    def test_daemon_router_options(self):
        override = self.directory / "custom presets.ini"
        override.write_text(self.presets.read_text())
        launcher = self.directory / "capture-options"
        launcher.write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"$XDG_RUNTIME_DIR/server-args"\nexit 1\n'
        )
        launcher.chmod(0o700)
        result = self.cli("--ensure-daemon", env={
            "SOCK": str(self.directory / "missing.sock"),
            "LLM_SERVER_EXEC": str(launcher), "LLM_MODELS_PRESET": str(override),
            "LLM_MODELS_MAX": "2", "LLM_CTX_SIZE": "8192",
            "LLM_PARALLEL": "3", "LLM_BATCH_SIZE": "256", "LLM_IDLE_SLEEP": "120",
        })
        self.assertNotEqual(result.returncode, 0)
        arguments = (self.directory / "server-args").read_text().splitlines()
        for flag, value in (
            ("--models-preset", str(override)), ("--models-max", "2"),
            ("--ctx-size", "8192"), ("--parallel", "3"), ("--threads-http", "6"),
            ("--batch-size", "256"), ("--ubatch-size", "256"), ("--sleep-idle-seconds", "120"),
        ):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        for maximum in ("-1", "no"):
            result = self.cli("--ensure-daemon", env={
                "SOCK": str(self.directory / "missing.sock"),
                "LLM_SERVER_EXEC": "/bin/false", "LLM_MODELS_MAX": maximum,
            })
            self.assertIn("LLM_MODELS_MAX", result.stderr)
            self.assertNotIn("[llm] starting:", result.stderr)
        for variable in ("LLM_MODEL", "LLM_MMPROJ"):
            result = self.cli("--ensure-daemon", env={
                "SOCK": str(self.directory / "missing.sock"),
                "LLM_SERVER_EXEC": "/bin/false", variable: "/legacy.gguf",
            })
            self.assertIn("Move LLM_MODEL and LLM_MMPROJ", result.stderr)
            self.assertNotIn("[llm] starting:", result.stderr)

    def test_daemon_xdg_home_fallback(self):
        env = self.env | {"HOME": str(self.directory), "XDG_CONFIG_HOME": ""}
        result = subprocess.run(
            ["bash", "-euc", 'source lib/daemon.sh; printf "%s" "$LLM_MODELS_PRESET"'],
            text=True, capture_output=True, timeout=2, cwd=ROOT, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.directory / ".config/llm/models.ini"))

    def test_daemon_timeout_status(self):
        launcher = self.directory / "never-ready"
        launcher.write_text("#!/usr/bin/env bash\nexec sleep 60\n")
        launcher.chmod(0o700)
        result = subprocess.run(
            ["bash", "-euc", "source lib/daemon.sh; sleep() { :; }; get_llm_daemon"],
            text=True, capture_output=True, timeout=5, cwd=ROOT,
            env=self.env | {
                "SOCK": str(self.directory / "missing.sock"),
                "LLM_SERVER_EXEC": str(launcher),
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            [line.split(": ", 1)[0] for line in result.stderr.splitlines()],
            ["[llm] starting", "[llm] loading", "[llm] failed"],
        )
        self.assertIn("within 15 seconds", result.stderr)
        self.assertFalse((self.directory / "llama-shared.pid").exists())


if __name__ == "__main__":
    unittest.main()
