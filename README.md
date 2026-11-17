# llm

`llm` is a Bash CLI and Python client for llama-server's OpenAI-compatible
chat-completions API. It manages a local llama-server router, sends requests
over a Unix domain socket, and can select a different configured model for
each request.

Requirements:

- Linux, Bash, curl with `--fail-with-body` support, jq, file, base64, realpath,
  flock, and llama-server
- Python 3.14 or newer and uv for installation, YAML schemas, or the Python API
- A llama.cpp build with `llama-server` and the model backends you need

The daemon default for `LLM_SERVER_EXEC` is
`/home/adrien/Dev/llama.cpp/build/bin/llama-server`; set it to the path of your
own executable when it is elsewhere.

## Installation and setup

From the project directory, create the virtual environment and install the
package and its dependencies:

```sh
uv sync --no-editable
```

Run the installed command with `uv run --no-editable llm`, or run the source
tree CLI directly with `./bin/llm`. The source-tree CLI needs the Python
environment activated (or the required executables on `PATH`) when YAML schema
support is used. To use the Python API, run your program with the project's
`.venv` interpreter or install the package in the environment where the program
runs.

The daemon reads model presets from
`${XDG_CONFIG_HOME:-$HOME/.config}/llm/models.ini`. The example presets are
stored in `.config/llm/models.ini`. To use them without copying the file:

```sh
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}"
ln -s /home/adrien/dev/packages/llm/.config/llm \
  "${XDG_CONFIG_HOME:-$HOME/.config}/llm"
```

If that destination already exists, merge the settings or choose another
`LLM_MODELS_PRESET` path rather than replacing existing configuration. The
included model paths are specific to this machine; update them for your model
directory before starting the router.

Start the router on first use, or start it explicitly:

```sh
./bin/llm --ensure-daemon
./bin/llm --stop-daemon
```

`SOCK` selects the local Unix socket. Its default is
`${XDG_RUNTIME_DIR:-/tmp/llm-UID}/llama-shared.sock`. Export the same socket for
the separate `llm-cli` frontend if you use it:

```sh
export LLAMA_SOCKET="${XDG_RUNTIME_DIR:-/tmp/llm-$(id -u)}/llama-shared.sock"
```

```sh
export LLM_MODEL_NAME=qwen3.5-4bunc
./bin/llm --system "Be concise." "Explain KV caching"
./bin/llm --model qwen3.5-9bunc "Use a different model"
./bin/llm --stream --max-tokens 256 "Write a poem"
./bin/llm --file ./notes.txt --schema ./schema.yaml "Summarize this file"
./bin/llm --batch --parallel 4 "First prompt" "Second prompt"
printf 'First prompt\nSecond prompt\n' | ./bin/llm --batch
```

To call the remote HTTPS endpoint instead of the local Unix-socket CLI:

```sh
curl --fail-with-body --silent --show-error \
  https://llm.taile72000.ts.net/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"qwen3.5-4bunc","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":512,"temperature":0}'
```

## Options

Run `llm --help` for the complete flag list. The prompt options are:

| Flag | Description |
| --- | --- |
| `--model NAME`, `-m NAME` | Select a router preset; overrides `LLM_MODEL_NAME`. |
| `--system TEXT` | Add system instructions. |
| `--max-tokens N` | Set the total generation token limit. |
| `--temperature N` | Set sampling temperature. |
| `--stop-sequence TEXT` | Stop at this one literal string. |
| `--max-thinking-tokens N` | Set the reasoning-token budget. |
| `--file PATH` | Attach a local text, image, audio, or video file; repeatable. |
| `--schema PATH` | Constrain the response using a YAML or JSON schema. |
| `--stream` | Stream the response. |
| `--batch` | Read prompts from arguments or one prompt per stdin line. |
| `--parallel N` | Set the maximum number of concurrent batch requests. |
| `--socket PATH` | Select the daemon's Unix socket. |
| `--ensure-daemon`, `--stop-daemon` | Start or stop the managed router without generating a response. |

Flags override their corresponding environment variables. An explicitly empty
value can clear an environment setting where supported. Request defaults are
`LLM_MAX_TOKENS=4096`, `LLM_TEMPERATURE=0.8`,
`LLM_MAX_THINKING_TOKENS=1024`, and `LLM_MODEL_NAME=qwen3.5-4bunc`.
`LLM_SYSTEM`, `LLM_STOP_SEQUENCE`, `LLM_FILES` (a JSON array of paths), and
`LLM_SCHEMA` have no default value. `--file` replaces the environment file
list. `--stop-sequence` takes one literal string, not JSON.

`--model NAME` (or `-m NAME`) selects a router preset for every request, including streaming and batch warmup. It overrides `LLM_MODEL_NAME`, which defaults to `qwen3.5-4bunc` when unset or empty. An explicitly empty or whitespace-only `--model` is rejected. A model name is not a GGUF path.

`--schema schema.yaml` accepts a standard JSON Schema written in YAML, not a response_format wrapper:

```yaml
type: object
properties:
  answer:
    type: integer
required: [answer]
additionalProperties: false
```

The schema is safely loaded and sent as llama-server's JSON `response_format` with `type: json_schema`. The generated answer is JSON, not YAML. The schema root must be a mapping and its values must be JSON-compatible; quote YAML boolean or numeric mapping keys. Without a schema, output is unconstrained text. `--schema ""` disables an environment-provided schema. `--format` and `LLM_FORMAT` have been replaced by `--schema` and `LLM_SCHEMA`.

`--max-thinking-tokens` maps to llama-server's `reasoning_budget_tokens`: -1
uses the server default, 0 immediately ends reasoning, and positive values cap
reasoning tokens. This requires a model/template with supported thinking
delimiters. It does not replace the total `--max-tokens` limit.

`--max-tokens` accepts -1 for unlimited generation, 0 for prompt-cache evaluation without generation, or a positive token limit.

Local paths are resolved before reading. Text files become text content parts; images, audio, and video are base64-encoded into llama-server's `image_url`, `input_audio`, and `input_video` content parts. Other binary types (including PDF) are rejected. Multimodal input requires a compatible model and `mmproj` in its preset. Video also requires a server built with video support. No server-side `--media-path` access is needed.

## Daemon

A healthy socket is reused; otherwise the CLI starts `LLM_SERVER_EXEC` (default `/home/adrien/Dev/llama.cpp/build/bin/llama-server`) in router mode. `SOCK` or `--socket` overrides the socket path. The default is `${XDG_RUNTIME_DIR:-/tmp/llm-UID}/llama-shared.sock`. Runtime PID, lock, and log files live in that runtime directory. Startup errors identify the log file.

Model presets are stored in [`.config/llm/models.ini`](./.config/llm/models.ini).
Symlink the directory into your XDG configuration directory before starting the daemon:

```sh
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}"
ln -s /home/adrien/dev/packages/llm/.config/llm "${XDG_CONFIG_HOME:-$HOME/.config}/llm"
```

If the destination already exists, merge it rather than overwriting it. The
daemon reads `${XDG_CONFIG_HOME:-$HOME/.config}/llm/models.ini`;
`LLM_MODELS_PRESET` overrides that path. The example presets reference five
local models with absolute paths; update those paths for another machine. Each
named section is the model name clients send in the `model` request field. It
contains the model's GGUF and, for multimodal models, its `mmproj` file. `[*]`
holds shared server options, which a model-specific section can override. See
llama.cpp's server documentation for the supported preset keys.

For example, a minimal preset for another model can be added as:

```ini
[my-model]
model = /absolute/path/to/my-model.gguf
mmproj = /absolute/path/to/mmproj-my-model.gguf
ctx-size = 16384
```

`LLM_MODELS_MAX` defaults to `1`: requesting another model unloads the least recently used model before loading the requested one. Set a larger value only when memory permits, or `0` for no limit. Loading is on demand, so router readiness does not mean a model is loaded. Swaps have loading latency and can interrupt active work; avoid concurrent requests to different models when the limit is one.

Lifecycle transitions are reported on stderr with a `[llm]` prefix: `starting`, `loading`, `ready`, `reused`, `stopping`, `stopped`, and `failed`. Loading is reported once while waiting for router readiness, without periodic progress messages. Status messages do not contaminate stdout, so model output can still be piped to other tools. The daemon library reports the same transitions; health probes remain silent. These are synchronous supervision updates, not continuous monitoring of an idle server.

The supplied presets use `parallel = 1`, `ctx-size = 32768`, batch and ubatch
sizes of 512, and `sleep-idle-seconds = 960`. Optional `LLM_PARALLEL`,
`LLM_CTX_SIZE`, `LLM_BATCH_SIZE`, `LLM_IDLE_SLEEP`, and
`LLM_SPEC_DRAFT_N_MAX` environment overrides are passed as server arguments,
taking precedence over presets. These affect newly started routers, not an
existing one. `LLM_SOCKET_UMASK` optionally sets the server process umask when
creating the Unix socket; it must be four octal digits. `--parallel` also sets
batch concurrency (default 1 for the CLI); when sourcing only the client
library, batch concurrency defaults to 4.

To share the Unix socket with a reverse proxy in another container, keep the
default private runtime directory for the PID, lock, and log files, but place
the socket in a dedicated group-shared directory:

```sh
sudo groupadd --system llm-proxy
sudo install -d -o "$USER" -g llm-proxy -m 2770 /run/llm-proxy
SOCK=/run/llm-proxy/llama-shared.sock LLM_SOCKET_UMASK=0007 \
  ./bin/llm --ensure-daemon
```

Bind-mount `/run/llm-proxy` into the proxy container at `/host-sockets`, and
add the directory's numeric group ID as a supplemental group for its Nginx
worker. The setgid directory makes the socket inherit that group, and umask
`0007` allows group read/write while denying access to others. Create the
directory again at boot before starting the daemon because `/run` is
ephemeral. Check directory traversal and socket permissions with `stat`;
avoid making the socket world-accessible.

The Qwen3.8 preset uses embedded MTP heads (`spec-type = draft-mtp`) with a
three-token draft window. The Qwen3.5 27B preset does not enable a separate
drafter by default: the installed Qwen3.5 4B drafter did not improve speed in
testing. For another target, set `spec-type = draft-mtp` when the model
includes MTP heads, or use `spec-type = draft-simple` together with
`model-draft` for a separate compatible draft model. `LLM_SPEC_DRAFT_N_MAX`
can override the per-preset draft window for a newly started router.

To migrate an already-running single-model daemon, first install the config symlink above, then run these commands on the server host from this repository:

```sh
./bin/llm --stop-daemon
unset LLM_MODEL LLM_MMPROJ
./bin/llm --ensure-daemon
```

`LLM_MODEL` and `LLM_MMPROJ` are no longer startup options: put their paths in a named preset. Startup rejects these legacy overrides rather than silently ignoring them. Router requests always include a model name; use `--model NAME` or `LLM_MODEL_NAME` instead of restarting the daemon for each swap. Changing these request settings does not change the running router's configuration. The separate `llm-cli` frontend also accepts `llm llama --model NAME`, but does not provide this package's `--stop-daemon`/`--ensure-daemon` flags, so use the explicit package executable for lifecycle commands.

`--stop-daemon` does not generate a completion or start a server. It uses the
tracked PID and verifies the configured server executable before sending
SIGTERM. It reports shutdown on stderr and fails if no managed PID exists or
the PID does not match. Active requests and streams may be interrupted. The
socket path stays the same after restarting, so a proxy using that socket does
not need to change.

## Router management

The router exposes the usual `/models`, `/models/load`, and `/models/unload`
endpoints. `LLM_MODELS_MAX=1` means requesting a different model unloads the
least recently used running model before loading the new one. `llm --stop-daemon`
stops the entire router; it does not unload only one model.

The separate `llm-cli` frontend includes a `manage` command for these endpoints:

```sh
llm manage status
llm manage load --model qwen3.5-9bunc
llm manage swap --model gemma4-31bunc
llm manage unload --model gemma4-31bunc
```

The package's `bin/llm` focuses on prompting and daemon lifecycle and does not
include that `manage` subcommand. Model selection on a prompt loads the selected
preset automatically when router autoload is enabled. Explicit `load` can
preload a model; `unload` releases it, but the next request can load it again.
With a one-model limit, do not assume two requests for different models can run
simultaneously. Avoid switching models while requests using the current model
are active.

## Library

The Bash functions send prompts through the socket and assume the router is
already running. The CLI starts the daemon automatically. Export
`LLM_MODEL_NAME` to choose their default router preset:

```sh
source lib/client.sh
export LLM_MODEL_NAME=qwen3.5-4bunc
call_one "Prompt" "System" 512 0.8 "STOP" 128 '["./notes.txt"]' ./schema.yaml
stream_one "Prompt" "System" 512
run_batch "System" 512 0.8 "" "" '[]' ./schema.yaml "Prompt one" "Prompt two"
```

`call_one` and `stream_one` accept prompt, system, max_tokens, temperature,
stop_sequence, max_thinking_tokens, files, and schema file path. Omitted
options use the corresponding environment variable. `run_batch` accepts those
same seven request options without a prompt, followed by prompt arguments; with
no prompts it reads one per stdin line. To pass prompt arguments, supply all
seven option positions. Output order is unspecified, and concurrent batch
output may interleave. A nonempty system prompt is warmed with a one-token
request before fan-out without a schema; cache reuse and slot selection remain
server-controlled.

All library requests use `LLM_MODEL_NAME` (default `qwen3.5-4bunc`), including batch warmup and worker processes. For example, `LLM_MODEL_NAME=qwen3.5-9bunc call_one "Prompt" "System" 512` selects that preset.

Source `lib/daemon.sh` for `get_llm_daemon`, `is_llm_daemon_running`, and
`stop_llm_daemon`. Stop only acts on a tracked PID matching the configured
server executable; it does not stop an independently launched server.

## Python core

```python
import asyncio
from llm import LLMEngine

async def main():
    async with LLMEngine(parallel=2) as engine:
        answer = await engine.generate(
            "What is 2 + 2?", temperature=0, schema="./schema.yaml", model="qwen3.5-4bunc",
        )
        answers = await engine.generate_parallel(
            ["What is 1 + 1?", "What is 2 + 2?"], system="Be concise.",
        )
        async for text in engine.stream("Write a poem", max_tokens=128):
            print(text, end="", flush=True)

asyncio.run(main())
```

`generate`, `generate_parallel`, and `stream` accept the CLI request options:
`system`, `max_tokens`, `temperature`, `stop_sequence`, `max_thinking_tokens`,
`files` (a sequence of local paths), and `schema` (a YAML or JSON schema path).
`None` uses the corresponding environment variable. Empty
`system`/`stop_sequence`/`schema` strings and an empty `files` list override
environment values. Defaults match the CLI: 4096 total tokens, 1024 thinking
tokens, temperature 0.8, and model `qwen3.5-4bunc`.

The keyword-only `model` option selects a preset, overriding `LLM_MODEL_NAME`. `None` uses the environment or `qwen3.5-4bunc`; blank names are rejected. Batch requests and their warmup use the same selected model.

`LLMEngine(parallel=None, sock=None)` uses `LLM_PARALLEL` (1) and `SOCK`, or
the CLI's runtime socket default. `LLM_BASE_URL` overrides the HTTP base URL,
still using the Unix socket; it is a URL prefix, not an alternative transport.
Connections are pooled and generation/streaming concurrency is bounded by
`parallel`. Batch results preserve input order, shared system prompts are
warmed without files/schema/stop strings, and failed batches cancel outstanding
requests.

`stream` yields text chunks without adding a newline and checks SSE errors and the terminal `[DONE]` event. `stream_raw(payload)` forwards raw SSE bytes for API clients, forcing `stream: true` and adding the default model only if `model` is absent, without modifying the caller's payload. Explicit model values must be nonempty strings. The recipient must interpret streamed API errors. Close partially consumed generators with `aclose()` (or `contextlib.aclosing`) to release their connection and concurrency slot promptly.

`ensure_started()` checks health and invokes the supervisor through
`llm --ensure-daemon` when needed, propagating lifecycle updates on stderr and
raising exceptions on failures. The supervisor-only CLI flag accepts no
prompt and produces no completion. Closing the engine or leaving its async
context closes HTTP connections only; it does not stop the shared daemon.
Engines should be used within one event loop and closed after active
requests/streams finish.

## Errors and troubleshooting

- **Preset file not found:** create the XDG symlink, or set
  `LLM_MODELS_PRESET` to a readable `models.ini` file.
- **Legacy model override rejected:** remove `LLM_MODEL` and `LLM_MMPROJ`;
  define the model and projection paths in the matching preset section.
- **Model not found:** the request's `--model` / `LLM_MODEL_NAME` must exactly
  match a preset name (or another model registered with the router).
- **Router startup failed:** inspect the log path printed after `starting` or
  `loading`. Preset parse errors and missing model files are reported there.
- **Request failed:** curl and server errors are sent to stderr and return a
  nonzero status; check the selected model name and server log.
- **Socket already has a server:** a healthy socket is reused. Stop the
  managed daemon before changing startup options or presets; those options do
  not reconfigure a running process.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use a mock HTTP server over a temporary Unix socket; no model is required.
Run from the project root after `uv sync --no-editable`. The tests cover the
Bash CLI, Python API, schema conversion, daemon supervision, and error handling
without downloading models or contacting an inference service.