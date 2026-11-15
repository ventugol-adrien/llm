#!/usr/bin/env bash

# lib/daemon.sh - Daemon supervision and Unix Domain Socket lifecycle

[[ -n "${_LIB_DAEMON_LOADED:-}" ]] && return 0
readonly _LIB_DAEMON_LOADED=1

LLM_MODELS_PRESET="${LLM_MODELS_PRESET:-${XDG_CONFIG_HOME:-$HOME/.config}/llm/models.ini}"
LLM_MODELS_MAX="${LLM_MODELS_MAX:-1}"
LLM_SERVER_EXEC=${LLM_SERVER_EXEC:-"/home/adrien/Dev/llama.cpp/build/bin/llama-server"}

RUN_DIR="${XDG_RUNTIME_DIR:-/tmp/llm-$(id -u)}"
mkdir -p -m 0700 "$RUN_DIR"

readonly SOCK="${SOCK:-$RUN_DIR/llama-shared.sock}"
readonly LOCK="$RUN_DIR/llama-shared.lock"
readonly PID_FILE="$RUN_DIR/llama-shared.pid"
readonly LOG_FILE="$RUN_DIR/llama-shared.log"

_llm_daemon_status() {
	printf '[llm] %s: %s\n' "$1" "$2" >&2
}

is_llm_daemon_running() {
	[[ -S "$SOCK" ]] && curl -sf --unix-socket "$SOCK" http://localhost/health >/dev/null 2>&1
}

get_llm_daemon() {
	# 1. Fast path: (< 2ms)
	if is_llm_daemon_running; then
		_llm_daemon_status reused "healthy server at $SOCK"
		return 0
	fi

	# 2. Slow path: Serialize initialization using a kernel flock
	(
		flock -x 9 || {
			_llm_daemon_status failed "could not acquire server lifecycle lock"
			exit 1
		}
		if is_llm_daemon_running; then
			_llm_daemon_status reused "healthy server at $SOCK"
			exit 0
		fi

		if ! command -v "$LLM_SERVER_EXEC" >/dev/null 2>&1; then
			_llm_daemon_status failed "Server executable not found: $LLM_SERVER_EXEC"
			exit 1
		fi

		if [[ ! -f "$LLM_MODELS_PRESET" || ! -r "$LLM_MODELS_PRESET" ]]; then
			_llm_daemon_status failed "Model presets not readable: $LLM_MODELS_PRESET"
			echo "Symlink .config/llm into XDG_CONFIG_HOME or set LLM_MODELS_PRESET." >&2
			exit 1
		fi
		if [[ -n "${LLM_MODEL:-}" || -n "${LLM_MMPROJ:-}" ]]; then
			_llm_daemon_status failed "Move LLM_MODEL and LLM_MMPROJ into $LLM_MODELS_PRESET, then unset them"
			exit 1
		fi
		if [[ ! "$LLM_MODELS_MAX" =~ ^[0-9]+$ ]]; then
			_llm_daemon_status failed "LLM_MODELS_MAX must be a nonnegative integer"
			exit 1
		fi
		if [[ -n "${LLM_PARALLEL:-}" && ! "$LLM_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
			_llm_daemon_status failed "LLM_PARALLEL must be a positive integer"
			exit 1
		fi
		rm -f -- "$SOCK" || {
			_llm_daemon_status failed "could not remove stale socket at $SOCK"
			exit 1
		}
		local server_args=(--host "$SOCK" --models-preset "$LLM_MODELS_PRESET" --models-max "$LLM_MODELS_MAX")
		if [[ -n "${LLM_PARALLEL:-}" ]]; then
			local http_threads=$((LLM_PARALLEL * 2 > 4 ? LLM_PARALLEL * 2 : 4))
			server_args+=(--parallel "$LLM_PARALLEL" --threads-http "$http_threads")
		fi
		if [[ -n "${LLM_CTX_SIZE:-}" ]]; then server_args+=(--ctx-size "$LLM_CTX_SIZE"); fi
		if [[ -n "${LLM_BATCH_SIZE:-}" ]]; then server_args+=(--batch-size "$LLM_BATCH_SIZE" --ubatch-size "$LLM_BATCH_SIZE"); fi
		if [[ -n "${LLM_IDLE_SLEEP:-}" ]]; then server_args+=(--sleep-idle-seconds "$LLM_IDLE_SLEEP"); fi

		_llm_daemon_status starting "router using $LLM_MODELS_PRESET"
		nohup "$LLM_SERVER_EXEC" "${server_args[@]}" \
			9>&- >"$LOG_FILE" 2>&1 </dev/null &
		local pid=$!
		if ! printf '%s\n' "$pid" >"$PID_FILE"; then
			_llm_daemon_status failed "could not record server PID in $PID_FILE"
			kill "$pid"
			wait "$pid" || true
			exit 1
		fi
		_llm_daemon_status loading "waiting for router readiness (PID $pid); log: $LOG_FILE"

		# Poll every 25ms until the socket is ready (timeout after 15s)

		for _ in {1..600}; do
			if is_llm_daemon_running; then
				_llm_daemon_status ready "server at $SOCK (PID $pid)"
				exit 0
			fi
			if ! kill -0 "$pid" 2>/dev/null; then
				_llm_daemon_status failed "llama-server exited during startup; see $LOG_FILE"
				rm -f -- "$PID_FILE"
				exit 1
			fi
			sleep 0.025
		done

		_llm_daemon_status failed "Failed to start llama-server within 15 seconds; see $LOG_FILE"
		kill "$pid"
		wait "$pid" || true
		rm -f -- "$PID_FILE"
		exit 1

	) 9>"$LOCK"
}

stop_llm_daemon() (
	flock -x 9 || {
		_llm_daemon_status failed "could not acquire server lifecycle lock"
		return 1
	}
	if [[ ! -f "$PID_FILE" ]]; then
		_llm_daemon_status failed "No managed LLM daemon PID file found"
		return 1
	fi
	local pid executable
	read -r pid <"$PID_FILE" || {
		_llm_daemon_status failed "could not read server PID from $PID_FILE"
		return 1
	}
	executable=$(command -v "$LLM_SERVER_EXEC") || {
		_llm_daemon_status failed "Server executable not found: $LLM_SERVER_EXEC"
		return 1
	}
	if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]] ||
		[[ "$(readlink -f -- "/proc/$pid/exe")" != "$(realpath -- "$executable")" ]]; then
		_llm_daemon_status failed "Managed daemon PID does not identify the configured server"
		return 1
	fi
	kill -TERM "$pid" || {
		_llm_daemon_status failed "could not stop server (PID $pid)"
		return 1
	}
	_llm_daemon_status stopping "server (PID $pid)"
	for _ in {1..1200}; do
		if ! kill -0 "$pid" 2>/dev/null || [[ ! -e "/proc/$pid/exe" ]]; then
			rm -f -- "$SOCK" "$PID_FILE" || {
				_llm_daemon_status failed "could not clean up stopped server files"
				return 1
			}
			_llm_daemon_status stopped "server (PID $pid)"
			return 0
		fi
		sleep 0.025
	done
	_llm_daemon_status failed "LLM daemon did not stop within 15 seconds"
	return 1
) 9>"$LOCK"
