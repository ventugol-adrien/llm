#!/usr/bin/env bash

# lib/client.sh - curl + jq operations over a Unix domain socket

[[ -n "${_LIB_CLIENT_LOADED:-}" ]] && return 0
readonly _LIB_CLIENT_LOADED=1
readonly _LLM_CLIENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

_llm_schema_format() {
	local prefix python
	prefix=$(dirname -- "$_LLM_CLIENT_DIR")
	if [[ -x "$prefix/.venv/bin/python" ]]; then
		python="$prefix/.venv/bin/python"
	elif [[ -x "$_LLM_CLIENT_DIR/../../../bin/python" ]]; then
		python="$_LLM_CLIENT_DIR/../../../bin/python"
	else
		python=python3
	fi
	"$python" "$_LLM_CLIENT_DIR/schema.py" "$1"
}

_llm_payload() (
	set -o pipefail
	local prompt="${1:?prompt is required}"
	local system="${2-${LLM_SYSTEM:-}}"
	local max_tokens="${3-${LLM_MAX_TOKENS:-}}"
	local temperature="${4-${LLM_TEMPERATURE:-0.8}}"
	local stop_sequence="${5-${LLM_STOP_SEQUENCE:-}}"
	local max_thinking_tokens="${6-${LLM_MAX_THINKING_TOKENS:-}}"
	local files="${7-${LLM_FILES:-[]}}"
	local schema="${8-${LLM_SCHEMA:-}}"
	local stream="${9:-false}"
	local model="${LLM_MODEL_NAME:-qwen3.5-4bunc}"
	if [[ "$model" != *[![:space:]]* ]]; then
		printf 'model must be a nonempty name\n' >&2
		return 1
	fi
	local paths path mime response_format work
	work=$(mktemp -d) || return
	trap 'rm -f -- "$work/parts" "$work/prompt" "$work/system" "$work/stop"; rmdir -- "$work"' EXIT
	printf '%s' "$prompt" >"$work/prompt"
	printf '%s' "$system" >"$work/system"
	printf '%s' "$stop_sequence" >"$work/stop"
	jq -nc --rawfile p "$work/prompt" '{type:"text", text:$p}' >"$work/parts" || return

	response_format='{"type":"text"}'
	if [[ -n "$schema" ]]; then
		schema=$(realpath -e -- "$schema") || return
		response_format=$(_llm_schema_format "$schema") || return
	fi

	paths=$(jq -ce '
		if type == "array" and all(.[]; type == "string" and length > 0)
		then . else error("files must be a JSON array of nonempty paths") end
	' <<<"$files") || return
	while IFS= read -r -d '' path; do
		path=$(realpath -e -- "$path") || return
		if [[ ! -f "$path" || ! -r "$path" ]]; then
			printf 'Not a readable file: %s\n' "$path" >&2
			return 1
		fi
		mime=$(file --brief --mime-type -- "$path") || return
		case "$mime" in
		image/* | audio/* | video/*)
			base64 -w 0 -- "$path" | jq -Rs --arg mime "$mime" '
					. as $data |
					if $mime | startswith("image/") then
						{type:"image_url", image_url:{url:("data:" + $mime + ";base64," + $data)}}
					elif $mime | startswith("audio/") then
						{type:"input_audio", input_audio:{data:$data}}
					else {type:"input_video", input_video:{data:$data}} end
				' >>"$work/parts" || return
			;;
		text/* | application/json | application/xml | application/javascript | inode/x-empty)
			jq -nc --arg path "$path" --rawfile text "$path" \
				'{type:"text", text:("File: " + $path + "\n" + $text)}' >>"$work/parts" || return
			;;
		*)
			printf 'Unsupported file type %s: %s\n' "$mime" "$path" >&2
			return 1
			;;
		esac
	done < <(jq -rj '.[] | ., "\u0000"' <<<"$paths")

	jq -nc --rawfile p "$work/prompt" --rawfile s "$work/system" \
		--slurpfile content "$work/parts" --argjson files "$paths" \
		--argjson max "$max_tokens" --argjson temp "$temperature" \
		--rawfile stop "$work/stop" --arg thinking "$max_thinking_tokens" \
		--argjson format "$response_format" --argjson stream "$stream" --arg model "$model" '
		def integer: type == "number" and floor == .;
		if ($max | integer) and $max >= -1 and $max <= 2147483647 then .
		else error("max_tokens must be an integer between -1 and 2147483647") end |
		if ($temp | type) == "number" then .
		else error("temperature must be a number") end |
		($thinking | if . == "" then null else tonumber end) as $budget |
		if $budget == null or (($budget | integer) and $budget >= -1 and $budget <= 2147483647) then .
		else error("max_thinking_tokens must be an integer between -1 and 2147483647") end |
		{
			model:$model,
			messages: (
				(if $s == "" then [] else [{role:"system", content:$s}] end) +
				[{role:"user", content:(if $files | length == 0 then $p else $content end)}]
			),
			max_tokens:$max, temperature:$temp, cache_prompt:true,
			stream:$stream, response_format:$format
		} +
		(if $stop == "" then {} else {stop:[$stop]} end) +
		(if $budget == null then {} else {reasoning_budget_tokens:$budget} end)
	'
)

call_one() (
	set -o pipefail
	local payload sock_target="${SOCK:-${XDG_RUNTIME_DIR:-/tmp/llm-$(id -u)}/llama-shared.sock}"
	payload=$(_llm_payload "$@") || return
	curl --silent --show-error --fail-with-body --unix-socket "$sock_target" \
		-H 'Content-Type: application/json' --data-binary @- \
		http://localhost/v1/chat/completions <<<"$payload" |
		jq -er '
			if .error then error(.error | tostring)
			elif (.choices[0].message.content | type) == "string" then .choices[0].message.content
			else error("response has no assistant text content") end
		'
)

stream_one() (
	set -o pipefail
	local payload sock_target="${SOCK:-${XDG_RUNTIME_DIR:-/tmp/llm-$(id -u)}/llama-shared.sock}"
	payload=$(_llm_payload "${1:?prompt is required}" "${2-${LLM_SYSTEM:-}}" \
		"${3-${LLM_MAX_TOKENS:-}}" "${4-${LLM_TEMPERATURE:-0.8}}" \
		"${5-${LLM_STOP_SEQUENCE:-}}" "${6-${LLM_MAX_THINKING_TOKENS:-}}" \
		"${7-${LLM_FILES:-[]}}" "${8-${LLM_SCHEMA:-}}" true) || return
	curl --silent --show-error --fail-with-body --no-buffer --unix-socket "$sock_target" \
		-H 'Content-Type: application/json' --data-binary @- \
		http://localhost/v1/chat/completions <<<"$payload" |
		jq --unbuffered -Rnj '
			foreach ((inputs | sub("\r$"; "") | select(startswith("data:")) |
				ltrimstr("data:") | ltrimstr(" ") |
				if . == "[DONE]" then {done:true} else fromjson end), null) as $event
				(false;
					if $event == null then
						if . then . else error("stream ended without [DONE]") end
					elif $event.done then true else . end;
					if $event == null or $event.done then empty
					elif $event.error then error($event.error | tostring)
					elif ($event.choices | type) != "array" then error("invalid stream event")
					else $event.choices[0].delta.content // empty end
				)
		' || return
	printf '\n'
)

run_batch() (
	set -o pipefail
	local system="${1-${LLM_SYSTEM:-}}" max_tokens="${2-${LLM_MAX_TOKENS:-512}}"
	local temperature="${3-${LLM_TEMPERATURE:-0.8}}" stop_sequence="${4-${LLM_STOP_SEQUENCE:-}}"
	local thinking="${5-${LLM_MAX_THINKING_TOKENS:-}}" files="${6-${LLM_FILES:-[]}}"
	local schema="${7-${LLM_SCHEMA:-}}" parallel="${LLM_PARALLEL:-4}"
	if [[ ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
		printf 'LLM_PARALLEL must be a positive integer\n' >&2
		return 1
	fi
	if (($# < 7)); then set --; else shift 7; fi
	if [[ -n "$system" ]]; then
		call_one "." "$system" 1 "$temperature" "" "$thinking" '[]' "" >/dev/null || return
	fi
	if [[ -z "${SOCK:-}" ]]; then
		SOCK="${XDG_RUNTIME_DIR:-/tmp/llm-$(id -u)}/llama-shared.sock"
	fi
	export SOCK _LLM_CLIENT_DIR
	export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3.5-4bunc}"
	export -f _llm_payload _llm_schema_format call_one
	if (($# > 0)); then
		printf '%s\0' "$@"
	else
		tr '\n' '\0'
	fi | xargs -0 -r -n 1 -P "$parallel" bash -c \
		'call_one "${@:8:1}" "${@:1:7}"' _ \
		"$system" "$max_tokens" "$temperature" "$stop_sequence" "$thinking" "$files" "$schema"
)
