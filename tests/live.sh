#!/usr/bin/env bash
# The live proof (design §9): the node in front of a real Ollama on this machine, in a throwaway data directory.
#   tests/live.sh [MODEL]         MODEL defaults to the first local model Ollama lists (ollama pull qwen3:0.6b)
# OLLAMA_HOST names the daemon (default 127.0.0.1:11434). Nothing of an installed node is touched.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/sllm ] || ./install.sh
tmp="$(mktemp -d)"
export SLLM_HOME="$tmp" SLLM_PORT="${SLLM_PORT:-11439}"
cleanup() { [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null; rm -rf "$tmp"; }
trap cleanup EXIT
.venv/bin/sllm init --name live-proof >/dev/null
.venv/bin/sllm serve >"$tmp/serve.log" 2>&1 &
pid=$!
for _ in $(seq 50); do [ -S "$tmp/data/sllm.sock" ] && break; sleep 0.1; done
[ -S "$tmp/data/sllm.sock" ] || { echo "live.sh: the service did not start:" >&2; cat "$tmp/serve.log" >&2; exit 1; }
.venv/bin/sllm status || true
.venv/bin/sllm prove ${1:+--model "$1"}
