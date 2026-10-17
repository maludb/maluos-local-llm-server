#!/usr/bin/env bash
# The node's proof without Ollama (design §9): pytest against the fake backend. Exit non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pytest -q "$@" tests
