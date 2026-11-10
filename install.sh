#!/usr/bin/env bash
# A development checkout: a virtual environment in .venv with the node and its test dependencies, pinned by requirements.lock.
# The packaged install for a host (a service user, launchd / systemd units) is docs/estate-node-plan.md N4 and N5.
#   ./install.sh            install from the lock file
#   ./install.sh --relock   install the newest versions pyproject.toml allows and rewrite the lock file
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' || { echo "install.sh: Python 3.12 or newer is needed ($("$PY" --version 2>&1)); set PYTHON=/path/to/python3.12" >&2; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip -q install --upgrade pip
if [ "${1:-}" = "--relock" ] || [ ! -f requirements.lock ]; then
    .venv/bin/python -m pip -q install -e '.[test]'
    { echo "# Written by ./install.sh --relock; the versions the proofs last passed with."; .venv/bin/python -m pip freeze --exclude-editable; } > requirements.lock
    echo "requirements.lock rewritten"
else
    .venv/bin/python -m pip -q install -r requirements.lock
    .venv/bin/python -m pip -q install --no-deps -e .
fi
echo "installed: .venv/bin/sllm ($(.venv/bin/sllm --version))"
echo "next:      .venv/bin/sllm init --name \"\$(hostname -s)\" && .venv/bin/sllm serve"
