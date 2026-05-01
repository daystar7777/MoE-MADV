#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

cd "$ROOT"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-local.txt

echo "Ready: $ROOT/.venv"
echo "MLX: $(.venv/bin/python - <<'PY'
import mlx.core as mx
print(mx.__version__)
PY
)"
