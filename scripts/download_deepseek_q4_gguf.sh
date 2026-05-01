#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
MODEL_DIR="${MODEL_DIR:-$EXP_ROOT/models/lovedheart-deepseek-v4-flash-gguf}"
REPO="${REPO:-lovedheart/DeepSeek-V4-Flash-GGUF}"
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-MXFP4_MOE.gguf}"

cd "$ROOT"

if [[ ! -x .venv/bin/hf ]]; then
  echo "ERROR: .venv is missing. Run scripts/bootstrap_local_env.sh first." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

cat <<EOF
Downloading DeepSeek V4 Flash Q4/MXFP4 GGUF.

Repo:   $REPO
File:   $GGUF_FILE
Target: $MODEL_DIR

Default file is about 140 GiB on disk.
Override REPO/GGUF_FILE/MODEL_DIR to test another GGUF source.
EOF

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
.venv/bin/hf download "$REPO" "$GGUF_FILE" README.md --local-dir "$MODEL_DIR"
