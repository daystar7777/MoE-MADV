#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
MODEL_DIR="${MODEL_DIR:-$EXP_ROOT/models/antirez-deepseek-v4-gguf}"
REPO="${REPO:-antirez/deepseek-v4-gguf}"
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"

cd "$ROOT"

if [[ ! -x .venv/bin/hf ]]; then
  echo "ERROR: .venv is missing. Run scripts/bootstrap_local_env.sh first." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"

cat <<EOF
Downloading DeepSeek V4 Flash GGUF.

Repo:   $REPO
File:   $GGUF_FILE
Target: $MODEL_DIR

This file is about 81 GiB on disk.
EOF

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
.venv/bin/hf download "$REPO" "$GGUF_FILE" README.md --local-dir "$MODEL_DIR"
