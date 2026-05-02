#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
MODEL_DIR="${MODEL_DIR:-$EXP_ROOT/models/lovedheart-deepseek-v4-flash-gguf}"
REPO="${REPO:-lovedheart/DeepSeek-V4-Flash-GGUF}"
REVISION="${REVISION:-cd42deba41ac0536e68b125dfc367197b0ec3038}"
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
URL:    https://huggingface.co/$REPO/blob/$REVISION/$GGUF_FILE
Rev:    $REVISION
File:   $GGUF_FILE
Target: $MODEL_DIR

Default file is 150,225,324,672 bytes: 150.23 GB / 139.91 GiB.
Override REPO/REVISION/GGUF_FILE/MODEL_DIR to test another GGUF source.
EOF

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
.venv/bin/hf download "$REPO" "$GGUF_FILE" README.md --revision "$REVISION" --local-dir "$MODEL_DIR"
