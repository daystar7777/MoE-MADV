#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen3.5-397b-a17b-4bit}"
TOKENS="${TOKENS:-40}"
PROMPT="${PROMPT:-Explain quantum computing in simple terms.}"

cd "$ROOT/metal_infer"

if [[ ! -x ./infer ]]; then
  make
fi

./infer \
  --model "$MODEL_DIR" \
  --weights "$ROOT/metal_infer/model_weights.bin" \
  --manifest "$ROOT/metal_infer/model_weights.json" \
  --prompt "$PROMPT" \
  --tokens "$TOKENS" \
  "$@"
