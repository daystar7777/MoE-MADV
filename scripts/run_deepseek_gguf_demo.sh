#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
LLAMA_DIR="${LLAMA_DIR:-$EXP_ROOT/llama.cpp-deepseek-v4-flash}"
MODEL_PATH="${MODEL_PATH:-$EXP_ROOT/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PROMPT="${PROMPT:-Hello}"
TOKENS="${TOKENS:-16}"
CONTEXT="${CONTEXT:-4096}"
GPU_LAYERS="${GPU_LAYERS:-0}"
LOGS="${LOGS:-0}"
TIMINGS="${TIMINGS:-1}"
STDERR="${STDERR:-1}"

BIN="$LLAMA_DIR/build/bin/llama-cli"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $(basename "$BIN") is missing. Run scripts/setup_deepseek_gguf_runtime.sh first." >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: GGUF model not found at $MODEL_PATH" >&2
  echo "Run: scripts/download_deepseek_gguf.sh" >&2
  exit 1
fi

ARGS=(
  "$BIN"
  -m "$MODEL_PATH"
  -p "$PROMPT"
  -n "$TOKENS"
  -c "$CONTEXT"
  -ngl "$GPU_LAYERS"
  --no-warmup
  --simple-io
  --no-display-prompt
  -st
)

if [[ -n "${THREADS:-}" ]]; then
  ARGS+=(-t "$THREADS")
fi

if [[ "$LOGS" != "1" ]]; then
  ARGS+=(--log-disable)
fi

if [[ "$TIMINGS" != "1" ]]; then
  ARGS+=(--no-show-timings)
fi

if [[ "$STDERR" == "1" ]]; then
  exec "${ARGS[@]}" "$@"
else
  exec "${ARGS[@]}" "$@" 2>/dev/null
fi
