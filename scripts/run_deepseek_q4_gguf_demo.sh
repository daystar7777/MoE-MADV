#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
MODEL_PATH="${MODEL_PATH:-$EXP_ROOT/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf}"

MODEL_PATH="$MODEL_PATH" "$ROOT/scripts/run_deepseek_gguf_demo.sh" "$@"
