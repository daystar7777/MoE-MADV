#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JSON_TOKENS="${JSON_TOKENS:-12}"
PLAIN_TOKENS="${PLAIN_TOKENS:-32}"
JSON_PROMPT="${JSON_PROMPT:-Return JSON only: {\"status\":\"ok\",\"model\":\"DeepSeek V4 Flash\"}}"
PLAIN_PROMPT="${PLAIN_PROMPT:-In one sentence, explain why MoE page loading matters for local inference.}"

MODEL="${MODEL_PATH:-$ROOT/../deepseek-v4-experiments/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf}"

section() {
  printf "\n"
  printf "============================================================\n"
  printf "%s\n" "$1"
  printf "============================================================\n\n"
}

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

run_generation() {
  local title="$1"
  local prompt="$2"
  local tokens="$3"
  local started
  local ended
  local elapsed

  section "$title"
  printf "Clock time: %s\n" "$(timestamp)"
  printf "Prompt:\n%s\n\n" "$prompt"

  started=$SECONDS
  PROMPT="$prompt" TOKENS="$tokens" TIMINGS=1 STDERR=1 \
    scripts/run_deepseek_q4_gguf_demo.sh
  ended=$SECONDS
  elapsed=$((ended - started))

  printf "\nCompleted at: %s\n" "$(timestamp)"
  printf "Elapsed wall time: %ss\n" "$elapsed"
}

if [[ ! -f "$MODEL" ]]; then
  printf "ERROR: model file not found at:\n%s\n\n" "$MODEL" >&2
  printf "Run scripts/download_deepseek_q4_gguf.sh first.\n" >&2
  exit 1
fi

section "Local model file"
ls -lh "$MODEL"
printf "Clock time: %s\n" "$(timestamp)"

run_generation "Live generation 1: JSON" "$JSON_PROMPT" "$JSON_TOKENS"
run_generation "Live generation 2: plain English" "$PLAIN_PROMPT" "$PLAIN_TOKENS"
