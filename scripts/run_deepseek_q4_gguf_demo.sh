#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
MODEL_PATH="${MODEL_PATH:-$EXP_ROOT/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf}"

# Match flash-moe's winning memory strategy: keep llama.cpp's own heap small
# and let macOS use the remaining RAM as reclaimable file-backed page cache.
export GGML_DISABLE_CPU_REPACK="${GGML_DISABLE_CPU_REPACK:-1}"
export LLAMA_MMAP_RANDOM="${LLAMA_MMAP_RANDOM:-0}"
export GGML_MOE_MADVISE_WILLNEED="${GGML_MOE_MADVISE_WILLNEED:-1}"

if [[ "${PREWARM_EXPERTS:-0}" == "1" ]]; then
  PREWARM_ARGS=(
    --gguf "$MODEL_PATH"
    --budget-gib "${PREWARM_BUDGET_GIB:-auto}"
    --leave-gib "${PREWARM_LEAVE_GIB:-13}"
    --max-auto-gib "${PREWARM_MAX_AUTO_GIB:-8}"
    --experts-per-layer "${PREWARM_EXPERTS_PER_LAYER:-16}"
    --merge-gap-mib "${PREWARM_MERGE_GAP_MIB:-4}"
    --chunk-mib "${PREWARM_CHUNK_MIB:-1}"
    --prompt "${PREWARM_PROMPT:-$PROMPT}"
  )
  if [[ -n "${PREWARM_PROFILE:-}" ]]; then
    PREWARM_ARGS+=(--profile "$PREWARM_PROFILE")
  fi
  if [[ -n "${PREWARM_HOTSET_JSON:-}" ]]; then
    PREWARM_ARGS+=(--hotset-json "$PREWARM_HOTSET_JSON")
  fi
  if [[ -n "${PREWARM_LAYERS:-}" ]]; then
    PREWARM_ARGS+=(--layers "$PREWARM_LAYERS")
  fi
  if [[ "${PREWARM_DRY_RUN:-0}" == "1" ]]; then
    PREWARM_ARGS+=(--dry-run)
  fi
  "$ROOT/scripts/warm_deepseek_q4_expert_cache.py" "${PREWARM_ARGS[@]}"
  if [[ "${PREWARM_ONLY:-0}" == "1" ]]; then
    exit 0
  fi
fi

DEFAULT_ARGS=()
if [[ "${DEEPSEEK_Q4_OPT_DEFAULTS:-1}" == "1" ]]; then
  DEFAULT_ARGS=(
    --fit off
    --device none
    --no-op-offload
    -b "${BATCH:-512}"
    -ub "${UBATCH:-64}"
    --cache-ram "${CACHE_RAM:-0}"
  )
fi

MODEL_PATH="$MODEL_PATH" "$ROOT/scripts/run_deepseek_gguf_demo.sh" "${DEFAULT_ARGS[@]}" "$@"
