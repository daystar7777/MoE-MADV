#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/hf ]]; then
  echo "ERROR: .venv is missing. Run scripts/bootstrap_local_env.sh first." >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_model_assets.sh metadata qwen
  scripts/download_model_assets.sh metadata deepseek
  scripts/download_model_assets.sh full qwen
  scripts/download_model_assets.sh full deepseek
  scripts/download_model_assets.sh shard deepseek model-00001-of-00033.safetensors

Notes:
  metadata downloads config/index/tokenizer files only.
  full downloads the full Hugging Face repository into models/<name>.
  shard downloads one named file plus the metadata files for targeted Q4 tests.
  qwen full is the source needed by the current Metal engine.
  deepseek full is useful for MLX experiments and future porting analysis.
USAGE
}

MODE="${1:-}"
MODEL="${2:-}"
SHARD="${3:-}"

if [[ -z "$MODE" || -z "$MODEL" ]]; then
  usage
  exit 1
fi

case "$MODEL" in
  qwen)
    REPO="mlx-community/Qwen3.5-397B-A17B-4bit"
    TARGET="qwen3.5-397b-a17b-4bit"
    META_FILES=(config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json generation_config.json)
    ;;
  deepseek)
    REPO="mlx-community/DeepSeek-V4-Flash-4bit"
    TARGET="deepseek-v4-flash-4bit"
    META_FILES=(config.json model.safetensors.index.json tokenizer.json tokenizer_config.json generation_config.json)
    ;;
  *)
    usage
    exit 1
    ;;
esac

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

case "$MODE" in
  metadata)
    mkdir -p "model_meta/$MODEL"
    .venv/bin/hf download "$REPO" "${META_FILES[@]}" --local-dir "model_meta/$MODEL"
    ;;
  full)
    mkdir -p "models/$TARGET"
    .venv/bin/hf download "$REPO" --local-dir "models/$TARGET"
    ;;
  shard)
    if [[ -z "$SHARD" ]]; then
      echo "ERROR: shard mode requires a filename." >&2
      usage
      exit 1
    fi
    mkdir -p "models/$TARGET"
    .venv/bin/hf download "$REPO" "$SHARD" "${META_FILES[@]}" --local-dir "models/$TARGET"
    ;;
  *)
    usage
    exit 1
    ;;
esac
