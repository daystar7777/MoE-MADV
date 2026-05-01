#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/hf ]]; then
  echo "ERROR: .venv is missing. Run scripts/bootstrap_local_env.sh first." >&2
  exit 1
fi

REPO="${REPO:-mlx-community/DeepSeek-V4-Flash-4bit}"
TARGET="${TARGET:-models/deepseek-v4-flash-4bit}"
START="${START:-1}"
END="${END:-33}"
META_FILES=(config.json model.safetensors.index.json tokenizer.json tokenizer_config.json generation_config.json)

# The multi-file HF transfer path can stall on very large repos. Default to the
# standard downloader here; callers can opt back into hf_transfer explicitly.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "$TARGET"

.venv/bin/hf download "$REPO" "${META_FILES[@]}" --local-dir "$TARGET"

for i in $(seq "$START" "$END"); do
  shard="$(printf 'model-%05d-of-00033.safetensors' "$i")"
  if [[ -f "$TARGET/$shard" ]]; then
    echo "SKIP $shard already exists"
    continue
  fi
  echo
  echo "Downloading $shard ($i/$END)"
  .venv/bin/hf download "$REPO" "$shard" --local-dir "$TARGET"
done
