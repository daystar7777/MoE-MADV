#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-$ROOT/../deepseek-v4-experiments}"
LLAMA_DIR="${LLAMA_DIR:-$EXP_ROOT/llama.cpp-deepseek-v4-flash}"
REPO_URL="${REPO_URL:-https://github.com/antirez/llama.cpp-deepseek-v4-flash.git}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is required. Install it with Homebrew: brew install cmake" >&2
  exit 1
fi

mkdir -p "$EXP_ROOT"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone "$REPO_URL" "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"${JOBS:-8}" --target llama-cli

cat <<EOF
Done.

Runtime:
  $LLAMA_DIR/build/bin/llama-cli

Next:
  scripts/download_deepseek_gguf.sh
  PROMPT="Hello" TOKENS=16 scripts/run_deepseek_gguf_demo.sh
EOF
