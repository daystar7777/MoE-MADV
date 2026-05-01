#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKED_DIR="${PACKED_DIR:-$ROOT/models/deepseek-v4-flash-4bit/packed_experts_q4}"

cd "$ROOT/metal_infer"
make deepseek-q4-probe
./deepseek_q4_probe --packed-dir "$PACKED_DIR" "$@"
