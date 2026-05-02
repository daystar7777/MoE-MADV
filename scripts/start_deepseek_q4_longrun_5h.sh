#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/logs/deepseek_q4_longrun_5h_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

exec > "$OUT_DIR/longrun.out" 2>&1

ARGS=(
  scripts/run_deepseek_q4_longrun.py
  --duration-hours "${DURATION_HOURS:-5}"
  --sleep-s "${SLEEP_S:-45}"
  --suites "${SUITES:-prefill,decode}"
  --cases "${CASES:-no_prewarm_madvise_off,best_gap4_chunk1_madvise_on,gap4_chunk1_madvise_off,no_prewarm_madvise_on}"
  --out-dir "$OUT_DIR"
)

if [[ "${NO_TRACE:-0}" == "1" ]]; then
  ARGS+=(--no-trace)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "root=$ROOT"
echo "out_dir=$OUT_DIR"
echo "args=${ARGS[*]}"
exec python3 "${ARGS[@]}"
