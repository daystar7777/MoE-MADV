#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAUSE="${PAUSE:-1}"
RUN_MODEL="${RUN_MODEL:-0}"
MODEL="${MODEL_PATH:-$ROOT/../deepseek-v4-experiments/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf}"

pause() {
  if [[ "$PAUSE" == "1" ]]; then
    printf "\nPress return for the next shot..."
    read -r _
  fi
}

section() {
  printf "\n"
  printf "============================================================\n"
  printf "%s\n" "$1"
  printf "============================================================\n\n"
}

say_cmd() {
  printf "$ %s\n" "$*"
}

run_cmd() {
  say_cmd "$*"
  "$@"
}

section "Shot 1: Repository headline"
run_cmd sed -n "1,80p" README.md
pause

section "Shot 2: Exact model source"
run_cmd sed -n "1,90p" docs/model-sources-and-parsers.md
pause

section "Shot 3: Runtime readiness"
if [[ -x "$ROOT/../deepseek-v4-experiments/llama.cpp-deepseek-v4-flash/build/bin/llama-cli" ]]; then
  printf "patched llama.cpp runtime is ready\n"
else
  say_cmd scripts/setup_deepseek_gguf_runtime.sh
  printf "Runtime is not built yet. Run the command above before recording real inference.\n"
fi
pause

section "Shot 4: Local model file"
if [[ -f "$MODEL" ]]; then
  run_cmd ls -lh "$MODEL"
else
  say_cmd scripts/download_deepseek_q4_gguf.sh
  printf "Model file is not present at:\n%s\n" "$MODEL"
fi
pause

section "Shot 5: JSON smoke test"
say_cmd "scripts/run_moe_madv_live_generation_demo.sh"
if [[ "$RUN_MODEL" == "1" ]]; then
  scripts/run_moe_madv_live_generation_demo.sh
else
  printf "Skipping inference. Re-run with RUN_MODEL=1 to capture real generation.\n"
fi
pause

section "Shot 6: Decode baseline command"
say_cmd "scripts/run_deepseek_q4_perf_matrix.py --mode infer --infer-cases no_prewarm_madvise_off,no_prewarm_madvise_on --prompts decode_json_seed,decode_plain_seed --tokens 24 --context 1024 --repeats 3"
printf "\nCompleted baseline summary:\n\n"
run_cmd sed -n "1,80p" docs/results/deepseek_q4_decode_baseline/summary.md
pause

section "Shot 7: Long-run dataset"
run_cmd sed -n "1,70p" docs/results/deepseek_q4_longrun_5h/README.md
pause

section "Shot 8: Rebuild and portability notes"
run_cmd sed -n "1,90p" docs/packed-experts-q4.md
printf "\n"
run_cmd sed -n "1,90p" docs/appendix-other-machines.md
pause

section "Done"
printf "Suggested chart to show next:\n"
printf "docs/assets/deepseek-q4-decode-headline.svg\n"
