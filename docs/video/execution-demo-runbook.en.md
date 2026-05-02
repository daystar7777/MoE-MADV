# Execution Demo Runbook

This runbook is designed for recording an English video. It shows the execution
process without requiring the full 5-hour benchmark to run on camera.

## Recording Setup

Recommended terminal settings:

- Resolution: 1920x1080 or 2560x1440
- Font size: 18-22 pt
- Shell prompt: short path, no private tokens
- Capture: terminal window plus browser/editor for charts

Suggested opening command:

```bash
cd /path/to/MoE-MADV
git status -sb
```

Optional guided tour helper:

```bash
# Does not run inference by default.
scripts/run_moe_madv_video_tour.sh

# Runs the two short smoke tests as part of the tour.
RUN_MODEL=1 scripts/run_moe_madv_video_tour.sh
```

Recommended live generation helper:

```bash
scripts/run_moe_madv_live_generation_demo.sh
```

Recommended sidecar monitor:

```bash
scripts/open_moe_madv_recording_apps.sh

OUT=logs/video_monitor_$(date +%Y%m%d_%H%M%S).csv \
  scripts/run_moe_madv_resource_monitor.sh
```

For a Shorts edit, record the full generation but speed up the slow middle
section by 6x to 12x. Keep the first few generated tokens and final timing line
at normal speed.

## Shot 1: Show The Repository

```bash
sed -n '1,80p' README.md
```

Narration point:

```text
This repository documents running a 150GB DeepSeek V4 Flash MoE GGUF on a
64GB M1 Max, using routing-aware page-in hints.
```

## Shot 2: Show Exact Model Source

```bash
sed -n '1,90p' docs/model-sources-and-parsers.md
```

Narration point:

```text
The final benchmark model is the lovedheart DeepSeek-V4-Flash-GGUF
MXFP4_MOE file. The model itself is not committed to GitHub.
```

## Shot 3: Build Runtime

Use this if recording on a fresh checkout:

```bash
scripts/setup_deepseek_gguf_runtime.sh
```

If the runtime is already built, use:

```bash
test -x ../deepseek-v4-experiments/llama.cpp-deepseek-v4-flash/build/bin/llama-cli \
  && echo "patched llama.cpp runtime is ready"
```

## Shot 4: Download Model

Use this if recording the real download:

```bash
scripts/download_deepseek_q4_gguf.sh
```

For a short video, it is usually better to show the command and then confirm the
local file:

```bash
MODEL="../deepseek-v4-experiments/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf"
ls -lh "$MODEL"
```

Expected talking point:

```text
The local file is about 139.91 GiB, displayed as 150GB on Hugging Face.
```

## Shot 5: Live Token Generation

Use this as a main proof shot. It shows two short prompts and lets tokens stream
from the local model.

```bash
scripts/run_moe_madv_live_generation_demo.sh
```

If you prefer to record each prompt separately, use the commands below.

### JSON Prompt

```bash
PROMPT='Return JSON only: {"status":"ok"}' TOKENS=8 \
  scripts/run_deepseek_q4_gguf_demo.sh
```

What to capture:

- model load beginning;
- generated JSON-like output;
- timing lines if visible.
- the sidecar monitor clock and elapsed time.

### Plain Text Prompt

```bash
PROMPT='Write one short sentence about local AI inference.' TOKENS=24 \
  scripts/run_deepseek_q4_gguf_demo.sh
```

What to capture:

- generated plain English text;
- token timing line.
- Activity Monitor Memory and GPU History beside the terminal.

## Shot 6: Reproduce Decode Baseline Command

This is short enough to show, but it may still take several minutes.

```bash
scripts/run_deepseek_q4_perf_matrix.py \
  --mode infer \
  --infer-cases no_prewarm_madvise_off,no_prewarm_madvise_on \
  --prompts decode_json_seed,decode_plain_seed \
  --tokens 24 \
  --context 1024 \
  --repeats 3
```

For a tighter video, show the command but cut to the completed result:

```bash
sed -n '1,80p' docs/results/deepseek_q4_decode_baseline/summary.md
```

## Shot 7: Show The Chart

Open this file in a browser or editor:

```text
docs/assets/deepseek-q4-decode-headline.svg
```

Key line to say:

```text
Baseline decode was 0.98 tokens per second. With MADV_WILLNEED enabled, decode
rose to 1.23 tokens per second.
```

## Shot 8: Show Long-Run Dataset

```bash
sed -n '1,70p' docs/results/deepseek_q4_longrun_5h/README.md
```

Then show these charts:

```text
docs/results/deepseek_q4_longrun_5h/charts/generation_tps_by_case.svg
docs/results/deepseek_q4_longrun_5h/charts/prefill_wall_by_case.svg
docs/results/deepseek_q4_longrun_5h/charts/decode_wall_by_case.svg
```

Talking point:

```text
The long run completed 99 runs and helped separate prefill behavior from decode
behavior.
```

## Shot 9: Close On Reproduction Notes

```bash
sed -n '1,120p' docs/packed-experts-q4.md
sed -n '1,120p' docs/appendix-other-machines.md
```

Closing line:

```text
MoE-MADV is a small experiment, but it points to a useful idea: for very large
local MoE models, memory residency and page-in timing can be first-class
optimization targets.
```
