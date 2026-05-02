# English Video Package

This folder contains production notes for an English demo video about MoE-MADV.
The video should show the real execution flow without trying to re-run the
entire 5-hour benchmark on camera.

Recommended title:

```text
Running a 150GB DeepSeek V4 Flash MoE Model on a 64GB M1 Max
```

Recommended subtitle:

```text
MoE-MADV: routing-aware page-in hints for local MoE inference
```

## Files

- [moe-madv-video-script.en.md](moe-madv-video-script.en.md): English
  narration and shot plan.
- [shorts-plan.en.md](shorts-plan.en.md): 45-60 second vertical Shorts plan
  with fast-forward notes.
- [execution-demo-runbook.en.md](execution-demo-runbook.en.md): exact terminal
  commands to record.
- [moe-madv-captions.en.srt](moe-madv-captions.en.srt): draft English
  subtitles.
- [moe-madv-shorts-captions.en.srt](moe-madv-shorts-captions.en.srt): draft
  60-second Shorts captions.
- [../../scripts/run_moe_madv_live_generation_demo.sh](../../scripts/run_moe_madv_live_generation_demo.sh):
  short live token-generation demo for the video.
- [../../scripts/run_moe_madv_resource_monitor.sh](../../scripts/run_moe_madv_resource_monitor.sh):
  terminal sidecar monitor with clock time, elapsed time, memory, RSS, disk
  read, and page-in rate.
- [../../scripts/open_moe_madv_recording_apps.sh](../../scripts/open_moe_madv_recording_apps.sh):
  opens System Information and Activity Monitor for recording.
- [../../scripts/run_moe_madv_video_tour.sh](../../scripts/run_moe_madv_video_tour.sh):
  optional terminal tour helper for screen recording.

## Suggested Video Length

Aim for 3.5 to 5 minutes.

The strongest story order is:

1. Show the size mismatch: 150GB model, 64GB machine.
2. Explain why MoE changes the bottleneck.
3. Show the actual setup and smoke-test commands.
4. Show the optimized vs baseline decode result.
5. Close with the GitHub repo and reproduction notes.

## Recording Notes

Use a terminal capture at 1920x1080 or 2560x1440. Keep the font large enough for
YouTube playback, around 18-22 pt. Record the chart SVG and README in a browser
or editor after the terminal demo.

The 5-hour benchmark does not need to be re-run live. Show the command and the
published result table instead.

Do record real token generation. It should be one of the main proof shots:

```bash
scripts/run_moe_madv_live_generation_demo.sh
```

This runs two short prompts through the local DeepSeek V4 Flash GGUF path and
lets the viewer see text streaming from the model.

For Shorts, keep a visible clock or elapsed timer on screen and fast-forward the
slow middle part of generation. The first generated tokens and the final timing
line should remain at normal speed.

Recommended sidecar layout:

```bash
# Opens System Information and Activity Monitor.
scripts/open_moe_madv_recording_apps.sh

# Run this in a narrow terminal next to the generation terminal.
OUT=logs/video_monitor_$(date +%Y%m%d_%H%M%S).csv \
  scripts/run_moe_madv_resource_monitor.sh
```

For a guided terminal tour:

```bash
# Safe mode: shows repo, docs, commands, and charts without running inference.
scripts/run_moe_madv_video_tour.sh

# Real smoke-test mode: runs the short JSON/plain-text generations too.
RUN_MODEL=1 scripts/run_moe_madv_video_tour.sh
```

## What Not To Record

Do not show private paths, tokens, shell history, Hugging Face credentials, or
any local model license acceptance screens. The model files are not committed to
this repo and should be treated as local artifacts.
