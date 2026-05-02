# YouTube Shorts Plan

Target format: vertical 9:16, 45-60 seconds.

Working title:

```text
150GB DeepSeek MoE on a 64GB M1 Max
```

## Layout

Use a split vertical composition:

- Top or left crop: terminal showing live token generation.
- Side strip: System Information showing `Apple M1 Max` and `64 GB`.
- Side strip or bottom crop: Activity Monitor Memory plus GPU History.
- Optional terminal sidecar: `scripts/run_moe_madv_resource_monitor.sh`
  showing clock time, elapsed time, memory, RSS, disk read, and page-in rate.

The most important visual proof is the combination of:

```text
64 GB machine specs + live token generation + elapsed time + memory graph
```

## Recording Commands

Open the macOS helper apps:

```bash
scripts/open_moe_madv_recording_apps.sh
```

Start the sidecar monitor in a small terminal:

```bash
OUT=logs/video_monitor_$(date +%Y%m%d_%H%M%S).csv \
  scripts/run_moe_madv_resource_monitor.sh
```

Start live generation in the main terminal:

```bash
scripts/run_moe_madv_live_generation_demo.sh
```

## Render And Voiceover

```bash
scripts/render_moe_madv_shorts.py

swift scripts/encode_frames_avfoundation.swift \
  docs/video/build/frames \
  docs/video/build/moe-madv-shorts-draft.mp4 \
  12 \
  720

scripts/generate_moe_madv_voiceover.py

swift scripts/mux_audio_avfoundation.swift \
  docs/video/build/moe-madv-shorts-draft.mp4 \
  docs/video/build/moe-madv-shorts-voiceover.mp3 \
  docs/video/build/moe-madv-shorts-final.mp4
```

Default narration voice: `River - Relaxed, Neutral, Informative` from
ElevenLabs. Override with `ELEVENLABS_VOICE_ID` in `.env` if needed.

## Edit Structure

### 0-3s: Hook

On-screen text:

```text
150GB MoE model
64GB M1 Max
```

Narration:

```text
This is a 150 gigabyte DeepSeek V4 Flash MoE model running on a 64 gigabyte M1 Max.
```

### 3-8s: Proof of machine

Show System Information and Activity Monitor.

On-screen text:

```text
Apple M1 Max / 64 GB unified memory
```

### 8-18s: Real generation starts

Show the command and the first generated tokens at normal speed.

On-screen text:

```text
Live local generation
```

### 18-42s: Fast-forward generation

Speed up the middle of the generation by 6x to 12x. Keep the monitor clock and
elapsed timer visible so the viewer understands that the long wait is real.

On-screen text:

```text
Fast-forwarded
Page-in and memory pressure are the bottleneck
```

### 42-53s: Result

Show the headline chart.

On-screen text:

```text
0.98 tok/s -> 1.23 tok/s
+25.4% decode throughput
```

### 53-60s: Close

Show GitHub repo.

On-screen text:

```text
github.com/daystar7777/MoE-MADV
Scripts, data, charts, and reproduction notes
```

## Editing Notes

- Do not fast-forward the first few generated tokens. Let viewers see real
  streaming first.
- Fast-forward the slow middle section only.
- Keep the final timing line at normal speed.
- Put `Fast-forwarded` on screen during sped-up sections.
- Do not crop out the memory/GPU charts during the generation section.
- If GPU History is mostly quiet, that is still useful: it supports the claim
  that the bottleneck is data delivery and page-in timing, not pure GPU compute.

## English Voiceover

```text
This is DeepSeek V4 Flash, a 284-billion-parameter MoE model.
The GGUF file is about 150 gigabytes.

The machine is a 64 gigabyte M1 Max.

I am showing the system specs, memory pressure, and GPU history next to the
terminal while the model generates tokens locally.

The slow part is not just compute. For this MoE model, the active experts change
as tokens are generated, so the runtime has to bring the right expert pages into
memory at the right time.

In MoE-MADV, after routing chooses the active experts, the runtime gives macOS a
MADV_WILLNEED hint for those expert ranges.

On the decode benchmark, that moved generation from 0.98 to 1.23 tokens per
second, a 25.4 percent throughput gain without changing the model.
```
