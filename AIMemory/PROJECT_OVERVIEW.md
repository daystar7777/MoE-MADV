# Project Overview

> Read this after `AIMemory/INDEX.md` and before acting on a new task.

## What is this project?

This is a fork/worktree of `danveloper/flash-moe` focused on running DeepSeek V4 Flash Q4 locally on this Mac.

The user goal is not merely a demo. The target is to get **DeepSeek V4 Flash Q4** running on this machine, then make a video and publish the modified project to GitHub if the result is good.

## Current branch and repository

- Project root: `/Users/storysq/Documents/New project/flash-moe`
- Branch: `codex/flash-moe-deepseek-prep`
- Upstream source: `https://github.com/danveloper/flash-moe`
- AIMemory source installed from: `https://github.com/daystar7777/agent-work-mem`, commit `57c81e42ad0779fb16e960628189999b5dd9e241`

Recent local commits:

- `488c907 Add routed DeepSeek Q4 probe`
- `d749ab6 Extend DeepSeek Q4 probe to multiple experts`
- `fc87096 Add DeepSeek Q4 expert packing probe`
- `e5e345c Add DeepSeek V4 Flash local runtime prep`

## Current technical state

The custom Q4 routed-expert path is working for all 43 layers:

- HF/MLX repo: `mlx-community/DeepSeek-V4-Flash-4bit`
- Local model dir: `/Users/storysq/Documents/New project/flash-moe/models/deepseek-v4-flash-4bit`
- Local packed Q4 expert dir: `/Users/storysq/Documents/New project/flash-moe/models/deepseek-v4-flash-4bit/packed_experts_q4`
- Full HF Q4 download is complete: 33/33 safetensors, no missing shards.
- Full Q4 routed expert repack is complete: 43/43 layer files, about `137 GiB`.
- One-expert Q4 Metal probe matches CPU.
- K=6 Q4 Metal probe matches CPU.
- Hash-routed K=6 Q4 Metal probe uses actual `tid2eid` and BF16 router weights and matches CPU.

Layer 0 token id 0 route:

- Experts: `254,222,245,200,53,35`
- Weights: `0.222215831,0.191317201,0.474580705,0.0883647129,0.468563139,0.0549584925`
- Metal-vs-CPU max abs error: `6.91413879e-06`

## Important constraints and observations

- Installed `mlx-lm` version is `0.31.3`, currently latest from pip at this time.
- `mlx-lm 0.31.3` does not include a `deepseek_v4` model module. It has nearby DeepSeek V3/V3.2 modules, but V4 needs additional work.
- Existing `metal_infer/infer.m` is deeply Qwen-specific: 60 layers, Qwen tensor names, Qwen attention assumptions, and affine 4-bit expert layout.
- Therefore the most practical path is:
  1. keep the Q4 Metal routed-expert path as the custom-engine foundation;
  2. use MLX or GGUF as a short-term end-to-end text-generation path if feasible;
  3. port DeepSeek V4 full model architecture into a separate path rather than bending the Qwen path in place.

## Active local work

A Hugging Face full download for `mlx-community/DeepSeek-V4-Flash-4bit` completed via:

```bash
scripts/download_deepseek_q4_shards_sequential.sh
```

The initial multi-file download stalled twice; the sequential downloader proved stable and should be reused for resume.

The immediate active work is no longer downloading or packing experts. The next work is an end-to-end Q4 smoke path.

## Useful commands

Run actual layer-0 routed Q4 Metal probe:

```bash
scripts/route_deepseek_q4_probe.py --layer 0 --token-id 0 --run-probe
```

Verify layer-0 Q4 expert pack:

```bash
scripts/repack_deepseek_q4_experts.py --verify-only 0
```

Inspect Q4 layout:

```bash
scripts/inspect_deepseek_q4_layout.py --json-out docs/deepseek-q4-layout.json
```

Repack all Q4 routed experts:

```bash
scripts/repack_deepseek_q4_experts.py
```

## Next step

Decide between:

- adding a local `deepseek_v4` MLX module for end-to-end Q4 generation;
- using a Q4 GGUF route as a temporary end-to-end smoke test while continuing the custom Metal port;
- porting official DeepSeek V4 architecture pieces into a separate custom engine path.

---
Last rebuild: 2026-05-02 07:05 JST by gpt-5-codex
