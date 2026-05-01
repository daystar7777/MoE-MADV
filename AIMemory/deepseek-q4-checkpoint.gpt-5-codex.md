# DeepSeek V4 Flash Q4 Checkpoint

## User goal

Run DeepSeek V4 Flash Q4 on this Mac. MLX is acceptable if it works, but Q4 local execution is the real goal. A separate engine port is acceptable. If it works well, the user wants a video and a GitHub publish flow.

## Repository state

- Root: `/Users/storysq/Documents/New project/flash-moe`
- Branch: `codex/flash-moe-deepseek-prep`
- Worktree was clean before AIMemory install.
- Model files under `models/` are ignored and should not be committed.

Commits already made:

- `e5e345c Add DeepSeek V4 Flash local runtime prep`
- `fc87096 Add DeepSeek Q4 expert packing probe`
- `d749ab6 Extend DeepSeek Q4 probe to multiple experts`
- `488c907 Add routed DeepSeek Q4 probe`

## Environment

- macOS: Darwin 25.4.0 arm64 on Mac Studio.
- Python venv: `.venv`
- `mlx.core` observed earlier: `0.31.2`
- `mlx-lm`: `0.31.3`
- `mlx-lm 0.31.3` is latest on pip and does not currently contain `mlx_lm.models.deepseek_v4`.

## Q4 HF/MLX model

- Repo: `mlx-community/DeepSeek-V4-Flash-4bit`
- Repo revision observed earlier: `38c0bd20a6fba70f22c5ee2940ec0092b36ab936`
- Total repository size: about `141 GiB`
- Config highlights:
  - `model_type`: `deepseek_v4`
  - `hidden_size`: `4096`
  - `num_hidden_layers`: `43`
  - `n_routed_experts`: `256`
  - `num_experts_per_tok`: `6`
  - `num_hash_layers`: `3`
- `scoring_func`: `sqrtsoftplus`
- `routed_scaling_factor`: `1.5`
- Local download is complete: 33/33 safetensors, no missing shards.
- Verified total safetensor shard bytes: `151,482,760,008`.

## Q4 expert layout

HF/MLX stores routed experts as separate MXFP4 values and scales. The local repacker writes each group as:

```text
[1 byte E8M0 scale][16 bytes packed 4-bit values]
```

Per packed expert:

- gate `[2048, 4096]`: `4,456,448` bytes
- up `[2048, 4096]`: `4,456,448` bytes
- down `[4096, 2048]`: `4,456,448` bytes
- expert total: `13,369,344` bytes
- layer total: `3,422,552,064` bytes
- 43-layer routed expert pack: `147,169,738,752` bytes, about `137.06 GiB`

## Files added for Q4 work

- `scripts/inspect_deepseek_q4_layout.py`
- `scripts/repack_deepseek_q4_experts.py`
- `scripts/probe_deepseek_q4_one_expert_cpu.py`
- `scripts/route_deepseek_q4_probe.py`
- `scripts/run_deepseek_q4_probe.sh`
- `metal_infer/deepseek_q4_probe.m`
- `metal_infer/deepseek_q4_probe.metal`
- `docs/deepseek-q4-experiment.md`
- `docs/deepseek-q4-layout.json`
- `docs/deepseek-v4-flash-porting.md`
- `docs/deepseek-one-layer-moe-plan.md`

## Verified Q4 results

Layer 0 repack:

- Source shard: `models/deepseek-v4-flash-4bit/model-00001-of-00033.safetensors`
- Output: `models/deepseek-v4-flash-4bit/packed_experts_q4/layer_00.bin`
- Verification passed for experts `0, 1, 128, 255`.

Full repack:

- Output dir: `models/deepseek-v4-flash-4bit/packed_experts_q4`
- Layer files: 43/43
- Packed size: about `137 GiB`
- Repack throughput: about `0.89 GiB/s` average
- Every layer `0..42` verified experts `0, 1, 128, 255`.

One expert CPU probe:

```text
out min=-4.24056 max=4.85627 mean=0.0322404 rms=1.2815
out sha256=1f1ad59f8bccc0915a50453f5a9ad3b6e0c110cc8e61a7fc23682d13ce41df17
```

One expert Metal probe:

```text
cpu elapsed: 0.032s
gpu elapsed: 0.003s
compare: max_abs=9.77516174e-06 at 667 cpu=2.1510272 gpu=2.15103698
```

K=6 uniform Metal probe:

```text
cpu elapsed: 0.191s
gpu elapsed: 0.011s
compare: max_abs=3.51667404e-06 at 1305 cpu=0.44933936 gpu=0.449335843
```

Layer-0 token-0 hash-routed Metal probe:

```text
Selected experts: 254,222,245,200,53,35
Raw sqrtsoftplus weights: 0.698836267,0.601664603,1.49248683,0.27789408,1.47356248,0.172836408
Normalized scaled weights: 0.222215831,0.191317201,0.474580705,0.0883647129,0.468563139,0.0549584925
Weight sum: 1.50000012
cpu elapsed: 0.190s
gpu elapsed: 0.010s
compare: max_abs=6.91413879e-06 at 3729 cpu=-2.15348911 gpu=-2.15349603
```

Layer-42 uniform K=6 Metal probe after full repack:

```text
Active experts: 0@0.166667 1@0.166667 2@0.166667 3@0.166667 4@0.166667 5@0.166667
cpu elapsed: 0.194s
gpu elapsed: 0.010s
compare: max_abs=5.1856041e-06 at 1491 cpu=-0.70002228 gpu=-0.700017095
```

## Commands that passed

```bash
python3 -m py_compile scripts/route_deepseek_q4_probe.py scripts/inspect_deepseek_q4_layout.py scripts/repack_deepseek_q4_experts.py scripts/probe_deepseek_q4_one_expert_cpu.py
bash -n scripts/run_deepseek_q4_probe.sh scripts/download_model_assets.sh
git diff --check
scripts/repack_deepseek_q4_experts.py --dry-run --layers 0
scripts/repack_deepseek_q4_experts.py --verify-only 0
scripts/run_deepseek_q4_probe.sh --layer 0 --expert 0 --weights 1
scripts/run_deepseek_q4_probe.sh --layer 0 --experts 0,1,2,3,4,5
scripts/route_deepseek_q4_probe.py --layer 0 --token-id 0 --run-probe
```

## GGUF side path

Existing experiment dir:

- `/Users/storysq/Documents/New project/deepseek-v4-experiments`

Already downloaded earlier:

- `antirez/deepseek-v4-gguf` GGUF, about `81 GiB`
- Built `antirez/llama.cpp-deepseek-v4-flash`
- Smoke-tested with `-ngl 0 -c 4096`
- `-c 512` assertion fails; `-c 4096` works
- Repacked GGUF routed experts to `packed_experts_deepseek`, about `72.6 GiB`

Other searched GGUF Q4 candidates:

- `tecaprovn/deepseek-v4-flash-gguf`: `DeepSeekV4-Flash-158B-Q4_K_M.gguf`, about `111.43 GiB`
- `lovedheart/DeepSeek-V4-Flash-GGUF`: native `MXFP4_MOE.gguf`, about `139.91 GiB`
- `nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF`: native FP4/FP8 GGUF, about `145.42 GiB`

Q4 GGUF helper scripts were added after AIMemory install:

```bash
scripts/download_deepseek_q4_gguf.sh
PROMPT="Hello" TOKENS=16 CONTEXT=4096 GPU_LAYERS=0 scripts/run_deepseek_q4_gguf_demo.sh
```

These default to `lovedheart/DeepSeek-V4-Flash-GGUF` and `DeepSeek-V4-Flash-MXFP4_MOE.gguf`, using the existing `antirez/llama.cpp-deepseek-v4-flash` runtime. This has not yet been downloaded or smoke-tested locally.

## Official V4 architecture reference

The official HF repo `deepseek-ai/DeepSeek-V4-Flash` contains `inference/model.py`.

Important porting facts from that reference:

- V4 has 43 layers and `hc_mult=4` Hyper-Connections, not the simpler residual path in Qwen.
- Attention uses low-rank Q, sparse/sliding attention, optional compression, `attn_sink`, `wq_a`, `wq_b`, `wkv`, `kv_norm`, `wo_a`, and `wo_b`.
- Early layers use hash routing through `tid2eid`; later layers use score-based routing with bias.
- Gate scoring is `sqrtsoftplus`, normalized and scaled by `route_scale=1.5`.
- Routed experts use SwiGLU clamp limit `10.0`, then apply expert routing weights before down projection.
- Shared expert is always present.
- Existing `mlx-lm` main branch still has no `deepseek_v4` module, so MLX end-to-end requires a local model module port.

## Active background job

There is no active HF download job now. The full Q4 source and packed routed expert files are local.

The robust download command that worked was:

```bash
scripts/download_deepseek_q4_shards_sequential.sh
```

The initial multi-file `scripts/download_model_assets.sh full deepseek` stalled twice. Prefer the sequential downloader for future resumes.

## Recommended continuation

1. Decide end-to-end path:
   - fastest likely local text path: GGUF Q4 with a DeepSeek V4-capable llama.cpp fork;
   - cleaner MLX path: add or vendor a `deepseek_v4` model module based on `deepseek_v32` and DeepSeek V4 config;
   - custom flash-moe path: keep porting from routed expert probe into a new DeepSeek engine path, not the Qwen-specific one.
2. For the custom engine, next milestone is replacing the deterministic probe vector with a real hidden state path.
3. For MLX, start from official `deepseek-ai/DeepSeek-V4-Flash/inference/model.py`, not from Qwen assumptions.

## Watchouts

- The existing `metal_infer/infer.m` is Qwen-specific. Avoid large in-place edits until a DeepSeek-specific path is designed.
- DeepSeek V4 uses hash routing tables for early layers: `ffn.gate.tid2eid`.
- Later layers use gate weights and correction bias, not the same hash route table.
- The deterministic probe input is not a real hidden state yet. The next custom-engine milestone is feeding a real hidden vector into the routed Q4 expert path.

---
Created: 2026-05-02 06:20 JST by gpt-5-codex
