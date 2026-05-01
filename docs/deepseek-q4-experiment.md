# DeepSeek V4 Flash Q4 Experiment

This is the primary target path for running DeepSeek V4 Flash Q4 on this Mac.

## Result So Far

The HF/MLX Q4 repository is `mlx-community/DeepSeek-V4-Flash-4bit`.

- Repository size: about 141 GiB.
- Routed expert data after repacking: 137.06 GiB.
- Local layer-0 source shard: `models/deepseek-v4-flash-4bit/model-00001-of-00033.safetensors`.
- Local layer-0 packed output: `models/deepseek-v4-flash-4bit/packed_experts_q4/layer_00.bin`.
- Q4 layer-0 repack verification passed for experts `0, 1, 128, 255`.
- One-expert Q4 Metal probe matches CPU reference with max absolute error below `1e-5`.
- K=6 Q4 Metal probe with experts `0,1,2,3,4,5` matches CPU reference with max absolute error below `4e-6`.
- Layer-0 hash routing now feeds the Q4 Metal probe with model-selected experts and routing weights.

## Q4 Expert Layout

HF/MLX stores routed expert tensors as separate packed values and scale tensors:

- `gate_proj.weight`: `U32 [256, 2048, 512]`
- `gate_proj.scales`: `U8 [256, 2048, 128]`
- `up_proj.weight`: `U32 [256, 2048, 512]`
- `up_proj.scales`: `U8 [256, 2048, 128]`
- `down_proj.weight`: `U32 [256, 4096, 256]`
- `down_proj.scales`: `U8 [256, 4096, 64]`

The repacker writes each group as a 17-byte `mxfp4` block:

```text
[1 byte E8M0 scale][16 bytes packed 4-bit values]
```

Packed expert layout:

| Component | Shape | Offset | Bytes |
| --- | --- | ---: | ---: |
| gate | `[2048, 4096]` | 0 | 4,456,448 |
| up | `[2048, 4096]` | 4,456,448 | 4,456,448 |
| down | `[4096, 2048]` | 8,912,896 | 4,456,448 |

Each packed Q4 expert is 13,369,344 bytes. With K=6, expert I/O is about 76.5 MiB per layer and about 3.21 GiB per generated token across 43 layers before caching.

## Commands

Inspect layout:

```bash
scripts/inspect_deepseek_q4_layout.py --json-out docs/deepseek-q4-layout.json
```

Download the first shard for layer-0 testing:

```bash
scripts/download_model_assets.sh shard deepseek model-00001-of-00033.safetensors
```

Repack layer 0:

```bash
scripts/repack_deepseek_q4_experts.py --layers 0
```

Run CPU reference:

```bash
scripts/probe_deepseek_q4_one_expert_cpu.py --layer 0 --expert 0
```

Run Metal probe:

```bash
scripts/run_deepseek_q4_probe.sh --layer 0 --expert 0
```

Run K=6 Metal probe:

```bash
scripts/run_deepseek_q4_probe.sh --layer 0 --experts 0,1,2,3,4,5
```

Select actual layer-0 hash-routed experts and weights for a token id, then optionally run the Metal probe:

```bash
scripts/route_deepseek_q4_probe.py --layer 0 --token-id 0
scripts/route_deepseek_q4_probe.py --layer 0 --token-id 0 --run-probe
```

## Current Probe Output

```text
cpu out: min=-4.24056 max=4.85627 mean=0.0322404 rms=1.2815
gpu out: min=-4.24057 max=4.85627 mean=0.0322404 rms=1.2815
compare: max_abs=9.77516174e-06 at 667 cpu=2.1510272 gpu=2.15103698
```

K=6 uniform-weight probe:

```text
cpu elapsed: 0.191s
gpu elapsed: 0.011s
compare: max_abs=3.51667404e-06 at 1305 cpu=0.44933936 gpu=0.449335843
```

Layer-0 token-0 hash-routed probe:

```text
Selected experts: 254,222,245,200,53,35
Normalized scaled weights: 0.222215831,0.191317201,0.474580705,0.0883647129,0.468563139,0.0549584925 (sum=1.50000012)
cpu elapsed: 0.198s
gpu elapsed: 0.010s
compare: max_abs=6.91413879e-06 at 3729 cpu=-2.15348911 gpu=-2.15349603
```

## Next Step

Use a real layer input vector instead of the deterministic synthetic vector. After that, the Q4 expert path can be wired into the full inference engine separately from the existing Qwen path.
