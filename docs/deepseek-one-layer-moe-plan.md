# DeepSeek One-Layer MoE Probe

This is the next porting checkpoint after the Q4 and GGUF expert repacks. The final target is HF/MLX Q4 on this Mac; the GGUF path is kept as a runnable comparison point.

## Goal

Validate one routed DeepSeek expert outside the full model:

```text
out = down(silu(gate(x)) * up(x))
```

where:

- `x` is a deterministic `float32[4096]` input vector.
- `gate` and `up` are one expert slice from `ffn_gate_exps.weight` and `ffn_up_exps.weight`.
- `down` is the matching expert slice from `ffn_down_exps.weight`.
- The packed source is `packed_experts_deepseek/layer_00.bin`.

Passing this checkpoint means the SSD layout and Metal dequantized matvecs agree with a CPU/ggml reference for one expert block. It does not require routing, attention, tokenizer, or end-to-end generation.

The CPU side of this checkpoint is available now for GGUF IQ2/Q2:

```bash
scripts/probe_deepseek_one_expert_cpu.py --layer 0 --expert 0
```

Use its `out sha256` and summary stats as the first reference for the upcoming Metal probe.

Current local reference for `--layer 0 --expert 0`:

```text
out min=-5.60802 max=4.97482 mean=-0.00243243 rms=1.42796
out sha256=94b8df8e5e861c8a04a54ae08e7629346fa3da8501b1684493c68e9b62d52e7e
```

The CPU side for the HF/MLX Q4 target is:

```bash
scripts/probe_deepseek_q4_one_expert_cpu.py --layer 0 --expert 0
```

Current local Q4 reference for `--layer 0 --expert 0`:

```text
out min=-4.24056 max=4.85627 mean=0.0322404 rms=1.2815
out sha256=1f1ad59f8bccc0915a50453f5a9ad3b6e0c110cc8e61a7fc23682d13ce41df17
```

The first Metal Q4 probe is available now:

```bash
scripts/run_deepseek_q4_probe.sh --layer 0 --expert 0
```

Current local Metal-vs-CPU result:

```text
cpu elapsed: 0.032s
gpu elapsed: 0.003s
compare: max_abs=9.77516174e-06 at 667 cpu=2.1510272 gpu=2.15103698
```

## Packed Expert Layout

Each DeepSeek packed expert block is 7,077,888 bytes:

| Component | GGUF type | Logical shape | Packed offset | Packed bytes |
| --- | --- | --- | ---: | ---: |
| gate | `IQ2_XXS` | `[4096, 2048]` | 0 | 2,162,688 |
| up | `IQ2_XXS` | `[4096, 2048]` | 2,162,688 | 2,162,688 |
| down | `Q2_K` | `[2048, 4096]` | 4,325,376 | 2,752,512 |

Layer size is 1,811,939,328 bytes for 256 experts.

## Kernel Source Map

The nearest known-good implementation is in the experimental DeepSeek llama.cpp checkout:

- CPU reference declarations:
  - `ggml/src/ggml-quants.h`: `dequantize_row_iq2_xxs`, `dequantize_row_q2_K`
- CPU reference implementations:
  - `ggml/src/ggml-quants.c`: `dequantize_row_q2_K`
  - `ggml/src/ggml-quants.c`: `dequantize_row_iq2_xxs`
- Metal matvec implementations:
  - `ggml/src/ggml-metal/ggml-metal.metal`: `kernel_mul_mv_q2_K_f32_impl`
  - `ggml/src/ggml-metal/ggml-metal.metal`: `kernel_mul_mv_iq2_xxs_f32_impl`
- Metal launch constants:
  - `ggml/src/ggml-metal/ggml-metal-impl.h`: `N_R0_Q2_K`, `N_SG_Q2_K`, `N_R0_IQ2_XXS`, `N_SG_IQ2_XXS`

The existing `flash-moe` Qwen kernels use affine 4-bit weights with separate scale and bias arrays. DeepSeek GGUF expert slices embed their quantization metadata inside GGUF quant blocks, so the Qwen kernels cannot be reused directly.

## Probe Shape

1. Read one packed expert block with `pread`:

```text
expert_offset = expert_id * 7077888
gate_ptr = expert_offset + 0
up_ptr = expert_offset + 2162688
down_ptr = expert_offset + 4325376
```

2. CPU reference:

```text
gate_out[2048] = matvec_iq2_xxs(gate, x[4096])
up_out[2048] = matvec_iq2_xxs(up, x[4096])
act[2048] = silu(gate_out) * up_out
out_ref[4096] = matvec_q2_K(down, act)
```

3. Metal probe:

```text
gate_gpu = metal_iq2_xxs_matvec(gate, x)
up_gpu = metal_iq2_xxs_matvec(up, x)
act_gpu = metal_silu_mul(gate_gpu, up_gpu)
out_gpu = metal_q2_K_matvec(down, act_gpu)
```

4. Compare:

```text
max_abs_error(out_gpu, out_ref)
max_rel_error(out_gpu, out_ref)
top mismatched indices
```

## Recommended Implementation

Add standalone probe targets first instead of modifying the full Qwen inference engine. Keep the write scope narrow:

- Load `layout.json` from the repacked expert directory.
- Load one expert from one layer.
- Use deterministic input values, for example `sin(i * 0.013)`.
- Compare against `scripts/probe_deepseek_one_expert_cpu.py`, which implements the `IQ2_XXS` and `Q2_K` CPU reference path.
- Compare Q4 against `scripts/probe_deepseek_q4_one_expert_cpu.py` and `metal_infer/deepseek_q4_probe.m`.
- Port the matching Metal kernels into a separate shader file or a clearly named DeepSeek section before wiring into full generation.

Once the one-expert probe passes, extend it to K=6 experts with routing weights. After that, wire it into the full engine and start replacing the architecture-level Qwen assumptions.

## Commands Around This Checkpoint

Inspect source GGUF:

```bash
scripts/inspect_deepseek_gguf.py --json-out docs/deepseek-gguf-layout.json
```

Repack routed experts:

```bash
scripts/repack_deepseek_gguf_experts.py
```

Run the immediate GGUF smoke demo while the custom engine is being ported:

```bash
PROMPT="Hello" TOKENS=16 scripts/run_deepseek_gguf_demo.sh
```

Show llama.cpp logs and timings:

```bash
LOGS=1 PROMPT="Hello" TOKENS=16 scripts/run_deepseek_gguf_demo.sh
```
