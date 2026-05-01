# DeepSeek V4 Flash Porting Notes

These notes capture the first pass at adapting `flash-moe` from Qwen3.5-397B-A17B to `mlx-community/DeepSeek-V4-Flash-4bit`.

## Current State

- The Metal engine builds locally on Apple Silicon.
- MLX is available in `.venv` via `mlx-lm`.
- Metadata for both public Hugging Face repos can be downloaded with:

```bash
scripts/bootstrap_local_env.sh
scripts/download_model_assets.sh metadata qwen
scripts/download_model_assets.sh metadata deepseek
scripts/compare_model_meta.py
```
- Qwen reproduction has been validated locally on Apple M1 Max with 64 GB unified memory:
  - `model_weights.bin`: 5.52 GB
  - `packed_experts/`: 202.5 GB
  - `packed_experts` verification: all 60 layers passed spot checks
  - Short generation run: about 3.05 tok/s with K=4 4-bit experts

## Qwen Target Implemented by This Repo

The current inference engine is specialized for Qwen3.5 MoE:

- `model_type`: `qwen3_5_moe`
- `hidden_size`: 4096
- `num_hidden_layers`: 60
- `num_attention_heads`: 32
- `num_key_value_heads`: 2
- `head_dim`: 256
- `vocab_size`: 248320
- `num_experts`: 512
- `num_experts_per_tok`: 10, run-time K commonly set to 4
- `moe_intermediate_size`: 1024
- `shared_expert_intermediate_size`: 1024
- Attention mix: 45 `linear_attn` GatedDeltaNet layers and 15 `self_attn` layers.
- Expert tensor family: `language_model.model.layers.*.mlp.switch_mlp.*`
- Shared expert tensor family: `language_model.model.layers.*.mlp.shared_expert.*`

The corresponding constants are hard-coded near the top of `metal_infer/infer.m`.

## DeepSeek V4 Flash Metadata

`mlx-community/DeepSeek-V4-Flash-4bit` is structurally different:

- `model_type`: `deepseek_v4`
- `hidden_size`: 4096
- `num_hidden_layers`: 43
- `num_attention_heads`: 64
- `num_key_value_heads`: 1
- `head_dim`: 512
- `vocab_size`: 129280
- `n_routed_experts`: 256
- `num_experts_per_tok`: 6
- `moe_intermediate_size`: 2048
- Attention tensor family: `model.layers.*.attn.*`
- Extra attention families: `attn_hc`, `attn.compressor`, and layer-42 `attn.indexer`
- Expert tensor family: `model.layers.*.ffn.switch_mlp.*`
- Shared expert tensor family: `model.layers.*.ffn.shared_experts.*`
- Expert quantization differs from Qwen: many `switch_mlp` tensors use `mxfp4` with group size 32.

## Porting Implications

This is not a drop-in model swap. The minimum port requires:

1. A DeepSeek V4 weight manifest extractor.
2. A DeepSeek expert packer for routed experts using 256 experts and 2048 intermediate size. For the HF/MLX Q4 path this is implemented by `scripts/repack_deepseek_q4_experts.py`.
3. New Metal dequant kernels for DeepSeek HF/MLX Q4 expert quantization: `mxfp4` blocks with one E8M0 scale byte per 32 values.
4. A DeepSeek attention implementation. Qwen's GatedDeltaNet and full-attention split does not map to DeepSeek V4's MLA/compressor/indexer layout.
5. A DeepSeek tokenizer/chat-template path. The token IDs and chat template differ from Qwen.
6. Runtime constants or a generated config header so `infer.m` is no longer bound to only Qwen.

## Practical Path

1. Reproduce the current Qwen path first and record local tokens/sec.
2. Try a DeepSeek V4 Flash GGUF/llama.cpp path separately. Q4 is acceptable; MLX is not a requirement.
3. Add a small model-config generation step for constants and tensor family names.
4. Implement only DeepSeek metadata extraction and expert packing first.
5. Add DeepSeek attention after expert streaming is validated with isolated tensor tests.
6. Only then attempt end-to-end generation.

The Qwen reproduction remains the fastest sanity test for SSD streaming, Metal kernels, tokenizer export, and the repo's benchmark claims on this Mac.

## DeepSeek Runtime Options

MLX is currently not the easiest route. `mlx-lm 0.31.3` does not ship a `deepseek_v4` model implementation, so `mlx-community/DeepSeek-V4-Flash-4bit` is not enough by itself.

GGUF is worth testing, but it is not automatically easier. Current DeepSeek V4 Flash GGUF builds need a llama.cpp variant that understands the DeepSeek V4 architecture and its FP4/FP8 tensor encodings. On a 64 GB Mac, a normal in-memory Q4 run may still fail; the reason this repo works for Qwen is that it streams expert weights from SSD instead of loading the whole model.

Recommended order:

1. Use this repo's Qwen path for the first video/demo because it is already validated locally.
2. Test DeepSeek V4 Flash GGUF in a separate experimental checkout.
3. If GGUF cannot fit or lacks stable macOS support, port DeepSeek to this SSD-streaming Metal design.

The Q4 experiment is tracked in `docs/deepseek-q4-experiment.md`. The Q4 layout snapshot is tracked in `docs/deepseek-q4-layout.json`.
The GGUF experiment is tracked in `docs/deepseek-gguf-experiment.md`. The routed expert layout snapshot is tracked in `docs/deepseek-gguf-layout.json`.
The one-layer custom-engine checkpoint is tracked in `docs/deepseek-one-layer-moe-plan.md`.

## Porting Milestone 1: Expert Streaming

Completed:

- `scripts/inspect_deepseek_gguf.py` extracts GGUF metadata and tensor offsets.
- `scripts/repack_deepseek_gguf_experts.py` repacks routed experts into per-layer files.
- Local output has 43 files under `packed_experts_deepseek/`, 72.6 GiB total.
- Each packed expert is 7,077,888 bytes, same outer size as the existing Qwen expert block.
- `scripts/inspect_deepseek_q4_layout.py` extracts the HF/MLX Q4 routed expert layout from safetensors headers.
- `scripts/repack_deepseek_q4_experts.py` repacks HF/MLX Q4 experts into `mxfp4` blocks.
- Q4 layer 0 has been packed locally from `model-00001-of-00033.safetensors` and spot verification passed.
- Each Q4 packed expert is 13,369,344 bytes. A full 43-layer routed expert pack is 137.06 GiB.
- `metal_infer/deepseek_q4_probe.m` validates one Q4 expert forward on Metal against a CPU reference.

Next validation target:

1. Extend the Q4 Metal probe from one expert to K=6 experts.
2. Add routed expert weighting and accumulation.
3. Wire Q4 expert streaming into the full engine behind a separate DeepSeek path.
4. Port DeepSeek attention, tokenizer, and routing details after the expert path remains stable.
5. Only then attempt end-to-end Q4 generation.

Q4 one-expert Metal probe:

```bash
scripts/run_deepseek_q4_probe.sh --layer 0 --expert 0
```

Immediate DeepSeek GGUF smoke demo:

```bash
PROMPT="Hello" TOKENS=16 scripts/run_deepseek_gguf_demo.sh
```

## Local Qwen Demo

After running `scripts/prepare_qwen_flash_moe.sh`, use:

```bash
TOKENS=40 PROMPT="Explain quantum computing in simple terms." scripts/run_qwen_demo.sh --timing
```

The first run compiles Metal shaders and warms caches; later runs usually have lower TTFT.
