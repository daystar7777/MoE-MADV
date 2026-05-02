# Model Sources and Parser Artifacts

This page pins down exactly which Hugging Face models were downloaded, which one
was used for the headline benchmark, and which parser artifacts are included in
this repo.

## Name Check

- Project name: `MoE-MADV`
- Target model name: `DeepSeek V4 Flash`
- Notation used in this repo: `DeepSeek V4 Flash Q4` or `DeepSeek V4 Flash MXFP4`
- No `DeepSeek R4` model is used here.
- The headline claim is about running a **284B total-parameter MoE model** on a
  64GB M1 Max. Only a small routed subset is active per token.

## Final Benchmark Model

The benchmark model is:

- Hugging Face repo:
  [`lovedheart/DeepSeek-V4-Flash-GGUF`](https://huggingface.co/lovedheart/DeepSeek-V4-Flash-GGUF)
- File: `DeepSeek-V4-Flash-MXFP4_MOE.gguf`
- Exact file URL:
  `https://huggingface.co/lovedheart/DeepSeek-V4-Flash-GGUF/blob/cd42deba41ac0536e68b125dfc367197b0ec3038/DeepSeek-V4-Flash-MXFP4_MOE.gguf`
- Base model:
  [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- Local path:
  `/Users/storysq/Documents/New project/deepseek-v4-experiments/models/lovedheart-deepseek-v4-flash-gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf`
- Hugging Face revision recorded locally: `cd42deba41ac0536e68b125dfc367197b0ec3038`
- Local file size: `150,225,324,672` bytes, or `139.91 GiB`
- Hugging Face display size: `150 GB`
- Hugging Face model-card labels: `284B params`, `deepseek4`, `4-bit`,
  `MXFP4_MOE`

This is the model used for the decode baseline and 5-hour benchmark results.

Download command:

```bash
scripts/download_deepseek_q4_gguf.sh
```

## Supporting Downloaded Model

The MLX/safetensors source was also downloaded and parsed:

- Hugging Face repo:
  [`mlx-community/DeepSeek-V4-Flash-4bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit)
- Exact revision URL:
  `https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/tree/38c0bd20a6fba70f22c5ee2940ec0092b36ab936`
- Local path: `models/deepseek-v4-flash-4bit`
- Hugging Face revision recorded locally: `38c0bd20a6fba70f22c5ee2940ec0092b36ab936`
- Downloaded shards: `model-00001-of-00033.safetensors` through
  `model-00033-of-00033.safetensors`
- Local source shard bytes: `151,482,760,008`
- Local config highlights: `model_type=deepseek_v4`, 43 layers, 256 routed
  experts, 6 experts per token, 3 hash-routing layers

This path was used to understand the raw MLX Q4/MXFP4 layout and to build the
custom routed-expert pack/probe work. It is not the model used by the final
headline GGUF benchmark.

Download command:

```bash
scripts/download_deepseek_q4_shards_sequential.sh
```

The exact packed export process is documented in
[packed-experts-q4.md](packed-experts-q4.md). The generated
`packed_experts_q4` binaries are not committed to Git because the local artifact
is 137.06 GiB.

## Other Model Paths Tried

| source | status | why it was not the headline model |
| --- | --- | --- |
| [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) / [`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`](https://huggingface.co/antirez/deepseek-v4-gguf/blob/3af08b96a788790ef6f1d113e5257794622884b8/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf) | Downloaded and parsed; local file is `86,720,111,200` bytes (`80.76 GiB`). | Useful early GGUF smoke/parser target, but it is not the final Q4/MXFP4 MoE file. |
| [`tecaprovn/deepseek-v4-flash-gguf`](https://huggingface.co/tecaprovn/deepseek-v4-flash-gguf) / [`DeepSeekV4-Flash-158B-Q4_K_M.gguf`](https://huggingface.co/tecaprovn/deepseek-v4-flash-gguf/blob/main/DeepSeekV4-Flash-158B-Q4_K_M.gguf) | Partial download was started, then canceled and deleted. | The user asked to stop changing models and optimize the current target instead. |
| [`mlx-community/Qwen3.5-397B-A17B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit) | Original `flash-moe` reference path; obsolete local generated Qwen weights were deleted. | Not DeepSeek V4 Flash. It remains useful only as original project context. |

All Hugging Face model files are intentionally excluded from this GitHub repo.
The `.gitignore` keeps `models/`, `model_meta/`, and `logs/` out of normal
commits.

## Parser Scripts

- `scripts/inspect_deepseek_gguf.py`
  - Parses GGUF metadata, tensor families, routed expert tensor offsets, per
    expert strides, and byte ranges.
  - Default target is now the final lovedheart
    `DeepSeek-V4-Flash-MXFP4_MOE.gguf` file.
- `scripts/inspect_deepseek_q4_layout.py`
  - Parses the MLX/safetensors Q4 layout from
    `mlx-community/DeepSeek-V4-Flash-4bit`.
  - Extracts MXFP4 weight/scales offsets and the packed expert layout used by
    the custom Metal probe.
- `scripts/analyze_deepseek_q4_cache_patterns.py`
  - Uses parsed model metadata plus prompt probes to estimate hot expert sets.
- `scripts/warm_deepseek_q4_expert_cache.py`
  - Uses parsed GGUF tensor byte ranges to prewarm selected expert pages with
    `pread`.

## Parser Outputs Included

- Full final GGUF layout:
  `docs/model-parsing/deepseek-v4-flash-mxfp4-moe-gguf-layout.json`
- Final GGUF layout summary:
  `docs/model-parsing/deepseek-v4-flash-mxfp4-moe-gguf-layout-summary.json`
- MLX Q4 layout summary:
  `docs/model-parsing/deepseek-v4-flash-mlx-q4-layout-summary.json`
- Early antirez GGUF layout summary:
  `docs/model-parsing/deepseek-v4-flash-antirez-iq2xxs-gguf-layout-summary.json`
- Historical full layout snapshots:
  `docs/deepseek-q4-layout.json` and `docs/deepseek-gguf-layout.json`
- Packed export manifest:
  `docs/artifacts/packed_experts_q4_manifest.json`

One parser nuance: the final lovedheart file is published as `MXFP4_MOE`, but
the local GGUF reader reports the routed expert tensor enum as `Q3_K`. The repo
keeps both facts visible: the model source/name comes from Hugging Face, and the
tensor enum comes from the local GGUF parser.
