# Rebuilding `packed_experts_q4`

`packed_experts_q4` is the custom export produced from the MLX/safetensors Q4
source model. The binary artifact itself is not committed to this repository:
the local output is 43 files, about 3.2 GiB each, 137.06 GiB total.

GitHub blocks normal Git files over 100 MiB and Git LFS has plan-dependent
single-file limits. For that reason, this repo publishes the generator,
manifest, and exact rebuild process instead of committing the 137 GiB binary
export.

References:

- GitHub large-file limits:
  <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>
- Git LFS limits:
  <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage>

## Source Model

Download source:

- Repo:
  [`mlx-community/DeepSeek-V4-Flash-4bit`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit)
- Exact revision used here:
  `38c0bd20a6fba70f22c5ee2940ec0092b36ab936`
- Exact tree URL:
  `https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/tree/38c0bd20a6fba70f22c5ee2940ec0092b36ab936`
- Files:
  `model-00001-of-00033.safetensors` through
  `model-00033-of-00033.safetensors`, plus config/index/tokenizer metadata

The source model files stay under `models/`, which is ignored by Git.

## What The Export Does

The MLX Q4 expert weights store each routed projection as separate tensors:

- `*.weight`: packed 4-bit values stored as `U32`
- `*.scales`: MXFP4/E8M0 scale bytes stored as `U8`

The exporter reads each expert's `gate`, `up`, and `down` projection tensors and
writes a streaming-friendly per-layer file:

```text
models/deepseek-v4-flash-4bit/packed_experts_q4/
  layer_00.bin
  layer_01.bin
  ...
  layer_42.bin
```

Each packed expert is laid out as:

```text
gate: [scale byte][16 packed 4-bit bytes] repeated by row/group
up:   [scale byte][16 packed 4-bit bytes] repeated by row/group
down: [scale byte][16 packed 4-bit bytes] repeated by row/group
```

Current dimensions:

| item | value |
| --- | ---: |
| layers | 43 |
| routed experts per layer | 256 |
| active experts per token | 6 |
| bytes per expert | 13,369,344 |
| bytes per layer | 3,422,552,064 |
| total packed bytes | 147,169,738,752 |
| total packed size | 137.06 GiB |

## Rebuild Steps

From the repository root:

```bash
scripts/bootstrap_local_env.sh

# Download exact MLX/safetensors Q4 source files.
scripts/download_deepseek_q4_shards_sequential.sh

# Inspect the source tensor layout.
scripts/inspect_deepseek_q4_layout.py \
  --json-out docs/deepseek-q4-layout.json

# Optional dry run: confirms files, layers, and expected output size.
scripts/repack_deepseek_q4_experts.py --dry-run

# Build all 43 packed layer files.
scripts/repack_deepseek_q4_experts.py
```

The exporter verifies every layer after writing by comparing experts
`0`, `1`, `128`, and `255` against the source safetensors data.

To verify one layer later:

```bash
scripts/repack_deepseek_q4_experts.py --verify-only 0
scripts/repack_deepseek_q4_experts.py --verify-only 42
```

## Local Manifest

The included manifest records the local artifact shape without storing the
artifact:

- `docs/artifacts/packed_experts_q4_manifest.json`

It includes the source repo, exact revision, generator script, file count, per
file sizes, and total size.

## Related Code

- `scripts/inspect_deepseek_q4_layout.py`: reads safetensors headers and derives
  offsets, strides, projection shapes, and packed component sizes.
- `scripts/repack_deepseek_q4_experts.py`: creates `packed_experts_q4`.
- `scripts/probe_deepseek_q4_one_expert_cpu.py`: CPU reference reader for one
  packed expert.
- `scripts/route_deepseek_q4_probe.py`: routes a token through real DeepSeek V4
  hash/gate logic and checks the packed expert path.
- `metal_infer/deepseek_q4_probe.m` and
  `metal_infer/deepseek_q4_probe.metal`: Metal probe for the packed Q4 expert
  layout.

## Publishing The Binary Artifact Later

If the packed export should be distributed directly, publish it outside normal
Git history, for example as a Hugging Face dataset/model artifact. The repo
should still keep `models/` ignored so downloaded source models and generated
packed binaries are not accidentally committed.
