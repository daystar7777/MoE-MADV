# DeepSeek V4 Flash GGUF Experiment

This is the non-MLX path for testing DeepSeek V4 Flash locally.

## Result

`antirez/llama.cpp-deepseek-v4-flash` builds on this Mac with Metal enabled, and the `antirez/deepseek-v4-gguf` model loads on Apple M1 Max with 64 GB unified memory.

Observed smoke-test command:

```bash
cd "../deepseek-v4-experiments/llama.cpp-deepseek-v4-flash"
./build/bin/llama-cli \
  -m ../models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  -p "Hello" \
  -n 8 \
  -c 4096 \
  -ngl 0 \
  --no-warmup \
  -st \
  --simple-io
```

Smoke-test output included:

- prompt speed around 1.3 tok/s
- generation speed around 1.7 tok/s on the single-turn run
- earlier 2-token interactive run reported generation around 3.1 tok/s

Convenience wrapper:

```bash
PROMPT="Hello" TOKENS=16 scripts/run_deepseek_gguf_demo.sh
```

Set `LOGS=1` to show llama.cpp logs.
Set `STDERR=0` to hide Metal initialization diagnostics during screen recording.

## Notes

- `-c 512` failed with an assertion in `src/models/deepseek4.cpp`; `-c 4096` worked.
- The GGUF is about 81 GiB on disk.
- Runtime memory pressure is high but manageable for a short smoke test.
- The test used `-ngl 0`, so this was CPU-oriented despite Metal being initialized. GPU offload can be explored next, but the M1 Max recommended Metal working set is about 55.7 GB, so full offload is unlikely.
- The CLI output currently mixes logs with generated text. For clean video capture, use a wrapper that filters logs or run the server mode and capture responses through the API.

## GGUF Expert Layout

The GGUF routed expert tensors are already grouped per layer:

- `blk.N.ffn_gate_exps.weight`: `IQ2_XXS`, shape `[4096, 2048, 256]`
- `blk.N.ffn_up_exps.weight`: `IQ2_XXS`, shape `[4096, 2048, 256]`
- `blk.N.ffn_down_exps.weight`: `Q2_K`, shape `[2048, 4096, 256]`

For each layer:

- experts: 256
- gate expert slice: 2,162,688 bytes
- up expert slice: 2,162,688 bytes
- down expert slice: 2,752,512 bytes
- total per expert: 7,077,888 bytes
- total per layer: 1,811,939,328 bytes

This exactly matches the Qwen `flash-moe` expert block size, but the internal quantization format is different. The existing affine 4-bit Metal kernels cannot be reused as-is.

Generated layout:

```bash
scripts/inspect_deepseek_gguf.py --json-out docs/deepseek-gguf-layout.json
```

The current local layout snapshot is in `docs/deepseek-gguf-layout.json`.

## Repacked Experts

The routed experts were repacked into contiguous SSD-streaming blocks:

```bash
scripts/repack_deepseek_gguf_experts.py
```

Output:

```text
../deepseek-v4-experiments/models/antirez-deepseek-v4-gguf/packed_experts_deepseek/
```

Result:

- 43 layer files
- 72.6 GiB total
- layer spot checks passed for experts `0, 1, 128, 255`
- packed expert layout per expert: `[gate IQ2_XXS][up IQ2_XXS][down Q2_K]`

## Setup

```bash
mkdir -p ../deepseek-v4-experiments
cd ../deepseek-v4-experiments
git clone https://github.com/antirez/llama.cpp-deepseek-v4-flash.git
cd llama.cpp-deepseek-v4-flash
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8 --target llama-cli
```

Download:

```bash
scripts/download_deepseek_gguf.sh
```
