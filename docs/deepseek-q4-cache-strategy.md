# DeepSeek V4 Flash Q4 Cache Strategy

## What flash-moe does

The current flash-moe default is not a large custom expert cache. Earlier Metal
LRU, malloc LRU, and compressed expert cache experiments were slower, so the
runtime now keeps application memory small and lets the OS page cache do the
expert caching.

Important details from flash-moe:

- Non-expert weights are mmap'd.
- Active experts are read on demand from packed per-layer expert files.
- Only the routed experts for the current token are touched.
- `--cache-entries` defaults to `0`, which means "trust OS page cache".
- The process stays small so most RAM can become reclaimable file-backed cache.

That is why flash-moe can use most available memory without allocating most of
it itself. macOS owns that memory as page cache and can evict it under pressure.

## What the current DeepSeek Q4 GGUF path does

The lovedheart `DeepSeek-V4-Flash-MXFP4_MOE.gguf` path runs through the patched
DeepSeek V4 llama.cpp fork. The model is one large 140 GiB GGUF file. llama.cpp
maps it with `mmap`, and the CPU kernels touch expert tensor pages as routing
uses them. Those pages are faulted from NVMe and then remain in the macOS file
cache while memory pressure allows.

So yes: for this path, expert data is currently demand-loaded from NVMe through
`mmap` page faults, not copied through a flash-moe-style explicit `pread` expert
cache.

## Current optimization decision

For this machine, the most practical strategy is:

1. Keep the process heap and private buffers small.
2. Avoid CPU repacking MXFP4 expert tensors into a second in-memory copy.
3. Avoid bulk mmap prefetch/read-ahead that can evict useful expert pages.
4. Leave roughly 12-13 GiB of real memory headroom for the OS and other apps.
5. Let the rest naturally become file-backed GGUF page cache.

This mirrors flash-moe's winning result. A separate malloc cache for GGUF expert
bytes would duplicate pages that the OS is already caching, reduce reclaimable
memory, and likely recreate the slower cache experiments flash-moe discarded.

## Implemented defaults

`scripts/run_deepseek_q4_gguf_demo.sh` now defaults to:

- `GGML_DISABLE_CPU_REPACK=1`
- `LLAMA_MMAP_RANDOM=0`
- `GGML_MOE_MADVISE_WILLNEED=1`
- `--fit off`
- `--device none`
- `--no-op-offload`
- `-b 512`
- `-ub 64`
- `--cache-ram 0`

The patched llama.cpp fork can honor `LLAMA_MMAP_RANDOM=1` as an experiment by
disabling mmap prefetch and advising random access on the mapped model file.
flash-moe's I/O notes showed that random-access hints can hurt cold bulk reads,
so the wrapper leaves this disabled by default and lets the kernel choose.

`GGML_MOE_MADVISE_WILLNEED=1` is a lighter hint: after the routed expert IDs are
known inside `mul_mat_id`, thread 0 asks the kernel to start paging in the
selected expert matrix ranges before worker threads enter the dot-product loop.
It keeps the GGUF mmap/page-cache design and does not copy the weights into a
separate heap cache.

## Cold-start expert preload

`scripts/warm_deepseek_q4_expert_cache.py` preloads selected routed expert tensor
ranges from the GGUF file into the macOS page cache before the first prompt. It
does not allocate or pin a custom cache. The warmed pages remain reclaimable
file-backed memory, so macOS can evict them if memory pressure rises.

The preloader can be run directly:

```bash
scripts/warm_deepseek_q4_expert_cache.py --budget-gib 4 --experts-per-layer 8
```

or through the Q4 wrapper:

```bash
PREWARM_EXPERTS=1 PREWARM_BUDGET_GIB=4 scripts/run_deepseek_q4_gguf_demo.sh
```

To only warm the page cache and exit:

```bash
PREWARM_EXPERTS=1 PREWARM_ONLY=1 PREWARM_BUDGET_GIB=4 scripts/run_deepseek_q4_gguf_demo.sh
```

The default preloader ranking uses:

- hash-router tables for the first hash-routed layers, based on the actual
  wrapper prompt or prompts passed to the preloader;
- `exp_probs_b` as a lightweight per-layer prior for the remaining routed
  layers;
- a memory budget that defaults to "auto", capped at 8 GiB and leaving 13 GiB of
  headroom.

Optional `--profile coding|plain|json` seed prompts can be layered in later, but
the wrapper does not enable a profile by default.

The preloader can merge nearby `pread` ranges with `--merge-gap-mib`, and can
tune individual read size with `--chunk-mib`. The Q4 wrapper defaults
`PREWARM_MERGE_GAP_MIB=4` and `PREWARM_CHUNK_MIB=1`, which increased the top-16
hotset read size from 8.21 GiB to 8.30 GiB in the first layout probe, but
reduced range count from 1914 to 1887. Larger gaps quickly become wasteful on
this GGUF layout: 8 MiB reads 8.84 GiB, while 16 MiB reads 11.42 GiB.

See `docs/deepseek-q4-cache-patterns.md` for the first diverse-prompt probe. In
that run, a top-16 static hotset costs about 8.21 GiB but covers only 17-18% of
the exact hash-routed expert selections in layers 0-2. This makes static prewarm
a useful cold-start prior, while pointing toward prompt-adaptive prewarm or a
small predictor as the next experiment.

See `docs/deepseek-q4-load-compute-profiling.md` for the first full-run
load/compute profile. In that run, the profiled process tree was I/O-active for
97.6% of wall time, and the traced decode window was still I/O-active for 92.3%.
That strongly favors async prefetch and routing prediction over compute-kernel
tuning as the next optimization.

## Next possible optimization

The larger step is to bypass GGUF expert tensors for the routed MLP and use the
already-packed `packed_experts_q4/layer_XX.bin` files with a flash-moe-style
`pread` pipeline. That is a separate engine-porting task because the current
llama.cpp graph expects expert tensors to live inside the GGUF backend buffer.
It is feasible, but much more invasive than the page-cache-oriented GGUF path.
