# Appendix: Applying MoE-MADV on Other Machines

This project was measured on a 64GB Apple M1 Max, but the core idea is not
Apple-specific:

> Keep the large MoE model file-backed, avoid unnecessary repacked copies, and
> make expert page-in happen slightly before decode needs the selected experts.

The exact cache budget changes by machine.

## Baseline Rule

For a local MoE model that is larger than RAM:

1. Keep the model mmap-backed or otherwise file-backed.
2. Avoid loading a second full copy of the model into heap memory.
3. Leave enough RAM for the OS, runtime, KV cache, and temporary buffers.
4. Use broad prefetch only for prefill, where many tokens can amortize it.
5. Use narrow routing-aware hints for decode, where the next token is latency
   sensitive.

The measured optimization here, `MADV_WILLNEED`, is a page-cache hint. It does
not create a custom expert cache and does not pin the model permanently in
application memory.

## 64GB Apple Silicon

This is the machine used for the published benchmark.

- Final GGUF model: 150.23 GB / 139.91 GiB
- Practical policy: keep prewarm optional, enable routed expert `madvise`
- Leave-RAM target used by the prewarmer: about 13 GiB
- Best measured public comparison:
  - no prewarm, no expert hint: 0.98 tok/s decode generation
  - no prewarm, routed `MADV_WILLNEED`: 1.23 tok/s decode generation

On 64GB, static expert prewarm is useful mainly as a cold-start prior. It was
not the steady-state winner in the 5-hour run.

## 128GB Unified-Memory Machines

Examples include NVIDIA DGX Spark-class machines. NVIDIA documents DGX Spark as
having 128GB LPDDR5x coherent unified system memory in its official product and
hardware documentation:

- <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>
- <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>

For this model, 128GB is still smaller than the final 150 GB GGUF file, so the
same file-backed strategy still matters. The difference is that there is more
room for page cache and a larger hot expert set.

Suggested starting policy:

```bash
GGML_DISABLE_CPU_REPACK=1 \
GGML_MOE_MADVISE_WILLNEED=1 \
PREWARM_EXPERTS=1 \
PREWARM_LEAVE_GIB=24 \
PREWARM_MAX_AUTO_GIB=48 \
PREWARM_MERGE_GAP_MIB=4 \
PREWARM_CHUNK_MIB=1 \
scripts/run_deepseek_q4_gguf_demo.sh
```

What to tune:

- Increase `PREWARM_MAX_AUTO_GIB` gradually: try 16, 32, 48, then 64 GiB.
- Keep `PREWARM_LEAVE_GIB` conservative at first. For 128GB, 24 GiB is a safer
  first reserve than the 13 GiB used on the 64GB M1 Max.
- If prefill dominates, try broader layer-level prewarm.
- If decode dominates, keep prewarm narrower and preserve routed
  `MADV_WILLNEED`.
- Re-run the matrix with `prefill` and `decode` suites separately; do not pick a
  single policy from aggregate wall time alone.

Expected difference from 64GB:

- More of the expert working set can stay resident between prompts.
- Static prewarm may become more useful, especially for repeated workloads.
- Decode can still page-fault if routing shifts to cold experts, so
  routing-aware hints remain relevant.

## 192GB-256GB Unified-Memory Machines

At this size, the 150 GB GGUF file can fit mostly or entirely in memory-backed
cache while still leaving room for runtime overhead. The optimization target
shifts:

- First, ensure there is no CPU repack or duplicate model copy.
- Then test whether a one-time model or expert warmup is worth it.
- Decode may become compute-bound sooner, so kernel/backend optimization becomes
  more important than page-in timing.

Suggested experiment:

```bash
scripts/run_deepseek_q4_perf_matrix.py \
  --mode infer \
  --infer-cases no_prewarm_madvise_off,no_prewarm_madvise_on,best_gap4_chunk1_madvise_on \
  --prompts prefill_long_plain,prefill_long_code,decode_json_seed,decode_plain_seed \
  --tokens 24 \
  --context 1024 \
  --repeats 3
```

If `madvise_on` and `madvise_off` converge, the machine has moved past the
page-in bottleneck for this workload.

## Discrete-GPU Servers

On systems with large discrete GPUs, the same principle becomes a placement
problem:

- Put dense/shared layers and frequently reused tensors on GPU memory first.
- Keep less frequently selected experts file-backed or host-backed.
- Use router output to schedule host-to-device prefetch for selected experts.
- Treat prefill and decode differently:
  - prefill can use broader batched transfers;
  - decode needs low-latency transfer of a small routed expert set.

For multi-GPU servers, expert parallelism can replace page-cache tricks if the
expert set can be sharded across GPUs. `MoE-MADV` is most useful when the model
is too large for available accelerator memory and experts must spill to host
memory or NVMe.

## Using `packed_experts_q4` Elsewhere

`packed_experts_q4` is a separate export path from the GGUF benchmark. It is
useful if you are building a custom engine that streams routed experts directly.

Rebuild it from the MLX/safetensors source:

```bash
scripts/download_deepseek_q4_shards_sequential.sh
scripts/repack_deepseek_q4_experts.py
```

On a 128GB+ machine, the packed export gives you another option: preload a much
larger subset of the per-layer packed expert files, or memory-map them directly
in a custom engine. The full export is 137.06 GiB, so it still should not be
blindly copied into heap memory unless the machine has enough RAM left for the
rest of the model, KV cache, and runtime.

## Measurement Checklist

For any new machine, collect at least:

- prefill wall time and decode wall time separately;
- generation tokens/sec, not only total wall time;
- disk-read bytes or page-in proxy if available;
- peak RSS / unified-memory pressure;
- cold-ish first run and warm-cache repeated run;
- `madvise_on` versus `madvise_off`;
- prewarm budget sweep.

The important question is not “does prewarm help?” in general. It is:

> At this memory size, is the bottleneck still expert page-in latency, or has it
> moved to compute/backend throughput?
