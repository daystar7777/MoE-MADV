# DeepSeek V4 Flash Q4 Load/Compute Profiling

This note tracks whether the current bottleneck is expert/model loading or
actual compute. The current profiler is a proxy, not a hardware stall counter:
it samples macOS `proc_pid_rusage` for page-ins, disk-read bytes, CPU time, RSS,
and process footprint while `llama-cli` runs.

## Commands

```bash
scripts/profile_deepseek_q4_run.py \
  --prewarm \
  --prompt 'Return JSON only: {"status":"ok","note":"trace"}' \
  --tokens 10 \
  --context 1024 \
  --out logs/deepseek_q4_profile_trace_prewarm.json \
  --env LLAMA_EXPERT_TRACE=logs/deepseek_q4_expert_trace.jsonl

scripts/summarize_deepseek_q4_expert_trace.py \
  --trace logs/deepseek_q4_expert_trace.jsonl \
  --json-out logs/deepseek_q4_expert_trace_summary.json

scripts/profile_deepseek_q4_run.py \
  --prewarm \
  --prompt 'Return JSON only: {"status":"ok","note":"madvise"}' \
  --tokens 10 \
  --context 1024 \
  --out logs/deepseek_q4_profile_trace_madvise_prewarm.json \
  --env LLAMA_EXPERT_TRACE=logs/deepseek_q4_expert_trace_madvise.jsonl \
  --env GGML_MOE_MADVISE_WILLNEED=1
```

## Run Result

- Output: `{"status":"ok","note":"trace"}`
- Wall time: 110.14s
- Overall I/O-active proxy: 107.48s, 97.6%
- Overall compute/other proxy: 2.66s, 2.4%
- Average CPU cores after mach-timebase correction: 1.30
- Page-ins: 9,528,180
- Disk read during profiled process tree: 152.19 GiB
- Peak RSS: 51.79 GiB
- Explicit prewarm: 8.21 GiB in 1.88s, 4.36 GiB/s
- llama timing: prompt 0.8 t/s, generation 1.2 t/s

The main process still read about 152 GiB even after the 8.21 GiB expert hotset
prewarm. That means the current prewarm is too small to avoid broad mmap page
faults, and/or a large part of startup touches non-hotset model pages.

## Prefill vs Decode

The trace callback records only `ffn_moe_topk-*` tensors, so it is much lighter
than a full graph dump. Rounds with `n_tokens > 1` are treated as prefill, while
`n_tokens == 1` rounds are decode.

- Trace events: 516
- MoE rounds: 12 total
- Prefill rounds: 3
- Decode rounds: 9
- Prefill trace duration: 29.53s
- Decode trace duration: 8.31s
- Prefill I/O-active proxy: 100.0%
- Decode I/O-active proxy: 92.3%
- Prefill disk read in traced window: 22.44 GiB
- Decode disk read in traced window: 3.03 GiB
- Decode average expert bytes touched per round: 3.08 GiB
- Decode adjacent expert-set overlap: avg Jaccard 0.22, min 0.16

Decode is not compute-clean yet: even token-by-token generation remains mostly
I/O-active by the current page-in/disk-read proxy. The low adjacent-token overlap
also explains why a small static hotset is weak.

## `mmap`, `pread`, and `madvise`

The current GGUF engine still uses one large `mmap`. That is memory-efficient
because file-backed pages are reclaimable and the model is not copied into a
second heap cache. The bad part is timing: if an expert page is first touched
inside the matmul loop, the compute worker takes the page fault and stalls.

`pread` is useful when it happens before the matmul needs the page. It warms the
same file-backed cache, but it copies bytes through a userspace buffer and can
compete with compute for memory bandwidth if used too aggressively. For this
reason the preloader remains bounded by a budget and now has a merge-gap knob
instead of blindly reading large file regions.

`madvise(MADV_WILLNEED)` is the middle option: it keeps mmap and avoids the
userspace copy. The local llama.cpp patch can call it after routed expert IDs are
known in `mul_mat_id`.

First `MADV_WILLNEED` run, with a similar but not identical prompt/token count:

- Output: `{"status":"ok","note":"madvise"}`
- Wall time: 104.75s
- Overall I/O-active proxy: 101.98s, 97.4%
- Overall compute/other proxy: 2.77s, 2.6%
- Average CPU cores: 2.15
- Page-ins: 9,766,469
- Disk read during profiled process tree: 156.09 GiB
- Peak RSS: 51.02 GiB
- Explicit prewarm: 8.21 GiB in 1.91s, 4.29 GiB/s
- llama timing: prompt 1.10 t/s, generation 1.30 t/s
- Trace prefill duration: 25.11s, I/O-active 100.0%, avg CPU cores 6.05
- Trace decode duration: 7.56s, I/O-active 90.1%, avg CPU cores 6.41
- Decode adjacent expert-set overlap: avg Jaccard 0.19, min 0.13

This is not a strict A/B because the generated token count and prompt length
differed. Still, it points in the right direction: selected-expert `madvise`
may shave some page-fault stall from both prefill and decode, but the total run
is still overwhelmingly I/O-active.

## `pread` Range Merge Probe

For the top-16 all-layer hotset (`data/deepseek_q4_probe_hotset_k16.json`):

| merge gap | ranges | actual read | over-read |
| --- | ---: | ---: | ---: |
| 0 MiB | 1914 | 8.21 GiB | 0.00 GiB |
| 0.5 MiB | 1914 | 8.21 GiB | 0.00 GiB |
| 1 MiB | 1914 | 8.21 GiB | 0.00 GiB |
| 4 MiB | 1887 | 8.30 GiB | 0.09 GiB |
| 8 MiB | 1770 | 8.84 GiB | 0.64 GiB |
| 16 MiB | 1530 | 11.42 GiB | 3.21 GiB |
| 32 MiB | 1228 | 18.31 GiB | 10.10 GiB |
| 64 MiB | 753 | 40.31 GiB | 32.10 GiB |

Warm-cache reads are too dependent on prior page-cache state to use as final
proof, but the shape is clear: 4 MiB is the conservative merge candidate, 8 MiB
is a more aggressive candidate, and 16 MiB+ wastes too much RAM/cache bandwidth.

With a 4 MiB merge gap on an already warm cache, chunk size also mattered:

| chunk size | actual read | elapsed | throughput |
| --- | ---: | ---: | ---: |
| 1 MiB | 8.30 GiB | 0.70-0.76s | 10.9-11.8 GiB/s |
| 8 MiB | 8.30 GiB | 0.95s | 8.75 GiB/s |
| 32 MiB | 8.30 GiB | 0.93s | 8.91 GiB/s |
| 64 MiB | 8.30 GiB | 0.92s | 9.00 GiB/s |

This does not prove the cold NVMe optimum, but it argues against one huge
`pread`. The current wrapper default is the measured conservative point:
`PREWARM_MERGE_GAP_MIB=4` and `PREWARM_CHUNK_MIB=1`.

## Immediate Interpretation

1. The machine is not mainly compute-bound in this configuration.
2. The main bottleneck is mmap-backed model/expert page-in during both startup
   and decode.
3. A top-16 all-layer static hotset is too small to cover real decode routing.
4. Per-token decode touches roughly 3 GiB of unique expert byte ranges in the
   current GGUF layout, so useful prefetch must be selective and early.
5. The next useful metric is not only "which experts", but "which expert byte
   ranges were touched before their layer needed them".

## Next Experiments

- Run the same profiler without hotset prewarm while the cache is cold enough to
  estimate the hotset benefit. Without root `purge`, this is approximate.
- Add async `pread` scheduling keyed by the trace/predictor hotset and measure
  whether decode I/O-active wall falls.
- Build a small predictor from actual traced rounds: previous layer top-k,
  previous token top-k, token id, and layer id.
- Compare predictor candidates against three baselines: static `exp_probs_b`,
  exact hash layers, and previous-token same-layer reuse.
