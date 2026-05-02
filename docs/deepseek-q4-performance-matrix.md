# MoE-MADV: 141GB DeepSeek V4 Flash Q4 on 64GB RAM

![Decode speed headline](assets/deepseek-q4-decode-headline.svg)

## Headline Result

The important result is decode generation speed, not the implementation detail.
On the decode-heavy benchmark, the baseline is the GGUF mmap path with no static
prewarm and no expert-page hint. The optimized path keeps the same model and the
same no-prewarm steady-state policy, but enables routed expert
`MADV_WILLNEED`.

| mode | prewarm | expert page hint | decode generation | wall time |
| --- | ---: | --- | ---: | ---: |
| baseline | off | off | 0.98 tok/s | 115.5s |
| optimized | off | `MADV_WILLNEED` | 1.23 tok/s | 103.1s |

That is a **+25.4% decode generation throughput gain** and a **10.7% reduction
in end-to-end decode wall time** on a 141GB DeepSeek V4 Flash Q4 GGUF model on a
64GB Apple Silicon machine.

The optimization is deliberately small: after the MoE router decides which
experts are needed, the CPU backend asks macOS to page in the selected expert
matrix ranges before worker threads enter the dot-product loop. No custom heap
cache is added, and the model remains mmap-backed and reclaimable by the OS.

## Thesis: DeepSeek V4 Flash Moves the Bottleneck

DeepSeek V4 Flash is not just a larger dense model. It is a large sparse MoE
model whose active weight set changes with the input and with each generated
token. On a local machine where the 141GB model is larger than RAM, that changes
the practical optimization target.

For a dense model, the repeated decode path tends to reuse the same mapped
weights. For this MoE model, decode repeatedly asks for different routed expert
matrices. The local bottleneck becomes:

> Can the OS make the right expert pages resident before decode needs them?

That is why the winning optimization here is not a new matmul kernel or a bigger
static cache. It is routing-aware page-in timing.

## Estimated Bottlenecks

The profiler is a proxy, not a hardware stall counter. It samples macOS
`proc_pid_rusage` for page-ins, disk-read bytes, CPU time, RSS, and wall time.
Even with that caveat, the signal was strong: this run is dominated by
mmap-backed model/expert page-in, not pure arithmetic.

Early traced run, before the final benchmark matrix:

| metric | value |
| --- | ---: |
| wall time | 110.14s |
| I/O-active proxy | 97.6% |
| disk read | 152.19 GiB |
| peak RSS | 51.79 GiB |
| traced prefill I/O-active | 100.0% |
| traced decode I/O-active | 92.3% |
| decode expert bytes touched per round | ~3.08 GiB |
| adjacent decode expert-set overlap | avg Jaccard 0.22 |

The key observation is that both phases are I/O-bound, but not in the same way:

- **Prefill** touches a broad expert set for many prompt tokens at once. Its
  bottleneck looks like wide, layer-level page-in throughput. It can tolerate
  broader prefetch if the reads are bounded.
- **Decode** touches a smaller but changing expert set token by token. Its
  bottleneck is latency: page faults land directly in the token generation path.
  Over-reading is more dangerous because the next token's expert set may differ.

This is why the report treats prefill and decode separately. Static all-layer
prewarm was too blunt for steady-state decode, while selected-expert
`MADV_WILLNEED` improved decode generation throughput because it moves page-in a
little earlier after routing is known.

## What Was Measured

This records the first condition matrix for the local DeepSeek V4 Flash Q4 GGUF
runtime. The goal is to separate three effects:

- page-cache prewarm read strategy;
- runtime expert `madvise(MADV_WILLNEED)`;
- full generation wall time.

The measurements are not cold-cache perfect because macOS does not expose a safe
non-root cache purge path here. Treat them as directional, especially when
comparing prewarm/no-prewarm after earlier runs.

## Matrix Runner

```bash
scripts/run_deepseek_q4_perf_matrix.py --mode prewarm --prewarm-cases all

scripts/run_deepseek_q4_perf_matrix.py \
  --mode infer \
  --infer-cases no_prewarm_madvise_off,best_gap4_chunk1_madvise_on,gap4_chunk1_madvise_off,no_prewarm_madvise_on \
  --prompts json \
  --tokens 6 \
  --context 1024
```

The runner writes JSONL rows and a Markdown table under `logs/`.

## Five-Hour Data Collection

The longer GitHub-ready dataset is collected by:

```bash
scripts/start_deepseek_q4_longrun_5h.sh
```

The current 5-hour run was launched under:

```text
logs/deepseek_q4_longrun_5h_20260502_093619
```

and `logs/deepseek_q4_longrun_latest` points at the active dataset. The run uses
12 round-robin specs:

- prefill suite: long prompt, 2 generated tokens;
- decode suite: short prompt, 24 generated tokens;
- cases: `best_gap4_chunk1_madvise_on`, `gap4_chunk1_madvise_off`,
  `no_prewarm_madvise_on`;
- prompts: 2 per suite;
- 45 seconds between runs.

New long-run defaults also include `no_prewarm_madvise_off`, so future datasets
collect the all-off baseline used by the headline chart automatically.

Each completed run updates:

- `runs.jsonl`: raw rows for all runs;
- `runs.csv`: spreadsheet-friendly raw rows;
- `summary.csv`: grouped averages and standard deviations;
- `README.md`: table and chart index;
- `charts/*.svg`: GitHub-renderable charts.

### Five-Hour Result

The run completed with 99/99 successful inference runs and no non-zero return
codes. Generated artifacts:

- `logs/deepseek_q4_longrun_5h_20260502_093619/runs.jsonl`
- `logs/deepseek_q4_longrun_5h_20260502_093619/runs.csv`
- `logs/deepseek_q4_longrun_5h_20260502_093619/summary.csv`
- `logs/deepseek_q4_longrun_5h_20260502_093619/README.md`
- `logs/deepseek_q4_longrun_5h_20260502_093619/charts/*.svg`

One caveat: this first long run did not create `traces/` before launching
`LLAMA_EXPERT_TRACE`, so trace JSONL files were not produced. The phase metrics
below use the token-timing estimate from `profile_deepseek_q4_run.py`, not exact
trace windows. The long-run harness has been fixed for future trace collection.

| suite | case | n | wall s | disk GiB | prompt t/s | gen t/s | prefill s | decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | best_gap4_chunk1_madvise_on | 16 | 110.98 | 164.28 | 1.14 | 1.41 | 10.11 | 14.28 |
| decode | gap4_chunk1_madvise_off | 16 | 120.08 | 156.03 | 0.77 | 1.08 | 14.91 | 19.14 |
| decode | no_prewarm_madvise_on | 16 | 102.25 | 151.87 | 1.12 | 1.42 | 10.40 | 14.11 |
| prefill | best_gap4_chunk1_madvise_on | 18 | 149.52 | 215.23 | 1.74 | 2.09 | 49.91 | 1.00 |
| prefill | gap4_chunk1_madvise_off | 17 | 184.06 | 225.22 | 1.24 | 1.62 | 68.07 | 1.49 |
| prefill | no_prewarm_madvise_on | 16 | 146.56 | 214.01 | 1.74 | 2.04 | 49.74 | 1.04 |

Interpretation:

- `madvise` is consistently valuable. Turning it off hurt both suites:
  prefill wall rose from roughly 147-150s to 184s, and decode wall rose from
  roughly 102-111s to 120s.
- Static top-16 prewarm is not a clear steady-state win. `no_prewarm_madvise_on`
  was fastest in both suites after the cache was already warm.
- Prefill remains deeply I/O-bound: estimated prefill I/O-active was 100% for
  all cases.
- Decode also remains I/O-bound, but is more latency-sensitive: decode
  I/O-active stayed around 96% in the decode suite.
- Next optimization should be phase-specific: broader layer-level prefetch for
  prefill, precise routing-aware just-in-time hints for decode.

The different bottleneck shape matters. In the prefill suite, disabling
`madvise` increased the estimated prefill window from about 50s to 68s. In the
decode suite, disabling it increased the estimated decode window from about 14s
to 19s. That suggests the same primitive helps both phases, but the next step
should not be a single global cache policy: prefill wants broader, throughput
oriented prefetch; decode wants earlier, narrower, latency-oriented hints.

### Decode Baseline Addendum

For the public headline chart, a smaller decode-only addendum measured the true
all-off steady-state baseline:

```bash
scripts/run_deepseek_q4_perf_matrix.py \
  --mode infer \
  --infer-cases no_prewarm_madvise_off,no_prewarm_madvise_on \
  --prompts decode_json_seed,decode_plain_seed \
  --tokens 24 \
  --context 1024 \
  --repeats 3 \
  --out-dir logs/deepseek_q4_decode_baseline_20260502_161653
```

| case | n | wall s | disk GiB | prompt t/s | generation t/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_prewarm_madvise_off | 6 | 115.55 | 152.38 | 0.75 | 0.98 |
| no_prewarm_madvise_on | 6 | 103.13 | 152.23 | 1.02 | 1.23 |

This is the cleanest public comparison: same model, same prompts, same no-static
prewarm policy, only the routed expert page hint changes.

## Prewarm I/O Matrix

Source: `logs/deepseek_q4_perf_matrix_prewarm_20260502_091032/summary.md`

| case | gap MiB | chunk MiB | ranges | read GiB | overread | seconds | GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gap0_chunk1 | 0 | 1 | 1914 | 8.21 | 0.0% | 0.67 | 12.18 |
| gap4_chunk1 | 4 | 1 | 1887 | 8.30 | 1.1% | 0.63 | 13.18 |
| gap8_chunk1 | 8 | 1 | 1770 | 8.84 | 7.8% | 0.72 | 12.32 |
| gap4_chunk8 | 4 | 8 | 1887 | 8.30 | 1.1% | 0.91 | 9.08 |
| gap4_chunk32 | 4 | 32 | 1887 | 8.30 | 1.1% | 0.93 | 8.91 |
| gap4_chunk64 | 4 | 64 | 1887 | 8.30 | 1.1% | 0.93 | 8.87 |

Read strategy takeaway: for this hotset and this GGUF layout, `gap=4 MiB` and
`chunk=1 MiB` is the best current prewarm point. A bigger merge gap reduces range
count but over-reads too much; bigger chunks are slower on the warm-cache probe.

## Inference Matrix

Source:

- `logs/deepseek_q4_perf_matrix_infer_20260502_091109/summary.md`
- `logs/deepseek_q4_perf_matrix_infer_prewarm_variants_20260502_091632/summary.md`

Prompt: `json`, 13 prompt tokens, 6 generated tokens, context 1024.

| case | prewarm | madvise | gap | chunk | wall s | I/O active | disk GiB | prompt t/s | gen t/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_prewarm_madvise_on | 0 | 1 | 4 | 1 | 93.74 | 99.2% | 146.02 | 1.10 | 1.40 |
| best_gap4_chunk1_madvise_on | 1 | 1 | 4 | 1 | 95.58 | 96.8% | 144.79 | 1.10 | 1.50 |
| gap0_chunk1_madvise_on | 1 | 1 | 0 | 1 | 99.64 | 97.3% | 155.13 | 1.10 | 1.50 |
| gap8_chunk1_madvise_on | 1 | 1 | 8 | 1 | 100.23 | 97.3% | 155.55 | 1.10 | 1.50 |
| gap4_chunk8_madvise_on | 1 | 1 | 4 | 8 | 99.41 | 97.6% | 153.88 | 1.10 | 1.50 |
| gap4_chunk1_madvise_off | 1 | 0 | 4 | 1 | 108.98 | 97.7% | 155.22 | 0.80 | 1.10 |

Inference takeaway: `GGML_MOE_MADVISE_WILLNEED=1` is the clearest win so far.
Turning it off made the comparable prewarm run about 13.4s slower and reduced
both prompt and generation throughput.

The static top-16 prewarm is not a clear full-run win once the cache is already
warm. It can still be useful as an explicit cold-start action, but it should stay
optional. The runtime default should emphasize `madvise`, not mandatory static
prewarm.

## Current Defaults

- Keep `GGML_MOE_MADVISE_WILLNEED=1` enabled by default.
- Keep `PREWARM_EXPERTS=0` by default.
- When prewarm is requested, use `PREWARM_MERGE_GAP_MIB=4` and
  `PREWARM_CHUNK_MIB=1`.

## Next Development Direction

The matrix says that optimizing standalone `pread` warmup is now less important
than moving expert paging earlier in the real graph.

The promising next step is phase-aware expert prefetch:

1. capture `ffn_moe_topk-*` as soon as routing is computed;
2. map layer/expert IDs to the three expert tensors for that layer;
3. issue `madvise(MADV_WILLNEED)` or bounded async `pread` for gate/up/down
   before the corresponding `mul_mat_id` kernels fault the pages.

That would turn the current same-op hint into a true routing-aware prefetcher.

## Phase-Specific Strategy

Prefill and decode should not use the exact same policy.

Prefill:

- many prompt tokens are evaluated together;
- each routed layer sees a much wider expert set;
- the priority is throughput and broad sequential-ish page-in;
- a layer-level merged prefetch window can be worthwhile, even if it reads a bit
  more than the exact expert set.

Decode:

- only one token is evaluated at a time;
- the next routed expert set changes often and adjacent overlap was low in the
  trace;
- the priority is tail latency per token;
- over-reading is more harmful, so hints should be precise, early, and bounded.

Concretely, prefill can tolerate `pread`/`madvise` over a wider merged set,
while decode should favor just-in-time `madvise` plus a tiny predictor or
previous-layer/previous-token cache hint. Static all-layer prewarm is too blunt
for decode.

## Appendix: Attempts and Dead Ends

The negative results are part of the result. The final path looks small because
several larger ideas were measured, rejected, or postponed.

### Runtime and Model Path

| attempt | result | decision |
| --- | --- | --- |
| Port the original Qwen-focused `flash-moe` engine directly to DeepSeek V4 | Too invasive for the first working run. DeepSeek V4 uses different architecture pieces, tensor names, routing, and MXFP4 expert layout. | Keep the Qwen engine intact and use a patched llama.cpp GGUF path for the first end-to-end Q4 run. |
| Use the full MLX 4-bit HF shard path as the primary runtime | The Q4 shards were downloaded and routed experts were repacked, but a full MLX/engine path would require a larger model implementation port. | Keep the packed experts for a future flash-moe-style engine port; do not block the first Q4 run on it. |
| Switch to another smaller Q4 GGUF after the first load failure | A `tecaprovn` Q4_K_M download was started, then stopped after deciding not to change models. | Stay on the 141GB lovedheart MXFP4 MoE GGUF and optimize that model. |
| Load lovedheart MXFP4 GGUF with default CPU repacking | macOS killed the process under memory pressure before token generation. The extra repacked copy was the problem. | Add `GGML_DISABLE_CPU_REPACK=1`; this made the 141GB model viable on 64GB RAM. |

### Cache and I/O Strategy

| attempt | result | decision |
| --- | --- | --- |
| Build a separate malloc cache for GGUF expert bytes | It would duplicate file-backed mmap pages that macOS already caches, reducing reclaimable memory. This repeats a class of cache experiments that hurt the original flash-moe path. | Do not add a heap expert cache for the GGUF path. Trust the OS page cache. |
| Make static top-16 all-layer expert prewarm the main optimization | It warms 8.21 GiB, but covers only about 17-18% of exact hash-layer selections in the diverse prompt probe. In the 5-hour steady-state run, `no_prewarm + madvise_on` beat static prewarm. | Keep prewarm optional for cold-start experiments; do not present it as the main speedup. |
| Increase prewarm merge gap aggressively | 8 MiB gap read 8.84 GiB, 16 MiB read 11.42 GiB, 32 MiB read 18.31 GiB, and 64 MiB read 40.31 GiB for the same 8.21 GiB hotset. | Use 4 MiB only when prewarming; larger gaps over-read too much. |
| Use larger `pread` chunks for prewarm | Warm-cache probe at 4 MiB merge gap: 1 MiB chunks read 8.30 GiB in 0.70-0.76s, while 8/32/64 MiB chunks were around 0.92-0.95s. | Default prewarm chunk is 1 MiB, not one huge read. |
| Treat mmap demand faults as acceptable | Profiling showed the process was I/O-active for roughly 97% of wall time, and decode remained I/O-active as well. | Add routed expert `MADV_WILLNEED` so page-in starts before worker dot products touch the pages. |
| Use `LLAMA_MMAP_RANDOM=1` by default | Prior flash-moe I/O notes showed random-access hints can hurt cold bulk reads. It also does not solve same-op expert page faults. | Leave it available as an experiment, but default it off. |

### Measurement and Instrumentation

| attempt | result | decision |
| --- | --- | --- |
| Use only llama.cpp timing output | It gives prompt/generation tok/s, but hides page-in wall time and cannot split I/O pressure. | Add `profile_deepseek_q4_run.py` using macOS `proc_pid_rusage` for page-ins, disk reads, RSS, CPU, and wall-time proxies. |
| Dump full graph tensors for routing analysis | Too expensive and noisy. | Add `LLAMA_EXPERT_TRACE` that captures only `ffn_moe_topk-*` and `ffn_moe_hash_topk-*` int32 tensors. |
| First 5-hour run with trace enabled | The run itself succeeded, but the trace directory was not created before launching, so trace JSONL files were absent. | Fix the long-run harness to create `traces/`, `profiles/`, and `logs/` before each run. Treat that dataset's phase numbers as token-timing estimates. |

### Ideas Postponed

| idea | why not first | next condition to revisit |
| --- | --- | --- |
| Tiny LLM or classifier to predict experts | Static prompt-category overlap was low, and a predictor needs clean trace data first. | Revisit after collecting trace JSONL with exact per-layer expert choices. Start with cheap baselines before a tiny LLM. |
| Async `pread` pipeline for experts | More invasive than `MADV_WILLNEED`, and can compete with compute for unified memory bandwidth. | Revisit once trace data identifies how early and how much to prefetch per phase. |
| Bypass GGUF tensors and use packed expert files directly | Feasible, but it changes the llama.cpp graph/backend ownership model. | Treat as a separate engine-porting project, closer to flash-moe proper. |
| KV-cache compression / TurboQuant-style ideas | Useful for long context, but this profile is dominated by expert/model page-in, not KV cache memory. | Revisit only after expert page-in stops dominating decode wall time. |

### What Survived

The current public result is intentionally conservative:

- keep the 141GB Q4 model fixed;
- keep the runtime mmap-backed so macOS can reclaim file pages;
- disable CPU repacking to avoid an extra in-memory copy;
- avoid mandatory static prewarm in steady state;
- use `MADV_WILLNEED` after routed expert IDs are known;
- measure prefill and decode separately.
