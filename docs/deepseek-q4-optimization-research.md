# DeepSeek Q4 Optimization Research Notes

The local profile says this run is dominated by model/expert page-in, not pure
matrix compute. Recent MoE offloading papers point to a consistent direction:
trace real expert activations, predict a broad next-layer/next-token candidate
set, then overlap asynchronous loading with useful computation.

## Most Relevant Ideas

- MoE-Infinity targets personal machines and batch size one. Its central idea is
  sparsity-aware expert tracing, then using selected traces to guide expert cache
  replacement and prefetching. This maps closely to our single-user Mac setup.
  Source: https://arxiv.org/abs/2401.14361

- Fate predicts experts from adjacent-layer gate inputs and reports separate
  prefill/decode speedups. This is directly relevant to the user's idea of a
  small predictor: the predictor should probably use cheap gate/hidden features,
  not a second LLM. Source: https://arxiv.org/abs/2502.12224

- FineMoE combines fine-grained expert selection patterns with semantic prompt
  hints for prefetching/caching/offloading. Our prompt-category hotset probe is a
  small local version of the same hypothesis, but the low category overlap says
  prompt semantics alone are not enough. Source: https://arxiv.org/abs/2502.05370

- PreScope proposes a learnable layer-aware predictor, global prefetch-aware
  scheduling, and async I/O. The scheduling part matters because a Mac's NVMe,
  CPU, and unified memory bandwidth can compete; overfetching can slow compute.
  Source: https://arxiv.org/abs/2509.23638

- SP-MoE uses speculative decoding to prefetch target-model experts. This is
  attractive later, but it requires a draft/target setup and is more invasive
  than first building a trace-driven predictor. Source:
  https://arxiv.org/abs/2510.10302

- Pre-gated MoE addresses the sequential dependency directly by making expert
  choice available earlier. It is conceptually relevant, but changing DeepSeek V4
  routing would be a model-level algorithm change, so it is not the first local
  Q4 optimization. Source:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf

- Google TurboQuant is about online vector quantization and especially KV-cache
  compression. It is useful for long-context memory and attention throughput, but
  it does not directly solve the current measured bottleneck: Q4 expert/model
  pages being read from NVMe through mmap. Source: https://arxiv.org/abs/2504.19874

## Local Strategy

The most rational path from these papers is:

1. Keep the current model fixed.
2. Keep measuring prefill and decode separately.
3. Use `LLAMA_EXPERT_TRACE` to collect real expert choices.
4. Train or fit a tiny predictor over cheap features.
5. Use predictor output only as a prefetch priority list, not as a replacement
   for the router.
6. Implement async page-cache warming with bounded bandwidth and cancellation.
7. Optimize for decode first, because repeated token-by-token page-ins dominate
   perceived generation speed.

## Predictor Candidates

Start with cheap baselines before a tiny LLM:

- previous token, same layer top-k;
- previous layer top-k for the same token;
- hash-router output from layers 0-2;
- token id and simple token class;
- prompt category or task tag if known;
- `exp_probs_b` as the global prior.

Then try a tiny multi-label classifier per layer. A small transformer/LLM only
becomes interesting if these cheap predictors cannot beat the static prior and
previous-token reuse.
