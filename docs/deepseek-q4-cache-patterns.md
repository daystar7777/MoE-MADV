# DeepSeek V4 Flash Q4 Cache Pattern Probe

This probe uses diverse prompts to estimate which routed experts are good
preload candidates before running full generation.

## Probe set

- Prompt file: `data/deepseek_q4_cache_probe_prompts.jsonl`
- Prompts: 16
- Categories: plain, JSON, coding, Korean, reasoning, MoE, research, creative
- Tokenized prompt tokens: 366
- Output JSON: `docs/deepseek-q4-cache-patterns.json`
- Hotset JSON: `data/deepseek_q4_probe_hotset_k16.json`

## What is exact today

Layers 0, 1, and 2 have `ffn_gate_tid2eid` hash-router tables. For these
layers, expert selection is determined directly by token id, so the probe can
measure prompt-specific expert routing without running the full model.

The later routed layers do not expose their prompt-specific selected experts
until inference computes hidden states. For those layers, the current probe uses
`exp_probs_b` as a prompt-independent prior.

## Current results

Global top-k coverage for the exact hash-routed layers:

| top-k experts/layer | hash-route avg coverage | worst layer coverage | all-layer prewarm estimate |
| ---: | ---: | ---: | ---: |
| 8 | 10.0% | 9.6% | 4.10 GiB |
| 16 | 17.7% | 16.8% | 8.21 GiB |
| 32 | 29.6% | 29.0% | 16.41 GiB |
| 64 | 47.1% | 46.6% | 32.82 GiB |

The 16-expert hotset is memory-friendly and fits the current conservative
prewarm budget, but it is not highly predictive for the first three hash-routed
layers. Category top-16 sets also overlap weakly: average Jaccard is 0.09, with
some category/layer pairs sharing no top-16 experts.

That means a single static hotset is useful mainly as a cold-start prior, not as
a strong prompt-specific cache plan.

## Implication

Prompt-adaptive prewarm is worth testing:

- Use exact token-id hash routing for layers 0-2.
- Keep `exp_probs_b` prior for layers 3-42 until runtime routing traces exist.
- Prefer a modest top-16 or top-24 all-layer budget first; top-32 already costs
  about 16.4 GiB, which may be useful only when the machine has comfortable
  headroom.

## Predictor idea

A tiny helper model could predict which experts to prefetch, but a tiny LLM is
probably not the first thing to try. The lower-risk path is:

1. Instrument runtime routing for layers 3-42 and collect actual expert choices
   per prompt/token.
2. Build a baseline predictor from token ids, position buckets, category, and
   the exact layer 0-2 hash-router outputs.
3. Try a small multi-label classifier per layer, such as linear/logistic,
   gradient-boosted trees, or a tiny MLP over cheap token features.
4. Output a broad candidate set, not a single answer: top-16/top-32 experts per
   layer with confidence.
5. Feed candidates into an async `pread` priority queue. Exact hash-routed
   experts come first, high-confidence predicted experts second, global prior
   last.

This keeps predictor cost tiny and makes bad predictions mostly waste bandwidth
instead of blocking inference. A small LLM-like transformer only becomes
interesting if cheap feature models cannot beat the static prior.

## Commands

```bash
scripts/analyze_deepseek_q4_cache_patterns.py \
  --prompts data/deepseek_q4_cache_probe_prompts.jsonl \
  --top-k 16 \
  --json-out docs/deepseek-q4-cache-patterns.json \
  --hotset-out data/deepseek_q4_probe_hotset_k16.json
```

The generated hotset can be used by the prewarmer:

```bash
scripts/warm_deepseek_q4_expert_cache.py \
  --hotset-json data/deepseek_q4_probe_hotset_k16.json \
  --budget-gib 8.5
```
