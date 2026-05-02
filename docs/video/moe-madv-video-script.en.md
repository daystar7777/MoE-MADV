# MoE-MADV English Demo Video Script

Target length: about 4 minutes.

## Cold Open

Visual: open on the headline chart, then cut to Activity Monitor or terminal
showing a 64GB Apple Silicon machine.

Narration:

```text
This is DeepSeek V4 Flash, a 284-billion-parameter Mixture-of-Experts model.
The GGUF file used here is about 150 gigabytes on Hugging Face.

The machine is a 64 gigabyte M1 Max.

So the question is not just: can the machine compute the model?
The question is: can the operating system bring the right expert weights into
memory at the right time?
```

On-screen text:

```text
150GB model file
64GB unified memory
284B total-parameter MoE
```

## Why This Is Different

Visual: show a simple terminal split: dense model reuse vs MoE expert routing.
Then show the README thesis section.

Narration:

```text
Dense models tend to reuse the same mapped weight ranges during decode.
DeepSeek V4 Flash is different. It routes each token through a changing subset
of experts.

When the model file is larger than RAM, decode can become dominated by page
faults and NVMe-backed page-in instead of pure matrix multiplication.

That is the reason for MoE-MADV.
```

On-screen text:

```text
Dense model: mostly repeated weight reuse
MoE model: changing expert sets per token
Local bottleneck: expert page loading
```

## Setup And Model Source

Visual: show repository root, then `README.md`, then the exact Hugging Face
model URL in `docs/model-sources-and-parsers.md`.

Narration:

```text
The benchmark model is lovedheart slash DeepSeek-V4-Flash-GGUF,
specifically the DeepSeek-V4-Flash-MXFP4_MOE dot gguf file.

The model file itself is not included in GitHub. The repository only includes
the download script, parser outputs, benchmark scripts, and documentation.
```

Terminal:

```bash
git clone https://github.com/daystar7777/MoE-MADV.git
cd MoE-MADV

scripts/setup_deepseek_gguf_runtime.sh
scripts/download_deepseek_q4_gguf.sh
```

On-screen text:

```text
Model files are local artifacts.
The repo publishes scripts, patches, charts, and reproduction notes.
```

## Live Token Generation

Visual: terminal running the JSON prompt smoke test. If the model is already
downloaded, show the command and the generated output.

Narration:

```text
For the video, I do not want this to be only a chart.
Here is the local machine actually generating tokens from the 150 gigabyte GGUF
file.

First I run a tiny JSON prompt. Then I run a plain English prompt so the viewer
can see normal text generation.
```

Terminal:

```bash
scripts/run_moe_madv_live_generation_demo.sh
```

On-screen text:

```text
Live generation from local GGUF
Prompt 1: JSON
Prompt 2: plain English
```

## The Optimization

Visual: show a code or patch excerpt where `MADV_WILLNEED` is applied after
routing chooses active experts. Then show the four-point summary in the README.

Narration:

```text
The optimization is intentionally small.

The model stays mmap-backed. CPU repacking is disabled, so the weights are not
copied into a second large in-memory representation.

After the MoE router chooses the active experts, the runtime calls
MADV_WILLNEED on those expert matrix ranges before the dot-product workers need
them.

The operating system still owns the file-backed cache, so pages remain
reclaimable.
```

On-screen text:

```text
mmap-backed model
no CPU repack
routing-aware MADV_WILLNEED
OS-managed page cache
```

## Benchmark Result

Visual: show `docs/assets/deepseek-q4-decode-headline.svg`, then the result
table in `docs/deepseek-q4-performance-matrix.md`.

Narration:

```text
The clean decode comparison uses the same model, the same prompts, and no
static prewarm in either case.

With the expert page hint disabled, decode generation averaged 0.98 tokens per
second.

With routing-aware MADV_WILLNEED enabled, it averaged 1.23 tokens per second.

That is a 25.4 percent throughput gain without changing the model.
```

On-screen text:

```text
Baseline: 0.98 tok/s
Optimized: 1.23 tok/s
+25.4% decode throughput
```

## Prefill vs Decode

Visual: show the long-run summary table and the prefill/decode charts.

Narration:

```text
The long-run data also showed that prefill and decode should be optimized
differently.

Prefill behaves like broad layer-level page-in throughput.
Decode is more latency-sensitive because each token may route to a different
expert set.

That is why a single global cache policy is probably not enough.
```

On-screen text:

```text
Prefill: broader throughput problem
Decode: narrower latency problem
Next step: phase-specific prefetch policy
```

## Closing

Visual: show GitHub repository page, then `README.ko.md` and the video docs.

Narration:

```text
The repository includes the patch, scripts, parser artifacts, benchmark data,
charts, and a rebuild process for the packed expert export.

The model files are intentionally excluded, but the exact Hugging Face sources
and revisions are documented.

This is not a claim that a 64 gigabyte machine is the ideal way to run a model
this large. It is a demonstration that for sparse MoE models, local inference is
often about memory residency and page-in timing, not just raw compute.
```

On-screen text:

```text
github.com/daystar7777/MoE-MADV
MoE-MADV = Mixture-of-Experts + MADV_WILLNEED
```
