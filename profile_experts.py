"""
profile_experts.py — Profile MoE expert routing frequency for Qwen3.5-397B-A17B.

Loads only router + attention weights (no expert weights needed), runs a prompt
through the model collecting routing decisions at every layer for every token,
then reports:
  - Activation frequency per (layer, expert_id) pair
  - What % of activations come from the top 5%, 10%, 20% of experts
  - Estimated DRAM to pin the top N% of experts

The key insight: we can run attention + router without loading expert weights
(~3-5 GB vs ~200 GB). The expert computation is skipped — we just record which
experts the router *would have* selected, then advance the hidden state using
only the attention residual. This gives realistic routing decisions on real
hidden states, minus the expert MLP contribution.

Usage:
    uv run python profile_experts.py
    uv run python profile_experts.py --prompt "Write a Python function to sort a list"
    uv run python profile_experts.py --tokens 50
"""

import argparse
import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
NUM_LAYERS = 60
NUM_EXPERTS = 512
TOP_K = 10
EXPERT_SIZE_MIB = 6.75  # per-expert weight size in MiB


# ---------------------------------------------------------------------------
# Safetensors I/O (from stream_infer.py)
# ---------------------------------------------------------------------------
def parse_safetensors_header(filepath):
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def read_tensors_direct(filepath, tensor_names, header_cache):
    """Read specific tensors via direct I/O (not mmap). Returns dict of name -> mx.array."""
    if filepath not in header_cache:
        header_cache[filepath] = parse_safetensors_header(filepath)
    header, data_start = header_cache[filepath]

    NP_DTYPE = {
        'U32': np.uint32, 'F32': np.float32, 'F16': np.float16,
        'I32': np.int32, 'I64': np.int64, 'U8': np.uint8,
    }

    sorted_names = sorted(tensor_names, key=lambda n: header[n]['data_offsets'][0])
    result = {}
    with open(filepath, 'rb') as f:
        for name in sorted_names:
            meta = header[name]
            off = meta['data_offsets']
            byte_len = off[1] - off[0]
            f.seek(data_start + off[0])
            raw = f.read(byte_len)
            dtype_str = meta['dtype']
            shape = meta['shape']
            if dtype_str in NP_DTYPE:
                np_arr = np.frombuffer(raw, dtype=NP_DTYPE[dtype_str]).reshape(shape)
                result[name] = mx.array(np_arr)
            elif dtype_str == 'BF16':
                np_uint16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                np_f32 = (np_uint16.astype(np.uint32) << 16).view(np.float32)
                result[name] = mx.array(np_f32).astype(mx.bfloat16)
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
    return result


# ---------------------------------------------------------------------------
# Model loading (router + non-expert weights only)
# ---------------------------------------------------------------------------
def load_model_router_only(model_path):
    """Load model architecture + global weights + all non-expert layer weights.

    This gives us everything needed to run attention + router forward passes
    without the ~200 GB of expert MoE weights. Total memory: ~3-5 GB.
    """
    import re
    from mlx_lm.utils import _get_classes, load_tokenizer

    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # Create model architecture
    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Apply quantization (structure only, no weights yet)
    qconfig = config.get("quantization", config.get("quantization_config", {}))
    if qconfig:
        nn.quantize(model, bits=qconfig["bits"], group_size=qconfig["group_size"])
    model.eval()

    # Build weight index: tensor_name -> safetensors file
    with open(model_path / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Identify non-expert weights (everything except switch_mlp.{gate,up,down}_proj.{weight,scales,biases})
    import re as _re
    expert_pattern = _re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')

    header_cache = {}
    file_to_names = defaultdict(list)

    for name, filename in weight_map.items():
        if expert_pattern.search(name):
            continue  # skip expert weights
        filepath = str(model_path / filename)
        file_to_names[filepath].append(name)

    # Read all non-expert weights via direct I/O
    all_weights = []
    t0 = time.time()
    for filepath, names in sorted(file_to_names.items()):
        tensors = read_tensors_direct(filepath, names, header_cache)
        for name in names:
            if name in tensors:
                all_weights.append((name, tensors[name]))
    load_time = time.time() - t0
    print(f"  Loaded {len(all_weights)} non-expert weight tensors in {load_time:.1f}s")

    # Apply weights to model
    model.load_weights(all_weights, strict=False)
    del all_weights

    # Load tokenizer
    eos_ids = config.get("eos_token_id", [])
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Router-only forward pass
# ---------------------------------------------------------------------------
def collect_routing_decisions(model, tokenizer, prompt, max_tokens):
    """Run a forward pass collecting expert routing decisions at every layer/token.

    For each token position, we run attention + router but SKIP the expert MoE
    computation. Instead, the hidden state advances using only the attention
    residual (h = h + attn_output). This means routing decisions are based on
    realistic hidden states from real attention, just without expert MLP updates.

    Returns:
        activations: np.array of shape [num_tokens, num_layers, top_k]
            Each entry is an expert index (0..511).
        generated_text: str of generated tokens
    """
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers
    num_layers = len(layers)

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    all_routing = []  # list of [num_layers, top_k] arrays per token
    generated_tokens = []

    num_prompt_tokens = input_ids.shape[1]
    total_positions = num_prompt_tokens + max_tokens

    print(f"  Prompt tokens: {num_prompt_tokens}, generating {max_tokens} more")
    print(f"  Total token positions to profile: {total_positions}")

    for token_idx in range(max_tokens + 1):  # +1 for initial prompt processing
        t0 = time.time()

        # Embed
        h = text_model.embed_tokens(input_ids)
        mx.eval(h)

        # Create masks
        fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

        seq_len = h.shape[1]
        token_routing = []  # [seq_len, num_layers, top_k]
        # Initialize per-position routing storage
        for _ in range(seq_len):
            token_routing.append([])

        for i in range(num_layers):
            layer = layers[i]
            c = cache[i]

            # --- Attention (uses real loaded weights) ---
            x_normed = layer.input_layernorm(h)
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer.is_linear:
                r = layer.linear_attn(x_normed, mask, c)
            else:
                r = layer.self_attn(x_normed, mask, c)
            h_mid = h + r
            mx.eval(h_mid)

            # --- Router (uses real loaded weights) ---
            h_post = layer.post_attention_layernorm(h_mid)
            gates = layer.mlp.gate(h_post)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            mx.eval(inds)

            # Record routing decisions for each position in the sequence
            inds_np = np.array(inds.reshape(-1, k).tolist())  # [seq_len, top_k]
            for pos in range(seq_len):
                token_routing[pos].append(inds_np[pos])

            # --- Skip expert MoE, advance with attention residual only ---
            # This means h doesn't get the MoE contribution, but the router
            # decisions are still based on real attention-processed hidden states.
            # For routing profiling this is a reasonable approximation.
            h = h_mid
            mx.eval(h)

        # Collect routing for all positions in this step
        for pos in range(seq_len):
            all_routing.append(np.array(token_routing[pos]))  # [num_layers, top_k]

        # Norm + LM head for sampling
        h_final = text_model.norm(h)
        if lm.args.tie_word_embeddings:
            logits = text_model.embed_tokens.as_linear(h_final)
        else:
            logits = lm.lm_head(h_final)
        mx.eval(logits)

        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        token_id = next_token.item()
        generated_tokens.append(token_id)

        elapsed = time.time() - t0
        total_so_far = len(all_routing)
        if token_idx == 0:
            print(f"  Prompt processing: {elapsed:.1f}s ({seq_len} positions)")
        elif (token_idx) % 10 == 0 or token_idx == max_tokens:
            print(f"  Token {token_idx}/{max_tokens}: {elapsed:.2f}s/tok, "
                  f"{total_so_far} total positions profiled")

        input_ids = next_token.reshape(1, 1)

    text = tokenizer.decode(generated_tokens)
    activations = np.array(all_routing)  # [total_positions, num_layers, top_k]
    return activations, text


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_routing(activations, num_experts=NUM_EXPERTS, expert_size_mib=EXPERT_SIZE_MIB):
    """Analyze expert routing frequency distribution.

    Args:
        activations: np.array [num_tokens, num_layers, top_k] of expert indices
        num_experts: total experts per layer
        expert_size_mib: size of each expert in MiB
    """
    num_tokens, num_layers, top_k = activations.shape
    total_expert_slots = num_tokens * num_layers * top_k
    total_unique_pairs = num_layers * num_experts

    print(f"\n{'='*70}")
    print(f"EXPERT ROUTING ANALYSIS")
    print(f"{'='*70}")
    print(f"  Tokens profiled:       {num_tokens}")
    print(f"  Layers:                {num_layers}")
    print(f"  Experts per layer:     {num_experts}")
    print(f"  Active per token:      {top_k}")
    print(f"  Total activation slots: {total_expert_slots:,}")
    print(f"  Total (layer, expert) pairs: {total_unique_pairs:,}")

    # Count activations per (layer, expert_id) pair
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for layer_i in range(num_layers):
        layer_inds = activations[:, layer_i, :]  # [num_tokens, top_k]
        for expert_id in layer_inds.flatten():
            counts[layer_i, expert_id] += 1

    # Flatten to a sorted frequency list
    flat_counts = counts.flatten()  # [num_layers * num_experts]
    sorted_counts = np.sort(flat_counts)[::-1]  # descending

    # How many pairs were actually activated?
    active_pairs = np.count_nonzero(flat_counts)
    inactive_pairs = total_unique_pairs - active_pairs
    print(f"\n  Active (layer, expert) pairs:   {active_pairs:,} / {total_unique_pairs:,} "
          f"({100*active_pairs/total_unique_pairs:.1f}%)")
    print(f"  Inactive pairs (never chosen):  {inactive_pairs:,} "
          f"({100*inactive_pairs/total_unique_pairs:.1f}%)")

    # Frequency distribution: what % of activations come from top N% of experts
    print(f"\n  {'='*50}")
    print(f"  CUMULATIVE ACTIVATION DISTRIBUTION")
    print(f"  {'='*50}")
    print(f"  {'Top N% of experts':<25} {'% of activations':<20} {'Pairs':<10} {'DRAM (GiB)'}")
    print(f"  {'-'*65}")

    total_activations = flat_counts.sum()
    cumsum = np.cumsum(sorted_counts)

    for pct in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        n_pairs = max(1, int(total_unique_pairs * pct / 100))
        activation_share = cumsum[min(n_pairs, len(cumsum)) - 1] / total_activations * 100
        dram_gib = n_pairs * expert_size_mib / 1024
        print(f"  Top {pct:>3}% ({n_pairs:>6} pairs)   {activation_share:>6.1f}%              {n_pairs:>6}     {dram_gib:>7.1f}")

    # Per-layer statistics
    print(f"\n  {'='*50}")
    print(f"  PER-LAYER EXPERT UTILIZATION")
    print(f"  {'='*50}")
    print(f"  {'Layer':<8} {'Active/512':<12} {'Top-1 expert':<15} {'Top-1 count':<14} {'Top-1 %':<10} {'Entropy'}")
    print(f"  {'-'*70}")

    per_layer_entropy = []
    for layer_i in range(num_layers):
        layer_counts = counts[layer_i]
        active = np.count_nonzero(layer_counts)
        top1_expert = np.argmax(layer_counts)
        top1_count = layer_counts[top1_expert]
        total_layer = layer_counts.sum()
        top1_pct = top1_count / total_layer * 100 if total_layer > 0 else 0

        # Shannon entropy (normalized to [0, 1] where 1 = uniform)
        probs = layer_counts / total_layer if total_layer > 0 else np.zeros_like(layer_counts, dtype=float)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(num_experts)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        per_layer_entropy.append(norm_entropy)

        if layer_i < 10 or layer_i >= num_layers - 5 or layer_i % 10 == 0:
            print(f"  {layer_i:<8} {active:<12} {top1_expert:<15} {top1_count:<14} {top1_pct:<10.1f} {norm_entropy:.3f}")

    avg_entropy = np.mean(per_layer_entropy)
    print(f"  {'...'}")
    print(f"  {'avg':<8} {'':12} {'':15} {'':14} {'':10} {avg_entropy:.3f}")

    # DRAM estimation for pinning strategies
    print(f"\n  {'='*50}")
    print(f"  DRAM PINNING ESTIMATES")
    print(f"  {'='*50}")
    print(f"  Expert size: {expert_size_mib} MiB each")
    print(f"  Total model experts: {total_unique_pairs:,} ({total_unique_pairs * expert_size_mib / 1024:.1f} GiB)")
    print()

    # Strategy 1: Pin experts that cover N% of activations
    print(f"  Strategy: Pin enough experts to cover N% of all activations")
    print(f"  {'Target coverage':<20} {'Experts needed':<18} {'DRAM (GiB)':<12} {'% of total'}")
    print(f"  {'-'*62}")
    for target_pct in [50, 75, 80, 85, 90, 95, 99]:
        target = total_activations * target_pct / 100
        n_needed = np.searchsorted(cumsum, target) + 1
        dram_gib = n_needed * expert_size_mib / 1024
        pct_of_total = n_needed / total_unique_pairs * 100
        print(f"  {target_pct:>3}% coverage        {n_needed:>6}            {dram_gib:>7.1f}       {pct_of_total:>5.1f}%")

    # Strategy 2: Pin top-N per layer (uniform budget across layers)
    print()
    print(f"  Strategy: Pin top-N experts per layer (uniform budget)")
    print(f"  {'N per layer':<15} {'Total pairs':<15} {'DRAM (GiB)':<12} {'Avg coverage %'}")
    print(f"  {'-'*55}")
    for n_per_layer in [10, 20, 30, 50, 75, 100, 128, 200, 256]:
        total_pairs = n_per_layer * num_layers
        dram_gib = total_pairs * expert_size_mib / 1024
        # Compute average coverage per layer
        coverages = []
        for layer_i in range(num_layers):
            layer_counts = counts[layer_i]
            layer_total = layer_counts.sum()
            top_n_sum = np.sort(layer_counts)[::-1][:n_per_layer].sum()
            coverages.append(top_n_sum / layer_total * 100 if layer_total > 0 else 0)
        avg_cov = np.mean(coverages)
        print(f"  {n_per_layer:>3} / layer      {total_pairs:>6}          {dram_gib:>7.1f}       {avg_cov:>5.1f}%")

    # Hotspot analysis: which (layer, expert) pairs are most activated?
    print(f"\n  {'='*50}")
    print(f"  TOP 20 HOTTEST (LAYER, EXPERT) PAIRS")
    print(f"  {'='*50}")
    print(f"  {'Rank':<6} {'Layer':<8} {'Expert':<10} {'Count':<10} {'% of total'}")
    print(f"  {'-'*44}")
    flat_indices = np.argsort(flat_counts)[::-1]
    for rank in range(min(20, len(flat_indices))):
        idx = flat_indices[rank]
        layer_i = idx // num_experts
        expert_id = idx % num_experts
        count = flat_counts[idx]
        pct = count / total_activations * 100
        print(f"  {rank+1:<6} {layer_i:<8} {expert_id:<10} {count:<10} {pct:.2f}%")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Profile MoE expert routing frequency")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=30,
                        help="Number of tokens to generate beyond prompt")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH),
                        help="Path to model directory")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"MoE Expert Routing Profiler")
    print(f"{'='*70}")
    print(f"  Model:  {model_path.name}")
    print(f"  Config: {NUM_LAYERS} layers, {NUM_EXPERTS} experts/layer, top-{TOP_K}")
    print(f"  Prompt: {args.prompt[:60]}{'...' if len(args.prompt) > 60 else ''}")
    print(f"  Tokens: {args.tokens}")
    print()

    # --- Load model (router + attention only, no expert weights) ---
    print("Loading model (non-expert weights only)...")
    t0 = time.time()
    model, tokenizer = load_model_router_only(model_path)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Set wired memory limit (we only need ~5 GB)
    mx.set_wired_limit(int(8 * 1024**3))

    # --- Collect routing decisions ---
    print(f"\nCollecting routing decisions...")
    t0 = time.time()
    activations, text = collect_routing_decisions(model, tokenizer, args.prompt, args.tokens)
    collect_time = time.time() - t0
    print(f"\n  Collection complete in {collect_time:.1f}s")
    print(f"  Generated text: {text[:100]}{'...' if len(text) > 100 else ''}")

    # --- Analyze ---
    counts = analyze_routing(activations)

    # --- Save raw data ---
    out_path = Path("expert_routing_profile.npz")
    np.savez_compressed(
        out_path,
        activations=activations,
        counts=counts,
        prompt=args.prompt,
        num_tokens=activations.shape[0],
    )
    print(f"\n  Raw data saved to {out_path}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
