#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT.parent / "deepseek-v4-experiments"
DEFAULT_GGUF = EXP_ROOT / "models" / "lovedheart-deepseek-v4-flash-gguf" / "DeepSeek-V4-Flash-MXFP4_MOE.gguf"
DEFAULT_LLAMA_DIR = EXP_ROOT / "llama.cpp-deepseek-v4-flash"
DEFAULT_GGUF_PY = DEFAULT_LLAMA_DIR / "gguf-py"
DEFAULT_TOKENIZER = DEFAULT_LLAMA_DIR / "build" / "bin" / "llama-tokenize"
DEFAULT_PROMPTS = ROOT / "data" / "deepseek_q4_cache_probe_prompts.jsonl"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

ROUTED_COMPONENTS = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


def add_gguf_import_path(path):
    if path.exists():
        sys.path.insert(0, str(path))


def run_capture(args):
    result = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, args))}\n{result.stdout}")
    return result.stdout


def tokenize_prompt(tokenizer, model_path, prompt):
    if not tokenizer.exists():
        raise FileNotFoundError(
            f"{tokenizer} is missing. Run scripts/setup_deepseek_gguf_runtime.sh to build llama-tokenize."
        )
    out = run_capture([str(tokenizer), "-m", str(model_path), "-p", prompt, "--no-bos"])
    tokens = []
    for line in out.splitlines():
        match = re.match(r"\s*(\d+)\s+->", line)
        if match:
            tokens.append(int(match.group(1)))
    if not tokens:
        raise RuntimeError("tokenizer produced no token ids")
    return tokens


def load_prompts(path):
    prompts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            for key in ("id", "category", "prompt"):
                if key not in item:
                    raise ValueError(f"{path}:{line_no} missing {key}")
            prompts.append(
                {
                    "id": str(item["id"]),
                    "category": str(item["category"]),
                    "prompt": str(item["prompt"]),
                }
            )
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def field_value(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    value = field.contents()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def parse_layer_kind(name):
    match = re.match(r"blk\.(\d+)\.(.+)", name)
    if not match:
        return None, name
    return int(match.group(1)), match.group(2)


def has_hash_router(tensors):
    return "ffn_gate_tid2eid.weight" in tensors or "ffn_gate_tid2eid" in tensors


def build_layout(reader):
    expert_count = int(field_value(reader, "deepseek4.expert_count"))
    per_layer = defaultdict(dict)
    for tensor in reader.tensors:
        layer, kind = parse_layer_kind(tensor.name)
        if layer is not None:
            per_layer[layer][kind] = tensor

    routed = {}
    for layer, tensors in per_layer.items():
        components = []
        complete = True
        for kind in ROUTED_COMPONENTS:
            tensor = tensors.get(kind)
            if tensor is None:
                complete = False
                break
            components.append(
                {
                    "kind": kind,
                    "offset": int(tensor.data_offset),
                    "n_bytes": int(tensor.n_bytes),
                    "stride": int(tensor.n_bytes) // expert_count,
                }
            )
        if complete:
            routed[layer] = components
    return expert_count, per_layer, routed


def hash_counts_from_tokens(per_layer, token_ids):
    counts = defaultdict(Counter)
    totals = {}
    for layer, tensors in per_layer.items():
        tensor = tensors.get("ffn_gate_tid2eid.weight") or tensors.get("ffn_gate_tid2eid")
        if tensor is None:
            continue
        table = tensor.data
        vocab = table.shape[0]
        route_total = 0
        for token_id in token_ids:
            if 0 <= token_id < vocab:
                for expert in table[token_id]:
                    counts[layer][int(expert)] += 1
                    route_total += 1
        totals[layer] = route_total
    return counts, totals


def bias_counts(per_layer):
    counts = defaultdict(Counter)
    for layer, tensors in per_layer.items():
        tensor = tensors.get("exp_probs_b") or tensors.get("exp_probs_b.bias")
        if tensor is None:
            continue
        values = tensor.data
        order = sorted(range(len(values)), key=lambda idx: float(values[idx]), reverse=True)
        for rank, expert in enumerate(order):
            counts[layer][int(expert)] = float(values[expert]) - rank * 1e-9
    return counts


def merge_layer_counts(items):
    merged = defaultdict(Counter)
    for counts in items:
        for layer, counter in counts.items():
            merged[layer].update(counter)
    return merged


def top_items(counter, k):
    return [{"expert": int(expert), "count": float(count)} for expert, count in counter.most_common(k)]


def top_set(counter, k):
    return {int(expert) for expert, _ in counter.most_common(k)}


def coverage(counter, selected):
    total = sum(counter.values())
    if not total:
        return 0.0
    return float(sum(counter[expert] for expert in selected) / total)


def jaccard(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def expert_bytes_by_layer(routed):
    result = {}
    for layer, components in routed.items():
        result[layer] = sum(component["stride"] for component in components)
    return result


def build_hotset_payload(counts, routed, experts_per_layer, source):
    expert_bytes = expert_bytes_by_layer(routed)
    layers = {}
    experts = []
    total = 0
    for layer in sorted(routed):
        if layer not in counts:
            continue
        selected = []
        for expert, score in counts[layer].most_common(experts_per_layer):
            item_bytes = expert_bytes[layer]
            selected.append(int(expert))
            experts.append(
                {
                    "layer": int(layer),
                    "expert": int(expert),
                    "score": float(score),
                    "bytes": int(item_bytes),
                }
            )
            total += item_bytes
        if selected:
            layers[str(layer)] = selected
    return {
        "format": "deepseek-q4-hotset-v1",
        "source": source,
        "experts_per_layer": experts_per_layer,
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "layers": layers,
        "experts": experts,
    }


def parse_int_list(value):
    result = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def prewarm_estimate_for_k(counts, routed, k):
    total = 0
    layers = 0
    expert_bytes = expert_bytes_by_layer(routed)
    for layer in sorted(routed):
        if layer not in counts:
            continue
        selected = counts[layer].most_common(k)
        if not selected:
            continue
        layers += 1
        total += len(selected) * expert_bytes[layer]
    return {
        "top_k": k,
        "layers": layers,
        "experts": sum(min(k, len(counts[layer])) for layer in routed if layer in counts),
        "total_bytes": total,
        "total_gib": total / 1024**3,
    }


def global_hash_coverage_for_k(global_counts, hash_layers, k):
    by_layer = {}
    for layer in hash_layers:
        selected = top_set(global_counts[layer], k)
        by_layer[str(layer)] = coverage(global_counts[layer], selected)
    values = list(by_layer.values())
    return {
        "top_k": k,
        "avg": sum(values) / len(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "by_layer": by_layer,
    }


def summarize_layer(counter, k):
    total = sum(counter.values())
    parts = []
    for expert, count in counter.most_common(k):
        pct = 100.0 * count / total if total else 0.0
        parts.append(f"{expert}:{pct:.1f}%")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DeepSeek V4 Flash Q4 GGUF expert-cache candidates across probe prompts"
    )
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--coverage-k", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--json-out", type=Path, default=ROOT / "docs" / "deepseek-q4-cache-patterns.json")
    parser.add_argument("--hotset-out", type=Path, help="Write a prewarm hotset JSON usable by the warm script")
    args = parser.parse_args()

    if not args.gguf.exists():
        raise FileNotFoundError(args.gguf)

    prompts = load_prompts(args.prompts)
    for item in prompts:
        item["tokens"] = tokenize_prompt(args.tokenizer_bin, args.gguf, item["prompt"])

    add_gguf_import_path(args.gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415

    reader = GGUFReader(str(args.gguf), "r")
    expert_count, per_layer, routed = build_layout(reader)
    hash_layers = sorted(layer for layer, tensors in per_layer.items() if has_hash_router(tensors))
    prior_counts = bias_counts(per_layer)
    expert_bytes = expert_bytes_by_layer(routed)

    prompt_reports = []
    category_counts = defaultdict(list)
    for item in prompts:
        counts, totals = hash_counts_from_tokens(per_layer, item["tokens"])
        category_counts[item["category"]].append(counts)
        prompt_reports.append(
            {
                "id": item["id"],
                "category": item["category"],
                "token_count": len(item["tokens"]),
                "hash_layers": {
                    str(layer): {
                        "total_routes": int(totals.get(layer, 0)),
                        "top_experts": top_items(counts[layer], args.top_k),
                    }
                    for layer in hash_layers
                },
            }
        )

    global_hash_counts = merge_layer_counts(
        [hash_counts_from_tokens(per_layer, item["tokens"])[0] for item in prompts]
    )
    by_category = {
        category: merge_layer_counts(count_list)
        for category, count_list in sorted(category_counts.items())
    }
    global_top_sets = {layer: top_set(global_hash_counts[layer], args.top_k) for layer in hash_layers}

    category_summary = {}
    for category, counts in by_category.items():
        category_summary[category] = {
            str(layer): {
                "coverage_by_global_top_k": coverage(counts[layer], global_top_sets[layer]),
                "top_experts": top_items(counts[layer], args.top_k),
            }
            for layer in hash_layers
        }

    pairwise = []
    for left, right in itertools.combinations(sorted(by_category), 2):
        layer_scores = [
            jaccard(top_set(by_category[left][layer], args.top_k), top_set(by_category[right][layer], args.top_k))
            for layer in hash_layers
        ]
        pairwise.append(
            {
                "left": left,
                "right": right,
                "avg_hash_layer_jaccard": sum(layer_scores) / len(layer_scores) if layer_scores else 0.0,
                "min_hash_layer_jaccard": min(layer_scores) if layer_scores else 0.0,
            }
        )

    prewarm_counts = merge_layer_counts([global_hash_counts, prior_counts])
    hotset_payload = build_hotset_payload(
        prewarm_counts,
        routed,
        args.top_k,
        source=f"probe prompts {args.prompts.name} plus exp_probs_b prior",
    )
    coverage_curve = [
        global_hash_coverage_for_k(global_hash_counts, hash_layers, k)
        for k in args.coverage_k
    ]
    prewarm_curve = [
        prewarm_estimate_for_k(prewarm_counts, routed, k)
        for k in args.coverage_k
    ]

    output = {
        "format": "deepseek-q4-cache-patterns-v1",
        "model": str(args.gguf),
        "prompt_file": str(args.prompts),
        "prompt_count": len(prompts),
        "token_count": sum(len(item["tokens"]) for item in prompts),
        "expert_count": expert_count,
        "hash_layers": hash_layers,
        "routed_layers": sorted(routed),
        "top_k": args.top_k,
        "expert_bytes_by_layer": {str(layer): int(size) for layer, size in sorted(expert_bytes.items())},
        "global_hash_summary": {
            str(layer): {
                "coverage_by_global_top_k": coverage(global_hash_counts[layer], global_top_sets[layer]),
                "top_experts": top_items(global_hash_counts[layer], args.top_k),
            }
            for layer in hash_layers
        },
        "category_hash_summary": category_summary,
        "category_pairwise_overlap": pairwise,
        "prior_summary": {
            str(layer): {"top_experts": top_items(prior_counts[layer], args.top_k)}
            for layer in sorted(prior_counts)
        },
        "prompt_reports": prompt_reports,
        "prewarm_hotset_estimate": {
            "experts_per_layer": args.top_k,
            "total_layers": len(hotset_payload["layers"]),
            "total_experts": len(hotset_payload["experts"]),
            "total_bytes": hotset_payload["total_bytes"],
            "total_gib": hotset_payload["total_gib"],
        },
        "global_hash_coverage_curve": coverage_curve,
        "prewarm_estimates_by_top_k": prewarm_curve,
        "limitations": [
            "Hash-routing layers are exact because ffn_gate_tid2eid maps token ids directly to experts.",
            "Later routed layers use exp_probs_b as a prompt-independent prior; full prompt-specific routing needs llama.cpp runtime instrumentation.",
        ],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.hotset_out:
        args.hotset_out.parent.mkdir(parents=True, exist_ok=True)
        args.hotset_out.write_text(json.dumps(hotset_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"model: {args.gguf}")
    print(f"prompts: {len(prompts)} ({sum(len(item['tokens']) for item in prompts)} tokens)")
    print(f"hash layers with exact token routing: {','.join(map(str, hash_layers))}")
    print(f"routed layers in hotset estimate: {len(hotset_payload['layers'])}")
    print(f"top-k experts per layer: {args.top_k}")
    print(f"estimated prewarm hotset: {hotset_payload['total_gib']:.2f} GiB")
    for item, size in zip(coverage_curve, prewarm_curve):
        print(
            f"  curve top-{item['top_k']:02d}: hash avg cover {item['avg'] * 100:.1f}% "
            f"(min {item['min'] * 100:.1f}%), prewarm {size['total_gib']:.2f} GiB"
        )
    for layer in hash_layers:
        cov = coverage(global_hash_counts[layer], global_top_sets[layer])
        print(f"  hash layer {layer:02d}: global top-{args.top_k} covers {cov * 100:.1f}%")
        print(f"    {summarize_layer(global_hash_counts[layer], min(args.top_k, 8))}")
    if pairwise:
        avg_pair = sum(item["avg_hash_layer_jaccard"] for item in pairwise) / len(pairwise)
        min_pair = min(item["min_hash_layer_jaccard"] for item in pairwise)
        print(f"category top-{args.top_k} overlap: avg Jaccard {avg_pair:.2f}, worst layer-pair {min_pair:.2f}")
    print(f"wrote JSON: {args.json_out}")
    if args.hotset_out:
        print(f"wrote hotset: {args.hotset_out}")


if __name__ == "__main__":
    main()
