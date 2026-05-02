#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT.parent / "deepseek-v4-experiments"
DEFAULT_GGUF = EXP_ROOT / "models" / "lovedheart-deepseek-v4-flash-gguf" / "DeepSeek-V4-Flash-MXFP4_MOE.gguf"
DEFAULT_LLAMA_DIR = EXP_ROOT / "llama.cpp-deepseek-v4-flash"
DEFAULT_GGUF_PY = DEFAULT_LLAMA_DIR / "gguf-py"
DEFAULT_TOKENIZER = DEFAULT_LLAMA_DIR / "build" / "bin" / "llama-tokenize"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

ROUTED_COMPONENTS = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

CODING_SEED_PROMPTS = (
    "Write a Python function that parses JSON, validates fields, and returns typed errors.",
    "Fix this TypeScript async bug and explain the race condition in the request cache.",
    "Implement a Rust iterator over chunks with tests for edge cases and lifetimes.",
    "Review this SQL migration for locking, indexes, rollback safety, and performance.",
    "Generate a compact JSON object describing a unit test result for a local model runner.",
)

PLAIN_SEED_PROMPTS = (
    "Answer briefly and clearly.",
    "Summarize the main point in one short paragraph.",
    "Explain the tradeoff in practical terms.",
)

JSON_SEED_PROMPTS = (
    "Return JSON only with fields task, status, and notes.",
    "Create a valid JSON object for a smoke test result.",
)

PROFILE_PROMPTS = {
    "coding": CODING_SEED_PROMPTS,
    "plain": PLAIN_SEED_PROMPTS,
    "json": JSON_SEED_PROMPTS,
}


if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


def add_gguf_import_path(path):
    if path.exists():
        sys.path.insert(0, str(path))


def parse_size_gib(value):
    if isinstance(value, (int, float)):
        return float(value)
    if str(value).lower() == "auto":
        return "auto"
    return float(value)


def parse_mib_list(value):
    gaps = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            gaps.append(float(part))
    if not gaps:
        raise argparse.ArgumentTypeError("expected comma-separated MiB values")
    return gaps


def run_capture(args, check=True):
    result = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
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


def load_token_ids(args):
    token_ids = []
    for item in args.token_ids or []:
        if item.startswith("@"):
            text = Path(item[1:]).read_text(encoding="utf-8")
        else:
            text = item
        token_ids.extend(int(x) for x in re.findall(r"\d+", text))

    prompts = []
    if args.profile:
        prompts.extend(PROFILE_PROMPTS[args.profile])
    prompts.extend(args.prompt or [])
    for path in args.prompt_file or []:
        prompts.append(Path(path).read_text(encoding="utf-8"))

    for prompt in prompts:
        token_ids.extend(tokenize_prompt(args.tokenizer_bin, args.gguf, prompt))

    return token_ids


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


def load_hotset_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = defaultdict(Counter)
    if "layers" in data:
        for layer, experts in data["layers"].items():
            for rank, expert in enumerate(experts):
                counts[int(layer)][int(expert)] += 1_000_000 - rank
    if "experts" in data:
        for item in data["experts"]:
            counts[int(item["layer"])][int(item["expert"])] += float(item.get("score", 1.0))
    return counts


def hash_hotset_from_tokens(per_layer, token_ids):
    counts = defaultdict(Counter)
    if not token_ids:
        return counts
    for layer, tensors in per_layer.items():
        tensor = tensors.get("ffn_gate_tid2eid.weight") or tensors.get("ffn_gate_tid2eid")
        if tensor is None:
            continue
        table = tensor.data
        vocab = table.shape[0]
        for token_id in token_ids:
            if 0 <= token_id < vocab:
                for expert in table[token_id]:
                    counts[layer][int(expert)] += 1
    return counts


def bias_hotset(per_layer):
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


def merge_counts(*sources):
    merged = defaultdict(Counter)
    for source in sources:
        for layer, counter in source.items():
            merged[layer].update(counter)
    return merged


def selected_layers(spec, routed_layers, hash_layers):
    if spec == "all":
        return sorted(routed_layers)
    if spec == "hash":
        return sorted(set(routed_layers) & set(hash_layers))
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layer for layer in layers if layer in routed_layers)


def auto_budget_bytes(leave_gib, max_auto_gib):
    total = int(run_capture(["sysctl", "-n", "hw.memsize"]).strip())
    vm = run_capture(["vm_stat"])
    page_size_match = re.search(r"page size of (\d+) bytes", vm)
    page_size = int(page_size_match.group(1)) if page_size_match else 16384

    def pages(label):
        match = re.search(rf"{re.escape(label)}:\s+(\d+)\.", vm)
        return int(match.group(1)) if match else 0

    non_reclaimable = (
        pages("Pages wired down")
        + pages("Anonymous pages")
        + pages("Pages occupied by compressor")
        + pages("Pages throttled")
    ) * page_size
    budget = total - non_reclaimable - int(leave_gib * 1024**3)
    budget = max(0, budget)
    return min(budget, int(max_auto_gib * 1024**3))


def expert_spans(routed, layer, expert):
    spans = []
    for component in routed[layer]:
        if expert * component["stride"] >= component["n_bytes"]:
            raise ValueError(f"expert {expert} is out of bounds for layer {layer} {component['kind']}")
        spans.append(
            {
                "layer": layer,
                "expert": expert,
                "kind": component["kind"],
                "offset": component["offset"] + expert * component["stride"],
                "n_bytes": component["stride"],
            }
        )
    return spans


def build_plan(routed, layers, counts, experts_per_layer, budget_bytes):
    ranked = []
    for layer in layers:
        if layer not in counts:
            continue
        experts = [expert for expert, _ in counts[layer].most_common(experts_per_layer)]
        for rank, expert in enumerate(experts):
            score = counts[layer][expert]
            ranked.append((rank, layer, -float(score), expert))

    plan = []
    total = 0
    for _, layer, neg_score, expert in sorted(ranked):
        spans = expert_spans(routed, layer, expert)
        n_bytes = sum(span["n_bytes"] for span in spans)
        if total + n_bytes > budget_bytes:
            continue
        plan.append(
            {
                "layer": layer,
                "expert": expert,
                "score": -neg_score,
                "bytes": n_bytes,
                "spans": spans,
            }
        )
        total += n_bytes
    return plan, total


def iter_plan_spans(plan):
    for item in plan:
        for span in item["spans"]:
            start = int(span["offset"])
            end = start + int(span["n_bytes"])
            yield start, end


def merge_ranges(ranges, max_gap_bytes):
    merged = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + max_gap_bytes:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def range_bytes(ranges):
    return sum(end - start for start, end in ranges)


def merged_read_plan(plan, merge_gap_bytes):
    raw_ranges = list(iter_plan_spans(plan))
    unique_ranges = merge_ranges(raw_ranges, 0)
    merged = merge_ranges(raw_ranges, merge_gap_bytes)
    raw_bytes = range_bytes(raw_ranges)
    unique_bytes = range_bytes(unique_ranges)
    merged_bytes = range_bytes(merged)
    return {
        "raw_ranges": raw_ranges,
        "unique_ranges": unique_ranges,
        "merged_ranges": merged,
        "raw_bytes": raw_bytes,
        "unique_bytes": unique_bytes,
        "merged_bytes": merged_bytes,
        "overread_bytes": max(0, merged_bytes - unique_bytes),
    }


def print_merge_summary(stats, merge_gap_bytes):
    unique_bytes = max(stats["unique_bytes"], 1)
    overread_ratio = stats["overread_bytes"] / unique_bytes
    print(
        "pread ranges: "
        f"raw={len(stats['raw_ranges'])}, "
        f"unique={len(stats['unique_ranges'])}, "
        f"merged={len(stats['merged_ranges'])}, "
        f"merge_gap={merge_gap_bytes / 1024**2:.2f} MiB"
    )
    print(
        "pread bytes: "
        f"planned={stats['raw_bytes'] / 1024**3:.2f} GiB, "
        f"unique={stats['unique_bytes'] / 1024**3:.2f} GiB, "
        f"actual={stats['merged_bytes'] / 1024**3:.2f} GiB, "
        f"overread={stats['overread_bytes'] / 1024**3:.2f} GiB ({overread_ratio:.1%})"
    )


def print_merge_curve(plan, gaps_mib):
    print("merge curve:")
    for gap_mib in gaps_mib:
        stats = merged_read_plan(plan, int(gap_mib * 1024**2))
        unique_bytes = max(stats["unique_bytes"], 1)
        overread_ratio = stats["overread_bytes"] / unique_bytes
        print(
            f"  gap={gap_mib:>6.2f} MiB "
            f"ranges={len(stats['merged_ranges']):>5} "
            f"actual={stats['merged_bytes'] / 1024**3:>7.2f} GiB "
            f"overread={stats['overread_bytes'] / 1024**3:>7.2f} GiB "
            f"({overread_ratio:>6.1%})"
        )


def warm_spans(model_path, plan, chunk_bytes, progress_bytes, merge_gap_bytes):
    stats = merged_read_plan(plan, merge_gap_bytes)
    ranges = stats["merged_ranges"]
    fd = os.open(model_path, os.O_RDONLY)
    total = stats["merged_bytes"]
    done = 0
    next_report = progress_bytes
    started = time.time()
    try:
        for start, end in ranges:
            remaining = end - start
            offset = start
            while remaining > 0:
                n = min(chunk_bytes, remaining)
                data = os.pread(fd, n, offset)
                if not data:
                    raise IOError(f"short read at offset {offset}")
                got = len(data)
                remaining -= got
                offset += got
                done += got
                if done >= next_report:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"warmed {done / 1024**3:.2f}/{total / 1024**3:.2f} GiB "
                        f"({done / elapsed / 1024**3:.2f} GiB/s)",
                        flush=True,
                    )
                    next_report += progress_bytes
    finally:
        os.close(fd)
    elapsed = max(time.time() - started, 1e-6)
    return elapsed, done


def summarize_plan(plan, total, layers):
    by_layer = defaultdict(list)
    for item in plan:
        by_layer[item["layer"]].append(item["expert"])
    print(f"layers planned: {len(by_layer)}/{len(layers)}")
    print(f"experts planned: {len(plan)}")
    print(f"bytes planned: {total / 1024**3:.2f} GiB")
    for layer in sorted(by_layer)[:8]:
        print(f"  layer {layer:02d}: {','.join(map(str, by_layer[layer][:12]))}")
    if len(by_layer) > 8:
        print(f"  ... {len(by_layer) - 8} more layers")


def write_hotset(path, plan):
    layers = defaultdict(list)
    experts = []
    for item in plan:
        layers[str(item["layer"])].append(item["expert"])
        experts.append(
            {
                "layer": item["layer"],
                "expert": item["expert"],
                "score": item["score"],
                "bytes": item["bytes"],
            }
        )
    payload = {
        "format": "deepseek-q4-hotset-v1",
        "layers": dict(sorted(layers.items(), key=lambda kv: int(kv[0]))),
        "experts": experts,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote hotset: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preload selected DeepSeek V4 Flash Q4 GGUF routed experts into the macOS page cache"
    )
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--profile", choices=sorted(PROFILE_PROMPTS), default=None)
    parser.add_argument("--prompt", action="append", help="Additional prompt text used for hash-layer routing")
    parser.add_argument("--prompt-file", action="append", help="Prompt file used for hash-layer routing")
    parser.add_argument("--token-ids", action="append", help="Comma/space separated token ids, or @file")
    parser.add_argument("--hotset-json", type=Path, help="Explicit hotset JSON produced by this script or a profiler")
    parser.add_argument("--write-hotset-json", type=Path, help="Write the planned hotset without warming")
    parser.add_argument("--layers", default="all", help="'all', 'hash', or a comma/range list such as 0-5,10")
    parser.add_argument("--experts-per-layer", type=int, default=16)
    parser.add_argument("--budget-gib", type=parse_size_gib, default="auto")
    parser.add_argument("--leave-gib", type=float, default=13.0)
    parser.add_argument("--max-auto-gib", type=float, default=8.0)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument("--progress-mib", type=int, default=512)
    parser.add_argument("--merge-gap-mib", type=float, default=0.0)
    parser.add_argument(
        "--merge-curve-gaps-mib",
        type=parse_mib_list,
        default=None,
        help="Print range-count/overread tradeoffs for comma-separated merge gaps",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.gguf.exists():
        raise FileNotFoundError(args.gguf)

    add_gguf_import_path(args.gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415

    reader = GGUFReader(str(args.gguf), "r")
    expert_count, per_layer, routed = build_layout(reader)
    hash_layers = [layer for layer, tensors in per_layer.items() if has_hash_router(tensors)]
    layers = selected_layers(args.layers, routed.keys(), hash_layers)
    if not layers:
        raise RuntimeError(f"no routed expert layers selected by --layers={args.layers}")

    token_ids = load_token_ids(args)
    source_counts = []
    if args.hotset_json:
        source_counts.append(load_hotset_json(args.hotset_json))
    source_counts.append(hash_hotset_from_tokens(per_layer, token_ids))
    source_counts.append(bias_hotset(per_layer))
    counts = merge_counts(*source_counts)

    if args.budget_gib == "auto":
        budget_bytes = auto_budget_bytes(args.leave_gib, args.max_auto_gib)
    else:
        budget_bytes = int(args.budget_gib * 1024**3)

    plan, total = build_plan(routed, layers, counts, args.experts_per_layer, budget_bytes)
    print(f"model: {args.gguf}")
    print(f"expert_count: {expert_count}")
    print(f"token_ids used for hash layers: {len(token_ids)}")
    print(f"budget: {budget_bytes / 1024**3:.2f} GiB (leave_gib={args.leave_gib:.1f})")
    summarize_plan(plan, total, layers)
    merge_gap_bytes = int(args.merge_gap_mib * 1024**2)
    merge_stats = merged_read_plan(plan, merge_gap_bytes)
    print_merge_summary(merge_stats, merge_gap_bytes)
    if args.merge_curve_gaps_mib:
        print_merge_curve(plan, args.merge_curve_gaps_mib)

    if args.write_hotset_json:
        write_hotset(args.write_hotset_json, plan)

    if args.dry_run:
        return

    if not plan:
        raise RuntimeError("preload plan is empty")

    elapsed, warmed = warm_spans(
        str(args.gguf),
        plan,
        args.chunk_mib * 1024 * 1024,
        args.progress_mib * 1024 * 1024,
        merge_gap_bytes,
    )
    print(f"done: warmed {warmed / 1024**3:.2f} GiB in {elapsed:.2f}s ({warmed / elapsed / 1024**3:.2f} GiB/s)")


if __name__ == "__main__":
    main()
