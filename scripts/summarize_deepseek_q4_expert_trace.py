#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "logs" / "deepseek_q4_expert_trace.jsonl"
DEFAULT_CACHE_PATTERNS = ROOT / "docs" / "deepseek-q4-cache-patterns.json"


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def load_expert_bytes(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(layer): int(size) for layer, size in data.get("expert_bytes_by_layer", {}).items()}


def top(counter, k):
    return [{"expert": int(expert), "count": int(count)} for expert, count in counter.most_common(k)]


def jaccard(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def event_phase(event):
    return "prefill" if int(event.get("n_tokens", 0)) > 1 else "decode"


def summarize_events(events, expert_bytes, top_k):
    phase_layer_counts = defaultdict(Counter)
    phase_layer_unique = defaultdict(set)
    phase_layer_events = Counter()
    phase_routes = Counter()

    for event in events:
        if event.get("kind") not in ("gate", "hash"):
            continue
        phase = event_phase(event)
        layer = int(event["layer"])
        key = (phase, layer, event.get("kind", "gate"))
        phase_layer_events[key] += 1
        for token_experts in event.get("experts", []):
            phase_routes[phase] += len(token_experts)
            for expert in token_experts:
                phase_layer_counts[key][int(expert)] += 1
                phase_layer_unique[key].add(int(expert))

    layers = {}
    for key in sorted(phase_layer_counts):
        phase, layer, kind = key
        unique_count = len(phase_layer_unique[key])
        bytes_est = unique_count * expert_bytes.get(layer, 0)
        layers.setdefault(phase, {}).setdefault(kind, {})[str(layer)] = {
            "events": int(phase_layer_events[key]),
            "routes": int(sum(phase_layer_counts[key].values())),
            "unique_experts": unique_count,
            "unique_expert_bytes_est": bytes_est,
            "unique_expert_gib_est": bytes_est / 1024**3,
            "top_experts": top(phase_layer_counts[key], top_k),
        }

    return {
        "phase_routes": dict(phase_routes),
        "layers": layers,
    }


def split_gate_rounds(events):
    rounds = []
    current = []
    last_layer = -1
    for event in events:
        if event.get("kind") != "gate":
            continue
        layer = int(event["layer"])
        if current and layer <= last_layer:
            rounds.append(current)
            current = []
        current.append(event)
        last_layer = layer
    if current:
        rounds.append(current)
    return rounds


def summarize_rounds(rounds, expert_bytes):
    result = []
    for idx, round_events in enumerate(rounds):
        phase = event_phase(round_events[0]) if round_events else "unknown"
        per_layer = {}
        total_bytes = 0
        for event in round_events:
            layer = int(event["layer"])
            experts = set()
            for token_experts in event.get("experts", []):
                experts.update(int(expert) for expert in token_experts)
            bytes_est = len(experts) * expert_bytes.get(layer, 0)
            total_bytes += bytes_est
            per_layer[str(layer)] = {
                "unique_experts": len(experts),
                "experts": sorted(experts),
                "bytes_est": bytes_est,
            }
        result.append(
            {
                "index": idx,
                "phase": phase,
                "layers": len(per_layer),
                "unique_expert_bytes_est": total_bytes,
                "unique_expert_gib_est": total_bytes / 1024**3,
                "per_layer": per_layer,
            }
        )
    return result


def summarize_decode_overlap(round_summaries):
    decode = [item for item in round_summaries if item["phase"] == "decode"]
    if len(decode) < 2:
        return {}

    scores = []
    for prev, cur in zip(decode, decode[1:]):
        layer_scores = []
        for layer, cur_info in cur["per_layer"].items():
            prev_info = prev["per_layer"].get(layer)
            if not prev_info:
                continue
            layer_scores.append(jaccard(set(prev_info["experts"]), set(cur_info["experts"])))
        if layer_scores:
            scores.append(sum(layer_scores) / len(layer_scores))

    return {
        "decode_rounds": len(decode),
        "avg_adjacent_round_jaccard": sum(scores) / len(scores) if scores else 0.0,
        "min_adjacent_round_jaccard": min(scores) if scores else 0.0,
        "max_adjacent_round_jaccard": max(scores) if scores else 0.0,
    }


def phase_durations(events):
    spans = {}
    for event in events:
        if event.get("kind") != "gate":
            continue
        phase = event_phase(event)
        t_s = float(event.get("t_us", 0)) / 1e6
        spans.setdefault(phase, [t_s, t_s])
        spans[phase][0] = min(spans[phase][0], t_s)
        spans[phase][1] = max(spans[phase][1], t_s)
    return {
        phase: {
            "start_s": span[0],
            "end_s": span[1],
            "duration_s": max(0.0, span[1] - span[0]),
        }
        for phase, span in spans.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize DeepSeek Q4 MoE expert trace JSONL")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--cache-patterns", type=Path, default=DEFAULT_CACHE_PATTERNS)
    parser.add_argument("--json-out", type=Path, default=ROOT / "logs" / "deepseek_q4_expert_trace_summary.json")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    events = load_jsonl(args.trace)
    expert_bytes = load_expert_bytes(args.cache_patterns)
    rounds = summarize_rounds(split_gate_rounds(events), expert_bytes)
    phase_summary = summarize_events(events, expert_bytes, args.top_k)
    decode_overlap = summarize_decode_overlap(rounds)
    durations = phase_durations(events)

    payload = {
        "format": "deepseek-q4-expert-trace-summary-v1",
        "trace": str(args.trace),
        "events": len(events),
        "rounds": len(rounds),
        "phase_summary": phase_summary,
        "round_summaries": rounds,
        "decode_overlap": decode_overlap,
        "phase_durations": durations,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    prefill = [item for item in rounds if item["phase"] == "prefill"]
    decode = [item for item in rounds if item["phase"] == "decode"]
    print(f"trace: {args.trace}")
    print(f"events: {len(events)}")
    print(f"rounds: {len(rounds)} (prefill={len(prefill)}, decode={len(decode)})")
    if prefill:
        print(f"prefill round expert bytes: {prefill[0]['unique_expert_gib_est']:.2f} GiB")
        if "prefill" in durations:
            print(f"prefill trace duration: {durations['prefill']['duration_s']:.2f}s")
    if decode:
        avg_decode = sum(item["unique_expert_gib_est"] for item in decode) / len(decode)
        print(f"decode avg expert bytes/round: {avg_decode:.2f} GiB")
        if "decode" in durations:
            print(f"decode trace duration: {durations['decode']['duration_s']:.2f}s")
    if decode_overlap:
        print(
            "decode adjacent overlap: "
            f"avg Jaccard {decode_overlap['avg_adjacent_round_jaccard']:.2f}, "
            f"min {decode_overlap['min_adjacent_round_jaccard']:.2f}"
        )
    print(f"wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
