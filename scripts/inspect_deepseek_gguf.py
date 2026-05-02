#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = (
    ROOT.parent
    / "deepseek-v4-experiments"
    / "models"
    / "lovedheart-deepseek-v4-flash-gguf"
    / "DeepSeek-V4-Flash-MXFP4_MOE.gguf"
)
DEFAULT_GGUF_PY = (
    ROOT.parent
    / "deepseek-v4-experiments"
    / "llama.cpp-deepseek-v4-flash"
    / "gguf-py"
)


VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


def add_gguf_import_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def field_value(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    value = field.contents()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def tensor_record(tensor):
    return {
        "name": tensor.name,
        "type": tensor.tensor_type.name,
        "shape": [int(x) for x in tensor.shape],
        "n_elements": int(tensor.n_elements),
        "n_bytes": int(tensor.n_bytes),
        "data_offset": int(tensor.data_offset),
    }


def layer_and_kind(name):
    match = re.match(r"blk\.(\d+)\.(.+)", name)
    if not match:
        return None, name
    return int(match.group(1)), match.group(2)


def main():
    parser = argparse.ArgumentParser(description="Inspect DeepSeek V4 Flash GGUF layout for flash-moe porting")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF, help="Path to DeepSeek V4 Flash GGUF")
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY, help="Path to llama.cpp gguf-py")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional layout JSON output path")
    args = parser.parse_args()

    add_gguf_import_path(args.gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415

    reader = GGUFReader(args.gguf)
    tensors = [tensor_record(t) for t in reader.tensors]
    by_name = {t["name"]: t for t in tensors}
    family_counts = Counter()
    per_layer = defaultdict(dict)

    for t in tensors:
        layer, kind = layer_and_kind(t["name"])
        family = kind.split(".")[0] if layer is not None else t["name"].split(".")[0]
        if layer is not None:
            if kind.startswith("ffn_"):
                family = kind.rsplit(".", 1)[0]
            elif kind.startswith("attn_"):
                family = kind.rsplit(".", 1)[0]
            elif kind.startswith("attn."):
                family = "attn." + kind.split(".")[1]
            per_layer[layer][kind] = t
        family_counts[family] += 1

    expert_count = int(field_value(reader, "deepseek4.expert_count"))
    routed_names = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
    routed_layout = []
    for layer in sorted(per_layer):
        layer_record = {"layer": layer, "components": [], "bytes_per_expert": 0}
        complete = True
        for kind in routed_names:
            tensor = per_layer[layer].get(kind)
            if tensor is None:
                complete = False
                continue
            stride = tensor["n_bytes"] // expert_count
            layer_record["bytes_per_expert"] += stride
            layer_record["components"].append({
                "kind": kind,
                "type": tensor["type"],
                "shape": tensor["shape"],
                "data_offset": tensor["data_offset"],
                "n_bytes": tensor["n_bytes"],
                "expert_stride": stride,
            })
        if complete:
            routed_layout.append(layer_record)

    summary = {
        "gguf": args.gguf.name,
        "data_offset": int(reader.data_offset),
        "alignment": int(reader.alignment),
        "tensor_count": len(tensors),
        "metadata": {
            "architecture": field_value(reader, "general.architecture"),
            "name": field_value(reader, "general.name"),
            "block_count": field_value(reader, "deepseek4.block_count"),
            "context_length": field_value(reader, "deepseek4.context_length"),
            "embedding_length": field_value(reader, "deepseek4.embedding_length"),
            "attention_heads": field_value(reader, "deepseek4.attention.head_count"),
            "attention_kv_heads": field_value(reader, "deepseek4.attention.head_count_kv"),
            "head_key_length": field_value(reader, "deepseek4.attention.key_length"),
            "head_value_length": field_value(reader, "deepseek4.attention.value_length"),
            "expert_count": expert_count,
            "expert_used_count": field_value(reader, "deepseek4.expert_used_count"),
            "expert_feed_forward_length": field_value(reader, "deepseek4.expert_feed_forward_length"),
            "expert_shared_count": field_value(reader, "deepseek4.expert_shared_count"),
            "compress_ratios": field_value(reader, "deepseek4.attention.compress_ratios"),
            "sliding_window": field_value(reader, "deepseek4.attention.sliding_window"),
        },
        "family_counts": dict(sorted(family_counts.items())),
        "routed_expert_layout": routed_layout,
        "tensors": tensors,
    }

    print(f"GGUF: {args.gguf}")
    print(f"Tensors: {summary['tensor_count']}, data_offset={summary['data_offset']}, alignment={summary['alignment']}")
    for key, value in summary["metadata"].items():
        if key == "compress_ratios":
            if value is None:
                print(f"{key}: None")
            else:
                print(f"{key}: {value[:10]} ... ({len(value)} entries)")
        else:
            print(f"{key}: {value}")

    print("\nRouted expert components:")
    if routed_layout:
        first = routed_layout[0]
        print(f"layers: {len(routed_layout)}")
        print(f"bytes_per_expert: {first['bytes_per_expert']:,}")
        for component in first["components"]:
            print(
                f"  {component['kind']}: {component['type']} "
                f"shape={component['shape']} stride={component['expert_stride']:,} "
                f"offset={component['data_offset']:,}"
            )

    print("\nTop tensor families:")
    for family, count in family_counts.most_common(20):
        print(f"  {family}: {count}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
