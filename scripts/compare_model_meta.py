#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen": ROOT / "model_meta" / "qwen",
    "deepseek": ROOT / "model_meta" / "deepseek",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def text_config(config):
    return config.get("text_config", config)


def layer_numbers(weight_map):
    layers = set()
    for name in weight_map:
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def count_contains(names, needle):
    return sum(1 for name in names if needle in name)


def main():
    for label, path in MODELS.items():
        config_path = path / "config.json"
        index_path = path / "model.safetensors.index.json"
        if not config_path.exists() or not index_path.exists():
            print(f"{label}: missing metadata in {path}")
            continue

        config = load_json(config_path)
        cfg = text_config(config)
        index = load_json(index_path)
        weight_map = index["weight_map"]
        names = sorted(weight_map)
        layers = layer_numbers(names)

        print(f"\n== {label}")
        print(f"model_type: {config.get('model_type')} / {cfg.get('model_type')}")
        print(f"architectures: {config.get('architectures')}")
        print(f"layers: {len(layers)} ({layers[0]}..{layers[-1]})")
        print(f"tensors: {len(names)}")
        print(f"shards: {len(set(weight_map.values()))}")

        keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
            "full_attention_interval",
            "rope_theta",
        ]
        for key in keys:
            if key in cfg:
                print(f"{key}: {cfg[key]}")

        print("tensor families:")
        for needle in [
            "linear_attn",
            "self_attn",
            ".attn.",
            ".attn_hc.",
            ".attn.compressor.",
            ".switch_mlp.",
            ".shared_expert.",
            ".shared_experts.",
            "ffn.gate",
        ]:
            print(f"  {needle}: {count_contains(names, needle)}")


if __name__ == "__main__":
    main()
