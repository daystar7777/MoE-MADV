#!/usr/bin/env python3
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "deepseek-v4-flash-4bit"
DEFAULT_PACKED_DIR = DEFAULT_MODEL_DIR / "packed_experts_q4"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np  # noqa: E402


HIDDEN = 4096
MIN_ROUTER_DENOM = 6.103515625e-5


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safetensors_header(path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


def tensor_bytes(model_dir, index, name):
    try:
        filename = index["weight_map"][name]
    except KeyError as exc:
        raise KeyError(f"{name} not found in index") from exc

    path = model_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; download it first, for example:\n"
            f"  scripts/download_model_assets.sh shard deepseek {filename}"
        )

    data_start, header = safetensors_header(path)
    meta = header[name]
    start, end = [int(x) for x in meta["data_offsets"]]
    fd = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(fd, end - start, data_start + start)
    finally:
        os.close(fd)

    if len(data) != end - start:
        raise IOError(f"short read for {name}: {len(data):,}/{end - start:,}")
    return meta, data


def bf16_to_f32(data, shape):
    raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
    return (raw << 16).view(np.float32).reshape(shape)


def make_probe_input():
    idx = np.arange(HIDDEN, dtype=np.float32)
    return np.sin(idx * np.float32(0.013)) + np.float32(0.5) * np.cos(idx * np.float32(0.021))


def sqrt_softplus(logits):
    # logaddexp keeps softplus stable for large positive or negative router logits.
    return np.sqrt(np.logaddexp(logits.astype(np.float32), np.float32(0.0))).astype(np.float32)


def format_csv(values, precision=9):
    return ",".join(f"{float(v):.{precision}g}" for v in values)


def format_int_csv(values):
    return ",".join(str(int(v)) for v in values)


def main():
    parser = argparse.ArgumentParser(
        description="Select DeepSeek V4 Flash Q4 hash-routed experts and weights for the Metal probe"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--run-probe", action="store_true", help="run scripts/run_deepseek_q4_probe.sh with the selected route")
    args = parser.parse_args()

    index_path = args.index or (args.model_dir / "model.safetensors.index.json")
    config_path = args.config or (args.model_dir / "config.json")
    index = load_json(index_path)
    config = load_json(config_path)

    num_layers = int(config["num_hidden_layers"])
    num_hash_layers = int(config.get("num_hash_layers", 0))
    experts_per_token = int(config["num_experts_per_tok"])
    scaling = float(config.get("routed_scaling_factor", 1.0))
    scoring_func = config.get("scoring_func", "sqrtsoftplus")

    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"--layer must be in 0..{num_layers - 1}")
    if args.layer >= num_hash_layers:
        raise NotImplementedError(
            f"layer {args.layer} is not a hash-routed layer. This probe currently supports "
            f"layers 0..{num_hash_layers - 1}, which have ffn.gate.tid2eid tables."
        )

    tid_name = f"model.layers.{args.layer}.ffn.gate.tid2eid"
    gate_name = f"model.layers.{args.layer}.ffn.gate.weight"
    tid_meta, tid_data = tensor_bytes(args.model_dir, index, tid_name)
    gate_meta, gate_data = tensor_bytes(args.model_dir, index, gate_name)

    if tid_meta["dtype"] != "I64" or len(tid_meta["shape"]) != 2:
        raise ValueError(f"{tid_name}: expected I64 rank-2, got {tid_meta['dtype']} {tid_meta['shape']}")
    vocab, k = [int(x) for x in tid_meta["shape"]]
    if k != experts_per_token:
        raise ValueError(f"{tid_name}: expected K={experts_per_token}, got {k}")
    if args.token_id < 0 or args.token_id >= vocab:
        raise ValueError(f"--token-id must be in 0..{vocab - 1}")

    if gate_meta["dtype"] != "BF16" or [int(x) for x in gate_meta["shape"]] != [256, HIDDEN]:
        raise ValueError(f"{gate_name}: expected BF16 [256, {HIDDEN}], got {gate_meta['dtype']} {gate_meta['shape']}")
    if scoring_func != "sqrtsoftplus":
        raise NotImplementedError(f"unsupported scoring_func={scoring_func!r}")

    selected_table = np.frombuffer(tid_data, dtype="<i8").reshape(vocab, k)
    selected = selected_table[args.token_id].astype(np.int32)
    gate_weight = bf16_to_f32(gate_data, (256, HIDDEN))
    x = make_probe_input()
    logits = gate_weight @ x
    probs = sqrt_softplus(logits)
    raw_weights = probs[selected]
    denom = max(float(raw_weights.sum()), MIN_ROUTER_DENOM)
    weights = (raw_weights / np.float32(denom) * np.float32(scaling)).astype(np.float32)

    expert_csv = format_int_csv(selected)
    weight_csv = format_csv(weights)
    command = [
        "scripts/run_deepseek_q4_probe.sh",
        "--layer",
        str(args.layer),
        "--experts",
        expert_csv,
        "--weights",
        weight_csv,
    ]

    print(f"Model dir: {args.model_dir}")
    print(f"Layer/token: {args.layer}/{args.token_id}")
    print(f"Hash routing table: {tid_name} shape={tid_meta['shape']}")
    print(f"Router gate: {gate_name} shape={gate_meta['shape']}")
    print(f"Selected experts: {expert_csv}")
    print(f"Raw sqrtsoftplus weights: {format_csv(raw_weights)}")
    print(f"Normalized scaled weights: {weight_csv} (sum={float(weights.sum()):.9g})")
    print("Probe command:")
    print(" ".join(command))

    if args.run_probe:
        print("\nRunning probe...")
        sys.stdout.flush()
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
