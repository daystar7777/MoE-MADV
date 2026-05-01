#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np  # noqa: E402


INSPECT_PATH = ROOT / "scripts" / "inspect_deepseek_q4_layout.py"
spec = importlib.util.spec_from_file_location("inspect_deepseek_q4_layout", INSPECT_PATH)
inspect_layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_layout)


DEFAULT_OUTPUT = ROOT / "models" / "deepseek-v4-flash-4bit" / "packed_experts_q4"


def open_required_files(layout, model_dir):
    fds = {}
    missing = []
    for filename in layout["files"]:
        path = model_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        fds[filename] = os.open(path, os.O_RDONLY)
    if missing:
        raise FileNotFoundError(
            "missing safetensors shards:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun scripts/download_model_assets.sh full deepseek, or download the needed shard first."
        )
    return fds


def close_files(fds):
    for fd in fds.values():
        os.close(fd)


def read_exact(fd, size, offset, label):
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise IOError(f"short read for {label}: {len(data):,}/{size:,}")
    return data


def pack_component(component, expert_idx, fds):
    weight = component["weight"]
    scales = component["scales"]
    weight_stride = component["weight_expert_stride"]
    scale_stride = component["scale_expert_stride"]
    out_dim = component["out_dim"]
    groups = component["groups"]
    packed_size = component["packed_size"]

    weight_data = read_exact(
        fds[weight["file"]],
        weight_stride,
        weight["data_offset"] + expert_idx * weight_stride,
        f"{weight['name']} expert {expert_idx}",
    )
    scale_data = read_exact(
        fds[scales["file"]],
        scale_stride,
        scales["data_offset"] + expert_idx * scale_stride,
        f"{scales['name']} expert {expert_idx}",
    )

    weights = np.frombuffer(weight_data, dtype=np.uint8).reshape(out_dim, groups, 16)
    scale_values = np.frombuffer(scale_data, dtype=np.uint8).reshape(out_dim, groups)
    packed = np.empty((out_dim, groups, 17), dtype=np.uint8)
    packed[:, :, 0] = scale_values
    packed[:, :, 1:] = weights
    return packed.reshape(packed_size).tobytes()


def repack_layer(layout, layer_record, output_dir, fds, dry_run):
    layer = layer_record["layer"]
    expert_count = layout["expert_count"]
    expert_size = layout["expert_size"]
    layer_size = layout["layer_size"]
    out_path = output_dir / f"layer_{layer:02d}.bin"

    if dry_run:
        files = sorted({
            component["weight"]["file"]
            for component in layer_record["components"]
        } | {
            component["scales"]["file"]
            for component in layer_record["components"]
        })
        print(f"  Layer {layer:2d}: DRY RUN OK - would read {', '.join(files)}")
        print(f"             would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)
    written = 0
    try:
        for expert_idx in range(expert_count):
            expert_base = expert_idx * expert_size
            for component in layer_record["components"]:
                data = pack_component(component, expert_idx, fds)
                dst = expert_base + component["packed_offset"]
                os.pwrite(fd_out, data, dst)
                written += len(data)
    finally:
        os.close(fd_out)

    return written, time.monotonic() - t0


def verify_layer(layout, layer_record, output_dir, fds):
    layer = layer_record["layer"]
    out_path = output_dir / f"layer_{layer:02d}.bin"
    if not out_path.exists():
        print(f"  Layer {layer}: packed file not found")
        return False

    fd_out = os.open(out_path, os.O_RDONLY)
    expert_count = layout["expert_count"]
    expert_size = layout["expert_size"]
    probes = sorted(set([0, 1, expert_count // 2, expert_count - 1]))
    mismatches = 0
    try:
        for expert_idx in probes:
            expert_base = expert_idx * expert_size
            for component in layer_record["components"]:
                expected = pack_component(component, expert_idx, fds)
                actual = os.pread(
                    fd_out,
                    component["packed_size"],
                    expert_base + component["packed_offset"],
                )
                if expected != actual:
                    print(f"  MISMATCH layer={layer} expert={expert_idx} {component['kind']}")
                    mismatches += 1
    finally:
        os.close(fd_out)

    if mismatches:
        print(f"  Layer {layer}: verification FAILED ({mismatches} mismatches)")
        return False
    print(f"  Layer {layer}: verification PASSED (experts {', '.join(map(str, probes))})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Repack DeepSeek V4 Flash MLX Q4 routed experts for SSD streaming")
    parser.add_argument("--repo", default=inspect_layout.DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=inspect_layout.DEFAULT_MODEL_DIR)
    parser.add_argument("--index", type=Path, default=inspect_layout.DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="all", help="Layer spec: all, 0-4, or 0,5,10")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", type=int, default=None)
    args = parser.parse_args()

    layers_spec = str(args.verify_only) if args.verify_only is not None else args.layers
    layout = inspect_layout.build_layout(args.repo, args.model_dir, args.index, layers_spec)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")

    print(f"Model dir: {args.model_dir}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layout['layers_included']}")
    print(f"Experts/layer: {layout['expert_count']}")
    print(f"Expert size: {layout['expert_size']:,} bytes ({layout['expert_size'] / 1024**2:.2f} MiB)")
    print(f"Layer size: {layout['layer_size']:,} bytes ({layout['layer_size'] / 1024**3:.2f} GiB)")
    print("Packed component layout:")
    for component in layout["packed_component_layout"]:
        print(
            f"  {component['kind']}: offset={component['offset']:,} "
            f"size={component['size']:,} shape=[{component['out_dim']}, {component['in_dim']}]"
        )

    fds = {} if args.dry_run else open_required_files(layout, args.model_dir)
    try:
        if args.verify_only is not None:
            ok = verify_layer(layout, layout["layers"][0], output_dir, fds)
            return 0 if ok else 1

        total_written = 0
        t0 = time.monotonic()
        for index, layer_record in enumerate(layout["layers"]):
            written, elapsed = repack_layer(layout, layer_record, output_dir, fds, args.dry_run)
            total_written += written
            if args.dry_run:
                continue
            speed = written / elapsed / 1024**3 if elapsed else 0.0
            avg_speed = total_written / (time.monotonic() - t0) / 1024**3
            remaining = len(layout["layers"]) - index - 1
            eta = remaining * ((time.monotonic() - t0) / (index + 1))
            print(
                f"  Layer {layer_record['layer']:2d}: {written / 1024**3:.2f} GiB in {elapsed:.1f}s "
                f"({speed:.2f} GiB/s) | avg {avg_speed:.2f} GiB/s | ETA {eta:.0f}s"
            )
            if not verify_layer(layout, layer_record, output_dir, fds):
                return 1

        if not args.dry_run:
            print(f"\nDone: {total_written / 1024**3:.1f} GiB written to {output_dir}")
    finally:
        close_files(fds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
