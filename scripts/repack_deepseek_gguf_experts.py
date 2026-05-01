#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = (
    ROOT.parent
    / "deepseek-v4-experiments"
    / "models"
    / "antirez-deepseek-v4-gguf"
    / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
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


COMPONENT_KINDS = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
]


def parse_layers(spec, max_layer):
    if spec is None or spec == "all":
        return list(range(max_layer))
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    for layer in layers:
        if layer < 0 or layer >= max_layer:
            raise ValueError(f"layer {layer} outside 0..{max_layer - 1}")
    return layers


def add_gguf_import_path(path):
    if path.exists():
        sys.path.insert(0, str(path))


def field_value(reader, key):
    field = reader.fields[key]
    value = field.contents()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_layout(reader):
    tensors = {t.name: t for t in reader.tensors}
    block_count = int(field_value(reader, "deepseek4.block_count"))
    expert_count = int(field_value(reader, "deepseek4.expert_count"))

    component_offsets = {}
    cursor = 0
    first_components = []
    for kind in COMPONENT_KINDS:
        tensor = tensors[f"blk.0.{kind}"]
        size = int(tensor.n_bytes) // expert_count
        component_offsets[kind] = cursor
        first_components.append({
            "kind": kind,
            "type": tensor.tensor_type.name,
            "shape": [int(x) for x in tensor.shape],
            "size": size,
            "offset": cursor,
        })
        cursor += size

    expert_size = cursor
    layers = []
    for layer in range(block_count):
        components = []
        for kind in COMPONENT_KINDS:
            tensor = tensors[f"blk.{layer}.{kind}"]
            stride = int(tensor.n_bytes) // expert_count
            expected_size = next(c["size"] for c in first_components if c["kind"] == kind)
            if stride != expected_size:
                raise ValueError(f"layer {layer} {kind} stride changed: {stride} != {expected_size}")
            components.append({
                "kind": kind,
                "type": tensor.tensor_type.name,
                "shape": [int(x) for x in tensor.shape],
                "source_offset": int(tensor.data_offset),
                "source_n_bytes": int(tensor.n_bytes),
                "expert_stride": stride,
                "packed_offset": component_offsets[kind],
                "packed_size": stride,
            })
        layers.append({"layer": layer, "components": components})

    return {
        "source": "gguf",
        "expert_count": expert_count,
        "block_count": block_count,
        "expert_size": expert_size,
        "layer_size": expert_size * expert_count,
        "component_layout": first_components,
        "layers": layers,
    }


def repack_layer(fd_in, output_dir, layout, layer, dry_run):
    layer_info = layout["layers"][layer]
    expert_count = layout["expert_count"]
    expert_size = layout["expert_size"]
    layer_size = layout["layer_size"]
    out_path = output_dir / f"layer_{layer:02d}.bin"

    if dry_run:
        print(f"  Layer {layer:2d}: DRY RUN OK - would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    written = 0
    try:
        for expert_idx in range(expert_count):
            base_dst = expert_idx * expert_size
            for component in layer_info["components"]:
                src = component["source_offset"] + expert_idx * component["expert_stride"]
                dst = base_dst + component["packed_offset"]
                size = component["packed_size"]
                data = os.pread(fd_in, size, src)
                if len(data) != size:
                    raise IOError(f"short read layer={layer} expert={expert_idx} {component['kind']}: {len(data)}/{size}")
                os.pwrite(fd_out, data, dst)
                written += size
    finally:
        os.close(fd_out)

    return written, time.monotonic() - t0


def verify_layer(fd_in, output_dir, layout, layer):
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
            base_dst = expert_idx * expert_size
            for component in layout["layers"][layer]["components"]:
                src = component["source_offset"] + expert_idx * component["expert_stride"]
                dst = base_dst + component["packed_offset"]
                size = component["packed_size"]
                original = os.pread(fd_in, size, src)
                packed = os.pread(fd_out, size, dst)
                if original != packed:
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
    parser = argparse.ArgumentParser(description="Repack DeepSeek V4 Flash GGUF routed experts for SSD streaming")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF, help="Path to DeepSeek V4 Flash GGUF")
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY, help="Path to llama.cpp gguf-py")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for packed layer files")
    parser.add_argument("--layers", default="all", help="Layer spec: all, 0-4, or 0,5,10")
    parser.add_argument("--dry-run", action="store_true", help="Validate offsets without writing")
    parser.add_argument("--verify-only", type=int, default=None, help="Verify an existing packed layer")
    args = parser.parse_args()

    add_gguf_import_path(args.gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415

    reader = GGUFReader(args.gguf)
    layout = build_layout(reader)
    output_dir = args.output or (args.gguf.parent / "packed_experts_deepseek")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")

    print(f"GGUF: {args.gguf}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layout['block_count']}")
    print(f"Experts/layer: {layout['expert_count']}")
    print(f"Expert size: {layout['expert_size']:,} bytes")
    print(f"Layer size: {layout['layer_size']:,} bytes")
    print(f"Total routed expert data: {layout['layer_size'] * layout['block_count'] / 1024**3:.1f} GiB")
    print("Component layout:")
    for component in layout["component_layout"]:
        print(
            f"  {component['kind']}: {component['type']} "
            f"offset={component['offset']:,} size={component['size']:,} shape={component['shape']}"
        )

    fd_in = os.open(args.gguf, os.O_RDONLY)
    try:
        if args.verify_only is not None:
            ok = verify_layer(fd_in, output_dir, layout, args.verify_only)
            return 0 if ok else 1

        layers = parse_layers(args.layers, layout["block_count"])
        total_written = 0
        t0 = time.monotonic()
        for index, layer in enumerate(layers):
            written, elapsed = repack_layer(fd_in, output_dir, layout, layer, args.dry_run)
            total_written += written
            if args.dry_run:
                continue
            speed = written / elapsed / 1024**3 if elapsed else 0.0
            avg_speed = total_written / (time.monotonic() - t0) / 1024**3
            remaining = len(layers) - index - 1
            eta = remaining * ((time.monotonic() - t0) / (index + 1))
            print(
                f"  Layer {layer:2d}: {written / 1024**3:.2f} GiB in {elapsed:.1f}s "
                f"({speed:.1f} GiB/s) | avg {avg_speed:.1f} GiB/s | ETA {eta:.0f}s"
            )
            if not verify_layer(fd_in, output_dir, layout, layer):
                return 1

        if not args.dry_run:
            print(f"\nDone: {total_written / 1024**3:.1f} GiB written to {output_dir}")
    finally:
        os.close(fd_in)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
