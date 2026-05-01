#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "mlx-community/DeepSeek-V4-Flash-4bit"
DEFAULT_MODEL_DIR = ROOT / "models" / "deepseek-v4-flash-4bit"
DEFAULT_INDEX = ROOT / "model_meta" / "deepseek" / "model.safetensors.index.json"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

from huggingface_hub import HfApi, hf_hub_url  # noqa: E402


PROJECTIONS = [
    ("gate", "gate_proj"),
    ("up", "up_proj"),
    ("down", "down_proj"),
]
DTYPE_SIZES = {
    "U8": 1,
    "U32": 4,
}
MXFP4_BLOCK_BYTES = 17  # one E8M0 scale byte + 16 packed 4-bit bytes for 32 values


def portable_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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


def read_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def read_safetensors_header(path, repo, filename):
    local = path / filename
    if local.exists():
        with local.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        return {
            "filename": filename,
            "source": "local",
            "header_len": header_len,
            "data_start": 8 + header_len,
            "tensors": header,
        }

    url = hf_hub_url(repo, filename)
    header_len = struct.unpack("<Q", read_range(url, 0, 7))[0]
    header = json.loads(read_range(url, 8, 7 + header_len))
    return {
        "filename": filename,
        "source": "remote",
        "header_len": header_len,
        "data_start": 8 + header_len,
        "tensors": header,
    }


def tensor_nbytes(meta):
    n = DTYPE_SIZES[meta["dtype"]]
    for dim in meta["shape"]:
        n *= int(dim)
    return n


def tensor_record(name, weight_map, headers):
    filename = weight_map[name]
    header = headers[filename]
    meta = header["tensors"][name]
    start, end = [int(x) for x in meta["data_offsets"]]
    nbytes = tensor_nbytes(meta)
    if end - start != nbytes:
        raise ValueError(f"{name}: header size {end - start} != computed {nbytes}")
    return {
        "name": name,
        "file": filename,
        "dtype": meta["dtype"],
        "shape": [int(x) for x in meta["shape"]],
        "data_offset": int(header["data_start"] + start),
        "n_bytes": nbytes,
    }


def component_record(layer, short_name, proj_name, weight_map, headers):
    base = f"model.layers.{layer}.ffn.switch_mlp.{proj_name}"
    weight = tensor_record(f"{base}.weight", weight_map, headers)
    scales = tensor_record(f"{base}.scales", weight_map, headers)

    if weight["dtype"] != "U32":
        raise ValueError(f"{weight['name']}: expected U32, got {weight['dtype']}")
    if scales["dtype"] != "U8":
        raise ValueError(f"{scales['name']}: expected U8, got {scales['dtype']}")
    if weight["shape"][0] != scales["shape"][0]:
        raise ValueError(f"{base}: expert dimension mismatch")
    if weight["shape"][1] != scales["shape"][1]:
        raise ValueError(f"{base}: row dimension mismatch")
    if weight["shape"][2] != scales["shape"][2] * 4:
        raise ValueError(f"{base}: expected 4 U32 words per scale group")

    experts, out_dim, packed_words = weight["shape"]
    groups = scales["shape"][2]
    in_dim = groups * 32
    weight_stride = out_dim * packed_words * 4
    scale_stride = out_dim * groups
    packed_size = out_dim * groups * MXFP4_BLOCK_BYTES
    if weight_stride + scale_stride != packed_size:
        raise ValueError(f"{base}: packed byte accounting mismatch")

    return {
        "kind": short_name,
        "projection": proj_name,
        "type": "MXFP4_SEPARATE",
        "experts": experts,
        "out_dim": out_dim,
        "in_dim": in_dim,
        "groups": groups,
        "packed_words": packed_words,
        "weight": weight,
        "scales": scales,
        "weight_expert_stride": weight_stride,
        "scale_expert_stride": scale_stride,
        "packed_size": packed_size,
    }


def build_layout(repo=DEFAULT_REPO, model_dir=DEFAULT_MODEL_DIR, index_path=DEFAULT_INDEX, layers_spec="all"):
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    max_layer = 1 + max(
        int(name.split(".")[2])
        for name in weight_map
        if name.startswith("model.layers.") and ".ffn.switch_mlp." in name
    )
    layers = parse_layers(layers_spec, max_layer)
    needed_files = sorted({
        weight_map[f"model.layers.{layer}.ffn.switch_mlp.{proj}.{suffix}"]
        for layer in layers
        for _, proj in PROJECTIONS
        for suffix in ("weight", "scales")
    })

    headers = {filename: read_safetensors_header(model_dir, repo, filename) for filename in needed_files}
    component_offsets = {}
    cursor = 0
    layer_records = []
    expert_count = None

    for layer in layers:
        components = []
        for short_name, proj_name in PROJECTIONS:
            component = component_record(layer, short_name, proj_name, weight_map, headers)
            if expert_count is None:
                expert_count = component["experts"]
            elif expert_count != component["experts"]:
                raise ValueError("expert count changed across components")
            if layer == layers[0]:
                component_offsets[short_name] = cursor
                cursor += component["packed_size"]
            component["packed_offset"] = component_offsets[short_name]
            components.append(component)
        layer_records.append({
            "layer": layer,
            "components": components,
        })

    expert_size = cursor
    return {
        "source": "mlx_safetensors",
        "repo": repo,
        "model_dir": portable_path(model_dir),
        "index": portable_path(index_path),
        "metadata": index.get("metadata", {}),
        "block_count": max_layer,
        "layers_included": layers,
        "expert_count": expert_count,
        "expert_size": expert_size,
        "layer_size": expert_size * expert_count,
        "total_routed_expert_bytes": expert_size * expert_count * max_layer,
        "packed_component_layout": [
            {
                "kind": c["kind"],
                "type": "MXFP4_BLOCK",
                "offset": component_offsets[c["kind"]],
                "size": c["packed_size"],
                "out_dim": c["out_dim"],
                "in_dim": c["in_dim"],
                "groups": c["groups"],
            }
            for c in layer_records[0]["components"]
        ],
        "files": {
            filename: {
                "source": header["source"],
                "header_len": header["header_len"],
                "data_start": header["data_start"],
            }
            for filename, header in headers.items()
        },
        "layers": layer_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect DeepSeek V4 Flash MLX 4-bit safetensors expert layout")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--layers", default="all", help="Layer spec: all, 0, 0-3, or 0,2,4")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    layout = build_layout(args.repo, args.model_dir, args.index, args.layers)
    info = HfApi().model_info(args.repo, files_metadata=True)

    print(f"Repo: {args.repo}")
    print(f"Revision: {info.sha}")
    print(f"Files inspected: {len(layout['files'])}")
    print(f"Layers included: {layout['layers_included'][0]}..{layout['layers_included'][-1]} ({len(layout['layers_included'])})")
    print(f"Experts/layer: {layout['expert_count']}")
    print(f"Expert size: {layout['expert_size']:,} bytes ({layout['expert_size'] / 1024**2:.2f} MiB)")
    print(f"Layer size: {layout['layer_size']:,} bytes ({layout['layer_size'] / 1024**3:.2f} GiB)")
    print(f"Total routed expert data: {layout['total_routed_expert_bytes'] / 1024**3:.2f} GiB")
    print("Packed component layout:")
    for component in layout["packed_component_layout"]:
        print(
            f"  {component['kind']}: offset={component['offset']:,} size={component['size']:,} "
            f"shape=[{component['out_dim']}, {component['in_dim']}] groups={component['groups']}"
        )
    print("Shard sources:")
    for filename, record in layout["files"].items():
        print(f"  {filename}: {record['source']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(layout, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
