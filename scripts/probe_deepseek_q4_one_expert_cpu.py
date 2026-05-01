#!/usr/bin/env python3
import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKED_DIR = ROOT / "models" / "deepseek-v4-flash-4bit" / "packed_experts_q4"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np  # noqa: E402


HIDDEN = 4096
INTERMEDIATE = 2048
EXPERT_COUNT = 256
GROUP = 32
MXFP4_BLOCK = 17
COMPONENT_SIZE = 4_456_448
EXPERT_SIZE = COMPONENT_SIZE * 3
KVALUES_MXFP4 = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.float32,
)
SCALE_LUT = np.array([math.ldexp(1.0, i - 128) for i in range(256)], dtype=np.float32)


def matvec_mxfp4(component, rows, cols, x):
    groups = cols // GROUP
    view = np.frombuffer(component, dtype=np.uint8).reshape(rows, groups, MXFP4_BLOCK)
    out = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        acc = 0.0
        row_blocks = view[row]
        for group_idx in range(groups):
            block = row_blocks[group_idx]
            d = SCALE_LUT[block[0]]
            q = block[1:]
            low = KVALUES_MXFP4[q & 0x0F] * d
            high = KVALUES_MXFP4[q >> 4] * d
            base = group_idx * GROUP
            acc += float(np.dot(low, x[base : base + 16]))
            acc += float(np.dot(high, x[base + 16 : base + 32]))
        out[row] = acc
    return out


def describe(name, values):
    top = np.argsort(np.abs(values))[-5:][::-1]
    print(
        f"{name}: shape={values.shape} min={values.min():.6g} max={values.max():.6g} "
        f"mean={values.mean():.6g} rms={np.sqrt(np.mean(values * values)):.6g}"
    )
    print(f"{name} top_abs:", ", ".join(f"{int(i)}:{values[i]:.6g}" for i in top))


def main():
    parser = argparse.ArgumentParser(description="CPU reference probe for one DeepSeek V4 Flash MLX Q4 routed expert")
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    if args.layer < 0 or args.layer >= 43:
        raise ValueError("--layer must be in 0..42")
    if args.expert < 0 or args.expert >= EXPERT_COUNT:
        raise ValueError(f"--expert must be in 0..{EXPERT_COUNT - 1}")

    layer_path = args.packed_dir / f"layer_{args.layer:02d}.bin"
    if not layer_path.exists():
        raise FileNotFoundError(f"{layer_path} not found; run scripts/repack_deepseek_q4_experts.py first")

    print(f"Packed layer: {layer_path}")
    print(f"Layer/expert: {args.layer}/{args.expert}")
    print(f"Expert offset: {args.expert * EXPERT_SIZE:,}")

    t0 = time.monotonic()
    fd = os.open(layer_path, os.O_RDONLY)
    try:
        expert = os.pread(fd, EXPERT_SIZE, args.expert * EXPERT_SIZE)
    finally:
        os.close(fd)
    if len(expert) != EXPERT_SIZE:
        raise IOError(f"short read: {len(expert):,}/{EXPERT_SIZE:,} bytes")

    idx = np.arange(HIDDEN, dtype=np.float32)
    x = np.sin(idx * np.float32(0.013)) + np.float32(0.5) * np.cos(idx * np.float32(0.021))

    gate_data = expert[:COMPONENT_SIZE]
    up_data = expert[COMPONENT_SIZE : COMPONENT_SIZE * 2]
    down_data = expert[COMPONENT_SIZE * 2 :]

    gate = matvec_mxfp4(gate_data, INTERMEDIATE, HIDDEN, x)
    up = matvec_mxfp4(up_data, INTERMEDIATE, HIDDEN, x)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
    act = (gate * sigmoid * up).astype(np.float32)
    out = matvec_mxfp4(down_data, HIDDEN, INTERMEDIATE, act)

    describe("gate", gate)
    describe("up", up)
    describe("act", act)
    describe("out", out)
    digest = hashlib.sha256(out.tobytes()).hexdigest()
    print(f"out sha256: {digest}")
    print(f"elapsed: {time.monotonic() - t0:.2f}s")


if __name__ == "__main__":
    main()
