#!/usr/bin/env python3
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKED_DIR = (
    ROOT.parent
    / "deepseek-v4-experiments"
    / "models"
    / "antirez-deepseek-v4-gguf"
    / "packed_experts_deepseek"
)
DEFAULT_GGML_COMMON = (
    ROOT.parent
    / "deepseek-v4-experiments"
    / "llama.cpp-deepseek-v4-flash"
    / "ggml"
    / "src"
    / "ggml-common.h"
)
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and not str(Path(sys.executable)).startswith(str(ROOT / ".venv")):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np  # noqa: E402


HIDDEN = 4096
INTERMEDIATE = 2048
EXPERT_COUNT = 256
GATE_SIZE = 2_162_688
UP_SIZE = 2_162_688
DOWN_SIZE = 2_752_512
EXPERT_SIZE = GATE_SIZE + UP_SIZE + DOWN_SIZE
IQ2_XXS_BLOCK = 66
Q2_K_BLOCK = 84
QK_K = 256
KMASK_IQ2XS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)


def parse_ggml_table(header_text, name):
    pattern = rf"GGML_TABLE_BEGIN\([^,]+,\s*{re.escape(name)},\s*[^)]+\)(.*?)GGML_TABLE_END\(\)"
    match = re.search(pattern, header_text, flags=re.S)
    if not match:
        raise ValueError(f"table {name} not found")
    body = re.sub(r"//.*", "", match.group(1))
    values = []
    for token in body.replace("\n", " ").split(","):
        token = token.strip()
        if token:
            values.append(int(token, 0))
    return values


def load_iq2_tables(path):
    text = path.read_text(encoding="utf-8")
    signs = np.array(parse_ggml_table(text, "ksigns_iq2xs"), dtype=np.uint8)
    grid_words = parse_ggml_table(text, "iq2xxs_grid")
    grid = np.empty((len(grid_words), 8), dtype=np.float32)
    for idx, word in enumerate(grid_words):
        grid[idx] = np.frombuffer(word.to_bytes(8, "little"), dtype=np.uint8).astype(np.float32)
    sign_vectors = np.where((signs[:, None] & KMASK_IQ2XS[None, :]) != 0, -1.0, 1.0).astype(np.float32)
    return grid, signs, sign_vectors


def f16_at(buf, offset):
    return float(np.frombuffer(buf, dtype="<f2", count=1, offset=offset)[0])


def dequant_iq2_xxs_block(buf, grid, signs, sign_vectors):
    d = f16_at(buf, 0)
    qs = memoryview(buf)[2:]
    out = np.empty(QK_K, dtype=np.float32)
    pos = 0
    for ib32 in range(8):
        aux = qs[ib32 * 8 : (ib32 + 1) * 8]
        aux32_hi = int.from_bytes(aux[4:8], "little")
        db = d * (0.5 + (aux32_hi >> 28)) * 0.25
        for lane in range(4):
            grid_idx = aux[lane]
            sign_idx = (aux32_hi >> (7 * lane)) & 127
            out[pos : pos + 8] = db * grid[grid_idx] * sign_vectors[sign_idx]
            pos += 8
    return out


def dequant_q2_k_block(buf):
    scales = np.frombuffer(buf, dtype=np.uint8, count=16, offset=0)
    q = np.frombuffer(buf, dtype=np.uint8, count=64, offset=16)
    d = f16_at(buf, 80)
    dmin = f16_at(buf, 82)
    out = np.empty(QK_K, dtype=np.float32)
    scale_idx = 0
    q_off = 0
    out_off = 0

    for _ in range(2):
        shift = 0
        for _ in range(4):
            sc = int(scales[scale_idx])
            scale_idx += 1
            vals = ((q[q_off : q_off + 16] >> shift) & 3).astype(np.float32)
            out[out_off : out_off + 16] = d * (sc & 0xF) * vals - dmin * (sc >> 4)
            out_off += 16

            sc = int(scales[scale_idx])
            scale_idx += 1
            vals = ((q[q_off + 16 : q_off + 32] >> shift) & 3).astype(np.float32)
            out[out_off : out_off + 16] = d * (sc & 0xF) * vals - dmin * (sc >> 4)
            out_off += 16
            shift += 2
        q_off += 32

    return out


def matvec_iq2_xxs(data, rows, cols, x, grid, signs, sign_vectors):
    row_size = (cols // QK_K) * IQ2_XXS_BLOCK
    out = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        row_base = row * row_size
        acc = 0.0
        for block_idx in range(cols // QK_K):
            start = row_base + block_idx * IQ2_XXS_BLOCK
            vals = dequant_iq2_xxs_block(data[start : start + IQ2_XXS_BLOCK], grid, signs, sign_vectors)
            x_part = x[block_idx * QK_K : (block_idx + 1) * QK_K]
            acc += float(np.dot(vals, x_part))
        out[row] = acc
    return out


def matvec_q2_k(data, rows, cols, x):
    row_size = (cols // QK_K) * Q2_K_BLOCK
    out = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        row_base = row * row_size
        acc = 0.0
        for block_idx in range(cols // QK_K):
            start = row_base + block_idx * Q2_K_BLOCK
            vals = dequant_q2_k_block(data[start : start + Q2_K_BLOCK])
            x_part = x[block_idx * QK_K : (block_idx + 1) * QK_K]
            acc += float(np.dot(vals, x_part))
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
    parser = argparse.ArgumentParser(description="CPU reference probe for one DeepSeek V4 Flash routed expert")
    parser.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    parser.add_argument("--ggml-common", type=Path, default=DEFAULT_GGML_COMMON)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    if args.layer < 0 or args.layer >= 43:
        raise ValueError("--layer must be in 0..42")
    if args.expert < 0 or args.expert >= EXPERT_COUNT:
        raise ValueError(f"--expert must be in 0..{EXPERT_COUNT - 1}")

    layer_path = args.packed_dir / f"layer_{args.layer:02d}.bin"
    if not layer_path.exists():
        raise FileNotFoundError(f"{layer_path} not found; run scripts/repack_deepseek_gguf_experts.py first")
    if not args.ggml_common.exists():
        raise FileNotFoundError(f"{args.ggml_common} not found; run scripts/setup_deepseek_gguf_runtime.sh first")

    print(f"Packed layer: {layer_path}")
    print(f"Layer/expert: {args.layer}/{args.expert}")
    print(f"Expert offset: {args.expert * EXPERT_SIZE:,}")

    t0 = time.monotonic()
    grid, signs, sign_vectors = load_iq2_tables(args.ggml_common)
    fd = os.open(layer_path, os.O_RDONLY)
    try:
        expert = os.pread(fd, EXPERT_SIZE, args.expert * EXPERT_SIZE)
    finally:
        os.close(fd)
    if len(expert) != EXPERT_SIZE:
        raise IOError(f"short read: {len(expert):,}/{EXPERT_SIZE:,} bytes")

    idx = np.arange(HIDDEN, dtype=np.float32)
    x = np.sin(idx * np.float32(0.013)) + np.float32(0.5) * np.cos(idx * np.float32(0.021))

    gate_data = expert[:GATE_SIZE]
    up_data = expert[GATE_SIZE : GATE_SIZE + UP_SIZE]
    down_data = expert[GATE_SIZE + UP_SIZE :]

    gate = matvec_iq2_xxs(gate_data, INTERMEDIATE, HIDDEN, x, grid, signs, sign_vectors)
    up = matvec_iq2_xxs(up_data, INTERMEDIATE, HIDDEN, x, grid, signs, sign_vectors)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
    act = (gate * sigmoid * up).astype(np.float32)
    out = matvec_q2_k(down_data, HIDDEN, INTERMEDIATE, act)

    describe("gate", gate)
    describe("up", up)
    describe("act", act)
    describe("out", out)
    digest = hashlib.sha256(out.tobytes()).hexdigest()
    print(f"out sha256: {digest}")
    print(f"elapsed: {time.monotonic() - t0:.2f}s")


if __name__ == "__main__":
    main()
