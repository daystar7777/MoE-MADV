#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-$ROOT/models/qwen3.5-397b-a17b-4bit}"
OUT_DIR="${2:-$ROOT/metal_infer}"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv is missing. Run scripts/bootstrap_local_env.sh first." >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "ERROR: model files not found at $MODEL_DIR" >&2
  echo "Run: scripts/download_model_assets.sh full qwen" >&2
  exit 1
fi

echo "Exporting tokenizer..."
.venv/bin/python metal_infer/export_tokenizer.py "$MODEL_DIR/tokenizer.json" "$OUT_DIR/tokenizer.bin"
.venv/bin/python scripts/export_vocab.py "$MODEL_DIR/tokenizer.json" "$OUT_DIR/vocab.bin"

echo "Extracting non-expert weights..."
.venv/bin/python metal_infer/extract_weights.py --model "$MODEL_DIR" --output "$OUT_DIR"

echo "Packing Qwen experts for SSD streaming..."
TMP_INDEX="$ROOT/.tmp_expert_index.json"
.venv/bin/python - "$ROOT/expert_index.json" "$TMP_INDEX" "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model_dir = str(Path(sys.argv[3]).resolve())

data = json.loads(src.read_text())
data["model_path"] = model_dir
dst.write_text(json.dumps(data, indent=2))
PY
.venv/bin/python repack_experts.py --index "$TMP_INDEX"
rm -f "$TMP_INDEX"

cat <<EOF
Done.

Run:
  cd "$ROOT/metal_infer"
  make
  ./infer --model "$MODEL_DIR" --weights "$OUT_DIR/model_weights.bin" --manifest "$OUT_DIR/model_weights.json" --prompt "Explain quantum computing" --tokens 20 --timing
EOF
