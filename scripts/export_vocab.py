#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path


def bytes_to_unicode():
    # GPT-2/ByteLevel BPE reversible byte mapping.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = bytes_to_unicode()


def decode_bytelevel_token(token: str) -> str:
    if token.startswith("<|") and token.endswith("|>"):
        return token
    raw = bytearray()
    for ch in token:
        value = BYTE_DECODER.get(ch)
        if value is None:
            return token
        raw.append(value)
    return raw.decode("utf-8", errors="replace")


def main():
    if len(sys.argv) != 3:
        print("Usage: scripts/export_vocab.py TOKENIZER_JSON OUT_VOCAB_BIN", file=sys.stderr)
        return 1

    tokenizer_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    token_by_id = {}

    for token, token_id in tokenizer["model"]["vocab"].items():
        token_by_id[int(token_id)] = token

    for token in tokenizer.get("added_tokens", []):
        token_by_id[int(token["id"])] = token["content"]

    max_id = max(token_by_id)
    num_entries = max_id + 1

    with out_path.open("wb") as f:
        f.write(struct.pack("<I", num_entries))
        f.write(struct.pack("<I", max_id))
        for token_id in range(num_entries):
            token = decode_bytelevel_token(token_by_id.get(token_id, ""))
            raw = token.encode("utf-8")
            if len(raw) > 0xFFFF:
                raise ValueError(f"token {token_id} is too large for vocab.bin")
            f.write(struct.pack("<H", len(raw)))
            f.write(raw)

    print(f"Exported vocab to {out_path}: {num_entries} slots, max_id={max_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
