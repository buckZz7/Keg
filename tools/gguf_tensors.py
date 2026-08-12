"""gguf_tensors.py — read the objective per-tensor quantization structure from a
GGUF file, for the receipt. This is house-inspected from the actual artifact (not
a miner claim), so it can't be faked and is bound to the sha256'd file.

Emits a JSON summary: the count of tensors per quant type, plus the per-tensor
map (name -> type), so a miner can see exactly how a recipe is quantized and
build on it.

usage: python3 gguf_tensors.py model.gguf [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ3_M", 23: "IQ2_S",
    24: "IQ2_M", 25: "IQ4_XS", 26: "IQ1_M", 27: "BF16", 28: "Q4_0_4_4",
    29: "Q4_0_4_8", 30: "Q4_0_8_8", 31: "TQ1_0", 32: "TQ2_0", 33: "IQ4_NL_4_4",
    34: "IQ4_NL_4_8", 35: "IQ4_NL_8_8", 36: "Q4_1_4_4", 37: "Q4_1_4_8",
    38: "Q4_1_8_8", 39: "Q5_0_4_4", 40: "Q5_0_4_8", 41: "Q5_0_8_8",
    42: "Q5_1_4_4", 43: "Q5_1_4_8", 44: "Q5_1_8_8", 45: "Q6_K_4_4",
    46: "Q6_K_4_8", 47: "Q6_K_8_8", 48: "IQ4_K", 49: "IQ5_K", 50: "IQ6_K",
    51: "IQ8_K", 52: "Q5_0_4_4_SP", 53: "Q6_K_XL", 54: "Q5_K_XL", 55: "Q4_K_XL",
    56: "TQ2_0_4_4", 57: "TQ2_0_4_8", 58: "TQ2_0_8_8",
}

# GGUF metadata value types: (1..13). 9=string, 10=array.
_META_VAL_SIZE = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 1, 11: 8, 12: 8, 13: 8,
}


def _read_str(f, pos):
    n = struct.unpack_from("<Q", f, pos)[0]
    s = f[pos + 8: pos + 8 + n].decode("utf-8", "replace")
    return s, pos + 8 + n


def _skip_value(f, pos, vtype):
    """Return pos after a metadata value of vtype."""
    if vtype == 9:  # string
        _, pos = _read_str(f, pos)
    elif vtype == 10:  # array: u32 count + u32 elem_type + elems
        count, elem = struct.unpack_from("<II", f, pos)
        pos += 8
        for _ in range(count):
            pos = _skip_value(f, pos, elem)
    else:
        pos += _META_VAL_SIZE[vtype]
    return pos


def inspect(path: str) -> dict:
    f = Path(path).read_bytes()
    magic = f[:4]
    assert magic == b"GGUF", "not a GGUF file"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", f, 4)
    pos = 4 + 4 + 8 + 8
    # skip metadata KV pairs
    for _ in range(n_kv):
        _, pos = _read_str(f, pos)          # key
        vtype = struct.unpack_from("<I", f, pos)[0]
        pos = _skip_value(f, pos + 4, vtype)
    # tensor infos
    tensors = {}
    for _ in range(n_tensors):
        name, pos = _read_str(f, pos)
        n_dims = struct.unpack_from("<I", f, pos)[0]
        pos += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, f, pos)
        pos += 8 * n_dims
        ggml_type = struct.unpack_from("<I", f, pos)[0]
        pos += 4
        offset = struct.unpack_from("<Q", f, pos)[0]
        pos += 8
        tensors[name] = {"type": GGML_TYPE.get(ggml_type, f"T{ggml_type}"),
                         "dims": list(dims), "offset": offset}
    # summarize
    from collections import Counter
    by_type = Counter(t["type"] for t in tensors.values())
    nbytes = len(f)
    summary = {
        "gguf_version": version,
        "n_tensors": n_tensors,
        "size_bytes": nbytes,
        "tensors_by_type": dict(by_type),
        "tensors": tensors,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--out")
    args = ap.parse_args()
    s = inspect(args.gguf)
    print(json.dumps(s["tensors_by_type"], indent=2))
    print(f"  ({s['n_tensors']} tensors, {s['size_bytes']/1e9:.1f}GB)")
    if args.out:
        Path(args.out).write_text(json.dumps(s, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
