"""safetensors_inspect.py — read the objective per-tensor structure from a
safetensors file WITHOUT loading the weights.

Safetensors stores all tensor metadata (names, shapes, dtypes, and byte offsets)
in a JSON header that precedes the raw bytes. So we can inspect the structure
cheaply — enough for the anti-memorization gate (prove it's a real model of the
lane's architecture with real weight tensors, not a lookup table) and to report
the true per-tensor layout for the receipt.

This is the non-GGUF counterpart to gguf_tensors.py.

usage: python3 safetensors_inspect.py model.safetensors [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def inspect(path: str) -> dict:
    f = Path(path).read_bytes()
    # header: u64 header-length (bytes of JSON), then the JSON, then tensor data
    (hdr_len,) = struct.unpack_from("<Q", f, 0)
    header = json.loads(f[8:8 + hdr_len].decode("utf-8"))
    meta = header.get("__metadata__", {})
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}

    # header byte offsets are relative to the START of the data section
    # (after the 8-byte length + hdr_len bytes of JSON)
    data_start = 8 + hdr_len
    sizes = {}
    for name, t in tensors.items():
        shape = t.get("shape", [])
        dtype = t.get("dtype", "?")
        n = 1
        for d in shape:
            n *= d
        bpe = _DTYPE_BYTES.get(dtype, 1)
        sizes[name] = bpe * n
    nbytes = len(f)

    # tensor dtype histogram
    from collections import Counter
    by_dtype = Counter(t.get("dtype", "?") for t in tensors.values())

    return {
        "format": "safetensors",
        "n_tensors": len(tensors),
        "size_bytes": nbytes,
        "tensors_by_dtype": dict(by_dtype),
        "architecture_hint": meta.get("general.architecture")
                              or meta.get("model_type")
                              or meta.get("quantized_model", ""),
        "quantization": meta.get("quant_method", "")
                         or meta.get("quantization_config", ""),
        "tensors": {name: {"dtype": t.get("dtype"), "shape": t.get("shape"),
                           "offset": data_start + t.get("data_offsets", [0])[0]}
                    for name, t in tensors.items()},
        "n_params": sum(sizes.values()),
    }


_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1, "I32": 4,
    "I64": 8, "BOOL": 1, "U16": 2, "I16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors")
    ap.add_argument("--out")
    args = ap.parse_args()
    s = inspect(args.safetensors)
    print(json.dumps({k: s[k] for k in
                      ["format", "n_tensors", "size_bytes", "tensors_by_dtype",
                       "architecture_hint", "quantization"]}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(s, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
