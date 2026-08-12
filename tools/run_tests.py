"""run_tests.py — unit tests for the Keg trustless core tools.

These guard the machinery a miner's submission and a receipt depend on. The
motivation: a synthetic-only GGUF test missed a real parsing bug that would have
broken eval. So the suite includes both a real-format builder AND, where possible,
tests against a small real GGUF.

Run: python3 tools/run_tests.py
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    status = "ok  " if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{status}] {name}{' — ' + detail if detail else ''}")


# ---------------------------------------------------------------- GGUF builder
def build_gguf(path: Path, arch: str, n_tensors: int = 3, quant_type: int = 14):
    """Build a minimal-but-correct GGUF exercising strings, arrays, and tensors."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def meta_str(key, val):   # string value (type 8)
        return s(key) + struct.pack("<I", 8) + s(val)

    def meta_arr_u32(key, vals):  # array of u32 (type 9, elem 4)
        out = s(key) + struct.pack("<I", 9)
        out += struct.pack("<I", 4)          # elem_type = UINT32
        out += struct.pack("<Q", len(vals))  # n_elements = u64
        for v in vals:
            out += struct.pack("<I", v)
        return out

    def tensor(name, ttype, dims):
        out = s(name) + struct.pack("<I", len(dims))
        out += struct.pack("<" + "Q" * len(dims), *dims)
        out += struct.pack("<IQ", ttype, 0)
        return out

    body = b"GGUF" + struct.pack("<IQQ", 3, n_tensors, 3)
    body += meta_str("general.architecture", arch)
    body += meta_str("general.name", f"test-{arch}")
    body += meta_arr_u32("tokenizer.ggml.token_type", [0, 1, 2, 3, 4])
    body += tensor("tok_embeddings.weight", 1, [32000, 4096])      # F16
    body += tensor("output.weight", quant_type, [32000, 4096])     # e.g. Q6_K
    body += tensor("blk.0.attn_q.weight", 1, [4096, 4096])         # F16
    path.write_bytes(body)
    return body


# ---------------------------------------------------------------------- tests
def test_gguf_inspect():
    from gguf_tensors import inspect
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf"
        build_gguf(p, "muse-glimmer", n_tensors=3, quant_type=14)
        g = inspect(str(p))
        check("gguf: arch", g["architecture"] == "muse-glimmer", g["architecture"])
        check("gguf: n_tensors", g["n_tensors"] == 3, str(g["n_tensors"]))
        check("gguf: has Q6_K", g["tensors_by_type"].get("Q6_K") == 1,
              json.dumps(g["tensors_by_type"]))
        check("gguf: size", g["size_bytes"] == p.stat().st_size)


def test_gguf_real_file():
    """Parse a small real GGUF if present (else skip). Download one on demand."""
    from gguf_tensors import inspect
    import urllib.request
    url = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
           "qwen2.5-0.5b-instruct-q4_0.gguf")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "q4.gguf"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                while True:
                    ch = r.read(1 << 20)
                    if not ch:
                        break
                    f.write(ch)
        except Exception as e:
            print("[skip] real GGUF download failed:", e)
            return
        g = inspect(str(p))
        check("real gguf: arch", g["architecture"] == "qwen2", g["architecture"])
        check("real gguf: tensors>100", g["n_tensors"] > 100, str(g["n_tensors"]))
        check("real gguf: has Q4_0", g["tensors_by_type"].get("Q4_0", 0) > 100,
              json.dumps(g["tensors_by_type"]))


def test_safetensors_inspect():
    from safetensors_inspect import inspect
    hdr = {
        "__metadata__": {"model_type": "muse-glimmer", "quant_method": "gptq"},
        "model.embed.weight": {"dtype": "F16", "shape": [32000, 4096],
                               "data_offsets": [0, 1]},
        "lm_head.weight": {"dtype": "F16", "shape": [32000, 4096],
                           "data_offsets": [1, 2]},
    }
    hdr_json = json.dumps(hdr).encode()
    data = struct.pack("<Q", len(hdr_json)) + hdr_json + b"\x00" * 64
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        p.write_bytes(data)
        s = inspect(str(p))
        check("safetensors: arch", s["architecture_hint"] == "muse-glimmer",
              s["architecture_hint"])
        check("safetensors: quant", s["quantization"] == "gptq", s["quantization"])
        check("safetensors: n_tensors", s["n_tensors"] == 2, str(s["n_tensors"]))


def test_verify_submission_local():
    from verify_submission import verify
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf"
        build_gguf(p, "muse-glimmer")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        sub = {"lane": "glimmer-30b", "quant": "Q6_K", "file": "m.gguf",
               "source": str(p), "sha256": sha}
        rec = verify(sub, "muse-glimmer", str(Path(td) / "stage"))
        check("verify: sha match", rec["sha256"] == sha)
        check("verify: size", rec["size_bytes"] == p.stat().st_size)
        check("verify: arch", rec["architecture"] == "muse-glimmer")
        check("verify: tensor types", "Q6_K" in rec["tensors_by_type"],
              json.dumps(rec["tensors_by_type"]))
    # hash-mismatch must raise
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.gguf"
        build_gguf(p, "muse-glimmer")
        sub = {"lane": "glimmer-30b", "quant": "Q6_K", "file": "m.gguf",
               "source": str(p), "sha256": "0" * 64}
        try:
            verify(sub, "muse-glimmer", str(Path(td) / "stage"))
            check("verify: mismatch raises", False)
        except ValueError:
            check("verify: mismatch raises", True)


def test_receipt_replay():
    from keg.receipt import build_receipt, verify_receipt
    from keg.recipe import Recipe
    r = Recipe(model="muse-glimmer-30b", model_file="x.gguf",
               model_sha256="a" * 64, quant="Q6_K", source="https://hf.co/x")
    rec = build_receipt(r, {"top1_match": 0.96, "kl_mean": 0.017,
                            "kl_max_component": 0.0199,
                            "kl_by_component": {"prose": 0.019}},
                        22867215136, "2026-08-12", tps=50.0)
    check("receipt: has sha", "receipt_sha256" in rec)
    check("receipt: has tps", rec["tps"] == 50.0)
    check("receipt: replay verifies", verify_receipt(rec) is True)


def main():
    test_gguf_inspect()
    test_safetensors_inspect()
    test_verify_submission_local()
    test_receipt_replay()
    test_gguf_real_file()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
