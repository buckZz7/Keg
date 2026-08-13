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


def build_safetensors(path: Path, arch: str = "muse-glimmer", quant: str = "gptq",
                      n_tensors: int = 2):
    """Build a minimal safetensors: u64 header length + JSON header + pad bytes."""
    meta = {}
    if arch:
        meta["model_type"] = arch
    if quant:
        meta["quant_method"] = quant
    tensors = {
        f"model.weight.{i}": {"dtype": "F16", "shape": [4096, 4096],
                              "data_offsets": [i * 8, i * 8 + 8]}
        for i in range(n_tensors)
    }
    if meta:
        tensors = {"__metadata__": meta, **tensors}
    hdr_json = json.dumps(tensors).encode()
    data = struct.pack("<Q", len(hdr_json)) + hdr_json + b"\x00" * 64
    path.write_bytes(data)
    return data


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
    """Parse a small real GGUF (downloads ~428MB). Skip via KEG_SKIP_REAL=1."""
    if os.environ.get("KEG_SKIP_REAL") == "1":
        print("[skip] real GGUF test disabled (KEG_SKIP_REAL=1)")
        return
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


def test_disambiguation_filter():
    from build_corpus import is_disambiguation
    disamb = [
        # German "X steht für" list page
        "Music steht für: Music (Kentucky), Ort in den USA Music (Lied), Lied von John Miles",
        # Dutch "X kan verwijzen naar"
        "Sun (Engels voor zon) kan verwijzen naar: SUN (media), een Surinaamse nieuwswebsite",
        # Spanish "X puede referirse a"
        "La palabra sun puede referirse a: Sun, unidad de longitud antiguamente utilizada en Japón",
        # structural: short + parenthetical-heavy (no marker)
        "Human (film 2015) Human (album) Human (band) Human (Lied)",
    ]
    clean = [
        "Die Physik (bundesdeutsches Hochdeutsch: [fyˈziːk]) ist die Naturwissenschaft...",
        "La chimie est une science de la nature qui étudie la matière et ses transformations...",
        "Technologie adalah penerapan pengetahuan konseptual untuk mencapai tujuan praktis...",
        "The chemical element oxygen is essential for life and makes up about 21% of the atmosphere...",
    ]
    for d in disamb:
        check("disamb filter drops: %s" % d[:24], is_disambiguation(d), d[:60])
    for c in clean:
        check("disamb filter keeps: %s" % c[:24], not is_disambiguation(c), c[:60])


def test_verify_submission_format_dispatch():
    """Format detection routes GGUF vs safetensors from bytes, not the claim."""
    from verify_submission import _detect_format, verify
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # GGUF file detected as gguf
        g = td / "m.gguf"
        build_gguf(g, "muse-glimmer")
        check("detect: gguf", _detect_format(str(g)) == "gguf")

        # safetensors file detected as safetensors
        st = td / "m.safetensors"
        build_safetensors(st, arch="muse-glimmer", quant="gptq")
        check("detect: safetensors", _detect_format(str(st)) == "safetensors")

        # junk rejected
        j = td / "junk.bin"
        j.write_bytes(b"\x00\x01\x02\x03" + b"\x00" * 64)
        check("detect: unknown", _detect_format(str(j)) == "unknown")


def test_verify_submission_safetensors():
    from verify_submission import verify
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "m.safetensors"
        build_safetensors(p, arch="muse-glimmer", quant="gptq")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        sub = {"lane": "glimmer-30b", "quant": "NVFP4", "file": "m.safetensors",
               "source": str(p), "sha256": sha}
        rec = verify(sub, "muse-glimmer", str(td / "stage"))
        check("safetensors: format", rec["format"] == "safetensors")
        check("safetensors: sha match", rec["sha256"] == sha)
        check("safetensors: arch verified",
              rec["architecture"] == "muse-glimmer" and rec["architecture_verified"])
        check("safetensors: quant scheme", rec["quantization_scheme"] == "gptq")
        check("safetensors: tensor hist", "F16" in rec["tensors_by_type"],
              json.dumps(rec["tensors_by_type"]))

    # wrong arch in safetensors metadata must raise
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "m.safetensors"
        build_safetensors(p, arch="qwen2", quant="gptq")
        sub = {"lane": "glimmer-30b", "quant": "NVFP4", "file": "m.safetensors",
               "source": str(p)}
        try:
            verify(sub, "muse-glimmer", str(td / "stage"))
            check("safetensors: wrong arch raises", False)
        except ValueError:
            check("safetensors: wrong arch raises", True)

    # safetensors with NO arch in header: structural check passes, arch deferred
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "m.safetensors"
        build_safetensors(p, arch="", quant="gptq")
        sub = {"lane": "glimmer-30b", "quant": "NVFP4", "file": "m.safetensors",
               "source": str(p)}
        rec = verify(sub, "muse-glimmer", str(td / "stage"))
        check("safetensors: no-arch deferred",
              rec["architecture"] is None and rec["architecture_verified"] is False)


def test_verify_submission_unknown_format():
    from verify_submission import verify
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "junk.bin"
        p.write_bytes(b"\x00\x01\x02\x03" + b"\x00" * 64)
        sub = {"lane": "glimmer-30b", "quant": "Q6_K", "file": "junk.bin",
               "source": str(p)}
        try:
            verify(sub, "muse-glimmer", str(td / "stage"))
            check("verify: junk rejected", False)
        except ValueError:
            check("verify: junk rejected", True)


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
    test_verify_submission_format_dispatch()
    test_verify_submission_safetensors()
    test_verify_submission_unknown_format()
    test_disambiguation_filter()
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
