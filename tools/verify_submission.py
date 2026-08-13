"""verify_submission.py — house-side intake verification for a submitted recipe.

The trustless submission flow: a miner submits a recipe (a JSON pointing at the
real model file + its source). The house does NOT trust the miner's claims — it
fetches the actual file, hashes it, checks it's a REAL model file of the lane's
architecture, and inspects the true per-tensor quantization structure. The output
is the set of house-verified facts that the measure pipeline + receipt bind to.

Only facts the house can verify from the artifact are emitted. Miner-claimed
fields (base, imatrix, quantization command) are deliberately NOT trusted and are
not part of the receipt — they can be faked.

The format is detected from the artifact's own bytes (never from the miner's
claim): GGUF (llama.cpp) or safetensors (vLLM / GPTQ / AWQ / NVFP4). Both go
through a header-only anti-memorization gate — a real model of the lane's arch
with real weight tensors, not a lookup table — and both report the objective
per-tensor layout for the receipt. (Structural inspection never loads weights.)

For GGUF the architecture is read from the file's own metadata, so it's checked
strictly. For safetensors the architecture may or may not be in the header: when
it's present we check it; when it's absent it cannot be confirmed from the file
alone, so the structural check stands and the definitive architecture + fidelity
verification is deferred to the eval adapter (a non-loadable/wrong-arch model
fails there).

Input JSON:
  {
    "lane": "glimmer-30b",
    "quant": "Q6_K",
    "file": "keg-Q6_K.gguf",
    "source": "https://huggingface.co/.../keg-Q6_K.gguf",   # URL or local path
    "sha256": "d553d2...",                                  # optional expected hash
  }

usage:
  python3 tools/verify_submission.py sub.json --expected-arch muse-glimmer \
      --out intake.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_tensors import inspect as inspect_gguf  # noqa: E402
from safetensors_inspect import inspect as inspect_safetensors  # noqa: E402

# GGUF magic: first 4 bytes. safetensors has no magic constant; the first 8 bytes
# are a u64 header length. Distinguish by GGUF magic first, else safetensors.
_GGUF_MAGIC = b"GGUF"


def _detect_format(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == _GGUF_MAGIC:
        return "gguf"
    # safetensors: first 8 bytes = header length, then that many bytes of JSON
    # (a mapping of tensor names -> info). Require the header to actually parse
    # as a JSON object, so random files with plausible lengths are rejected.
    if len(head) == 8:
        import struct
        try:
            (hdr_len,) = struct.unpack_from("<Q", head)
        except Exception:
            return "unknown"
        if 0 < hdr_len < (1 << 40):
            with open(path, "rb") as f:
                hdr = f.read(8 + hdr_len)[8:]
            try:
                obj = json.loads(hdr)
            except (ValueError, UnicodeDecodeError):
                return "unknown"
            if isinstance(obj, dict):
                return "safetensors"
    return "unknown"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(source: str, dest: str) -> None:
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    else:
        Path(dest).write_bytes(Path(source).read_bytes())


def verify(sub: dict, expected_arch: str, staging_dir: str) -> dict:
    lane = sub["lane"]
    quant = sub["quant"]
    source = sub["source"]
    expected = sub.get("sha256", "").strip().lower()

    os.makedirs(staging_dir, exist_ok=True)
    local = str(Path(staging_dir) / sub["file"])

    # 1) fetch the actual artifact
    _fetch(source, local)
    sha = _sha256(local)
    if expected and sha != expected:
        raise ValueError(
            f"hash mismatch: submitted sha256 {expected} != actual {sha[:16]}... "
            f"(the file does not match what the miner claimed)")
    size = os.path.getsize(local)

    # 2) detect the format from the artifact's own bytes (not the miner's claim)
    fmt = _detect_format(local)
    if fmt == "gguf":
        try:
            g = inspect_gguf(local)
        except AssertionError as e:
            raise ValueError(f"not a loadable GGUF: {e}")
        arch = g["architecture"]
        if arch != expected_arch:
            raise ValueError(
                f"architecture mismatch: file is {arch!r}, lane expects {expected_arch!r}")
        arch_verified = True
        tensors_hist = g["tensors_by_type"]
        n_tensors = g["n_tensors"]
        quant_scheme = None  # gguf scheme is implied by tensors_by_type, not metadata
    elif fmt == "safetensors":
        try:
            s = inspect_safetensors(local)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"not a loadable safetensors: {e}")
        hint = (s.get("architecture_hint") or "").strip()
        if hint:
            if hint != expected_arch:
                raise ValueError(
                    f"architecture mismatch: file metadata says {hint!r}, "
                    f"lane expects {expected_arch!r}")
            arch_verified = True
        else:
            # header carries no arch — structural check only; defer to adapter
            arch_verified = False
        arch = hint or None
        tensors_hist = s["tensors_by_dtype"]
        n_tensors = s["n_tensors"]
        quant_scheme = s.get("quantization") or ""
    else:
        raise ValueError(
            f"unrecognized model format for {sub['file']} "
            f"(expected GGUF or safetensors)")

    # 3) house-verified intake record — only what we measured/inspected
    return {
        "lane": lane,
        "quant": quant,
        "format": fmt,
        "file": sub["file"],
        "source": source,
        "sha256": sha,
        "size_bytes": size,
        "architecture": arch,
        "architecture_verified": arch_verified,
        "n_tensors": n_tensors,
        "tensors_by_type": tensors_hist,
        "quantization_scheme": quant_scheme,
        # note: full per-tensor map is available from the inspector; keep the
        # receipt lean but the tensor histogram proves the objective layout
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="submission JSON path")
    ap.add_argument("--expected-arch", required=True, help="lane's architecture")
    ap.add_argument("--staging", default="/tmp/keg_ingest", help="staging dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub = json.loads(Path(args.submission).read_text())
    rec = verify(sub, args.expected_arch, args.staging)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
