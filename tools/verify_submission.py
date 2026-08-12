"""verify_submission.py — house-side intake verification for a submitted recipe.

The trustless submission flow: a miner submits a recipe (a JSON pointing at the
real model file + its source). The house does NOT trust the miner's claims — it
fetches the actual file, hashes it, checks it's a real GGUF of the lane's
architecture, and inspects the true per-tensor quantization structure. The output
is the set of house-verified facts that the measure pipeline + receipt bind to.

Only facts the house can verify from the artifact are emitted. Miner-claimed
fields (base, imatrix, quantization command) are deliberately NOT trusted and are
not part of the receipt — they can be faked.

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
from gguf_tensors import inspect  # noqa: E402


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

    # 2) must be a real GGUF of the lane's architecture (anti-memorization gate)
    try:
        g = inspect(local)
    except AssertionError as e:
        raise ValueError(f"not a loadable GGUF: {e}")

    arch = g["architecture"]
    if arch != expected_arch:
        raise ValueError(
            f"architecture mismatch: file is {arch!r}, lane expects {expected_arch!r}")

    # 3) house-verified intake record — only what we measured/inspected
    return {
        "lane": lane,
        "quant": quant,
        "file": sub["file"],
        "source": source,
        "sha256": sha,
        "size_bytes": size,
        "architecture": arch,
        "n_tensors": g["n_tensors"],
        "tensors_by_type": g["tensors_by_type"],
        # note: full per-tensor map is available in g["tensors"]; keep the receipt
        # lean but the tensor-type histogram proves the objective quantization layout
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
