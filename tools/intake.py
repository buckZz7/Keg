"""intake.py — record a model file's real size + sha256 at intake.

The receipt must bind the actual artifact (sha256) and its true footprint
(size_bytes) — never a miner's claim. This records them from either:
  --local  a local model file (computes sha256 + os.path.getsize),
  --hf     the HuggingFace LFS metadata (oid/sha256 + size).

Emits JSON mapping a quant label -> {file, sha256, size_bytes}, which is merged
with the fidelity results and fed to post_results.py to build the receipt.

usage:
  # local file
  python3 tools/intake.py --local "Q8_0=Muse-Glimmer-30B-Q8_0.gguf" --out intake.json
  # from HF
  python3 tools/intake.py --hf unsloth/Muse-Glimmer-30B-GGUF \
      --files "Q8_0=Muse-Glimmer-30B-Q8_0.gguf" "UD-Q6_K_XL=Muse-Glimmer-30B-UD-Q6_K_XL.gguf" \
      --out intake.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def _from_local(path: str) -> dict:
    p = Path(path)
    data = p.read_bytes()
    return {
        "file": p.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _from_hf(repo: str, filename: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tree = json.load(urllib.request.urlopen(req, timeout=60))
    for entry in tree:
        if entry.get("path") == filename:
            lfs = entry.get("lfs", {})
            return {
                "file": filename,
                "sha256": lfs.get("oid", ""),
                "size_bytes": entry.get("size", 0),
            }
    raise ValueError(f"{filename} not found in {repo}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", help="quant=path pairs")
    ap.add_argument("--hf", help="HF repo")
    ap.add_argument("--files", nargs="*", help="quant=filename pairs (with --hf)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    if args.local:
        for pair in args.local:
            q, path = pair.split("=", 1)
            out[q] = _from_local(path)
            print(f"{q}: {out[q]['file']} {out[q]['size_bytes']/1e9:.1f}GB "
                  f"sha={out[q]['sha256'][:12]}...")
    if args.hf and args.files:
        for pair in args.files:
            q, fn = pair.split("=", 1)
            out[q] = _from_hf(args.hf, fn)
            print(f"{q}: {out[q]['file']} {out[q]['size_bytes']/1e9:.1f}GB "
                  f"sha={out[q]['sha256'][:12]}...")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
