#!/usr/bin/env python3
"""Keg — stage the first crown + receipts from ladder fidelity results.

Consumes a JSON file mapping quant -> {top1, kl, size_bytes, sha256, file}
(one entry per ladder quant measured against the reference), decides
acceptance via the real `recipe.accepted`, and emits:
  - a receipt for every accepted quant (house baselines), and
  - the crown (smallest accepted) + the board.md rows to post.

Usage: python3 tools/post_results.py ladder_results.json [--out dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keg.recipe import Recipe, accepted  # noqa: E402
from keg.receipt import build_receipt  # noqa: E402

EPOCH = "2026-08-11"  # first real run
MODEL = "muse-glimmer-30b"
RUNTIME = "llama.cpp"


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("lanes/glimmer-30b/receipts")
    results = json.loads(src.read_text())

    ok = {q: r for q, r in results.items() if accepted(r["top1"], r.get("kl"))}
    print(f"accepted quants: {list(ok) if ok else '(none)'}")
    if not ok:
        print("no accepted quants — no crown yet")
        return 1

    crown = min(ok, key=lambda q: ok[q]["size_bytes"])
    print(f"crown = {crown} ({ok[crown]['size_bytes']/1e9:.1f}GB, "
          f"top1={ok[crown]['top1']:.4f}, kl={ok[crown]['kl']:.4f})")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in sorted(ok, key=lambda x: ok[x]["size_bytes"]):
        r = ok[q]
        recipe = Recipe(
            model=MODEL, model_file=r["file"], model_sha256=r["sha256"],
            quant=q, format="gguf", runtime=RUNTIME,
        )
        rec = build_receipt(
            recipe, {"top1_match": r["top1"], "kl_mean": r["kl"]},
            r["size_bytes"], epoch=EPOCH,
        )
        # NOTE: do NOT mutate `rec` after build_receipt — the receipt_sha256 is
        # computed over the final dict, and any post-hoc change breaks replay.
        (out / f"{q.lower()}.receipt.json").write_text(json.dumps(rec, indent=2))
        is_crown = "  <- CROWN" if q == crown else ""
        rows.append(f"| {q} | {r['size_bytes']/1e9:.1f} | "
                    f"{rec['fidelity']['top1_match']:.3f} | "
                    f"{'accepted' if rec['accepted'] else 'rejected'}{is_crown} |")
        print(f"  wrote {q.lower()}.receipt.json")

    print("\nboard.md rows (accepted, size-sorted, crown last):")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
