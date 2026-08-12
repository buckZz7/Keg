#!/usr/bin/env python3
"""Keg — stage the first crown + receipts from ladder fidelity results.

Consumes a JSON file mapping quant -> {top1, kl, size_bytes, sha256, file}
(one entry per ladder quant measured against the reference), decides
acceptance via the real `recipe.accepted`, and emits:
  - a receipt for every accepted quant (house baselines),
  - the crown (smallest accepted), and
  - the site's docs/board.json (crown + leaderboard), so the website updates.

Usage:
  python3 tools/post_results.py ladder_results.json [--out lanes/glimmer-30b/receipts]
      [--ref lanes/glimmer-30b/reference/reference.json] [--board docs/board.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keg.recipe import Recipe, accepted  # noqa: E402
from keg.receipt import build_receipt  # noqa: E402

EPOCH = "2026-08-11"  # first real run
MODEL = "muse-glimmer-30b"
RUNTIME = "llama.cpp"


def artifact_meta(ref_path: Path) -> dict:
    d = json.loads(ref_path.read_text())
    return {
        "sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest()[:16],
        "corpus_sha256": d.get("corpus_sha256"),
        "corpus_version": d.get("corpus_version"),
        "top_k": d.get("top_k"),
        "n": d.get("n"),
        "generated": EPOCH,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="ladder results JSON")
    ap.add_argument("--out", default="lanes/glimmer-30b/receipts")
    ap.add_argument("--ref", default="lanes/glimmer-30b/reference/reference.json")
    ap.add_argument("--board", default="docs/board.json")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    out = Path(args.out)
    ref_path = Path(args.ref)
    board_path = Path(args.board)

    ok = {q: r for q, r in results.items() if accepted(r.get("kl"), r.get("kl_max_component"))}
    print(f"accepted quants: {list(ok) if ok else '(none)'}")
    if not ok:
        print("no accepted quants — no crown yet")
        return 1

    crown = min(ok, key=lambda q: ok[q]["size_bytes"])
    print(f"crown = {crown} ({ok[crown]['size_bytes']/1e9:.1f}GB, "
          f"top1={ok[crown]['top1']:.4f}, kl={ok[crown]['kl']:.4f})")

    out.mkdir(parents=True, exist_ok=True)
    board = {
        "lane": MODEL,
        "reference": artifact_meta(ref_path),
        "crown": {"quant": crown},
        "leaderboard": [],
        "receipts": [],
    }
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
        is_crown = "  <- current best" if q == crown else ""
        rows.append(f"| {q} | {r['size_bytes']/1e9:.1f} | "
                    f"{rec['fidelity']['top1_match']:.3f} | "
                    f"{'accepted' if rec['accepted'] else 'rejected'}{is_crown} |")
        board["leaderboard"].append({
            "quant": q,
            "size_gb": round(r["size_bytes"] / 1e9, 1),
            "top1": round(r["top1"], 4),
            "accepted": True,
            "receipt": f"{q.lower()}.receipt.json",
        })
        print(f"  wrote {q.lower()}.receipt.json")

    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_text(json.dumps(board, indent=2))
    print(f"  wrote {board_path}")

    print("\nboard.md rows (accepted, size-sorted, current best last):")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
