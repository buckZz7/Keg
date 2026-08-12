#!/usr/bin/env python3
"""Keg — run a compression race: measure a recipe against the model reference.

usage: run_race.py <reference_artifact.json> <submission_url> --recipe recipe.json
       --model-file <path> [--num-params 30e9] [--epoch 2026-08]
       [--king-receipt <king receipt.json>] [--out receipt.json]

The gate (fidelity vs the stored BF16 model reference) is deterministic and
involves no LLM. The race metric is SIZE — the house-measured footprint of
the model file. The crown moves only if a challenger is SMALLER than the
incumbent king AND accepted (>= 0.99 top-1): a smaller but lossier recipe
cannot take the crown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keg.fidelity import measure_vs_reference  # noqa: E402
from keg.receipt import build_receipt, verify_receipt  # noqa: E402
from keg.recipe import Recipe, accepted  # noqa: E402

# Container formats the house can load, serve, and reproduce. GGUF-only for
# launch (one runtime, airtight verification); extend as runtimes are added.
SUPPORTED_FORMATS = {"gguf"}
_GGUF_MAGIC = b"GGUF"
# Coarse floor: a real quant of a raced model is tens of GB; an answer-table
# memorizer is < 1 MB. The GGUF magic + serve-check below are authoritative.
_MIN_MODEL_BYTES = 1 * 1024 * 1024


def validate_model_file(path: str, fmt: str) -> str:
    """Reject anything that isn't a real model file of the lane's architecture.

    This is the anti-memorization gate: a lookup table / answer store is NOT a
    loadable model, so it fails here before any fidelity is measured.
      1. the container format is supported (currently gguf);
      2. the file exists and is plausibly a real model (not an answer table);
      3. the GGUF container header is valid.
    The house MUST ALSO load and serve the file with the standard runtime
    (llama.cpp) as the model — a file that won't load/serve as the model is
    not a submission.
    """
    if fmt.lower() not in SUPPORTED_FORMATS:
        return (f"unsupported format '{fmt}' "
                f"(supported: {', '.join(sorted(SUPPORTED_FORMATS))})")
    p = Path(path)
    if not p.exists():
        return f"model file not found: {path}"
    if p.stat().st_size < _MIN_MODEL_BYTES:
        return (f"model file implausibly small ({p.stat().st_size} bytes) — "
                f"not a real model (answer tables are rejected here)")
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic != _GGUF_MAGIC:
        return f"not a valid GGUF file (magic {magic!r}) — not a loadable model"
    return ""


def _king(receipt_path: str) -> dict:
    with open(receipt_path) as f:
        return json.load(f)


def crown_decision(challenger: dict, king: dict | None) -> tuple[bool, str]:
    """The crown moves only if the challenger is SMALLER AND as faithful.

    Both must use the same reference artifact / corpus, else not comparable.
    The challenger must be accepted (>= 0.99 top-1). Smaller-but-lossier does
    not dethrone.
    """
    cf = challenger.get("fidelity", {})
    cs = challenger.get("size", {})
    if not accepted(cf.get("kl_mean"), cf.get("kl_max_component")):
        return False, "below the acceptance bar (KL within bound in every component)"
    if king is None:
        return True, ""
    if (cf.get("corpus_version") != king["fidelity"].get("corpus_version")
            or cf.get("reference_sha256") != king["fidelity"].get("reference_sha256")):
        return False, "reference mismatch (not comparable)"
    king_size = king.get("size", {}).get("size_bytes", float("inf"))
    if cs.get("size_bytes", float("inf")) < king_size:
        return True, ""
    return False, "not smaller than the incumbent king"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one Keg compression race measurement.")
    ap.add_argument("reference_artifact", help="Stored BF16 reference artifact JSON (this lane's reference)")
    ap.add_argument("submission_url", help="Submission endpoint (the recipe under test)")
    ap.add_argument("--recipe", required=True, help="recipe.json (the submission)")
    ap.add_argument("--model-file", required=True, help="Path to the actual model file (house measures its size)")
    ap.add_argument("--num-params", type=float, default=None, help="Model parameter count (for bpw)")
    ap.add_argument("--king-receipt", default="", help="Current crown's receipt.json — enables the dominance rule")
    ap.add_argument("--epoch", default="2026-08", help="Release epoch (per model release)")
    ap.add_argument("--out", default="", help="Receipt output path")
    args = ap.parse_args()

    recipe = Recipe(**json.loads(Path(args.recipe).read_text()))

    # Anti-memorization gate: a submission must be a REAL, loadable model file
    # of the lane's architecture. A lookup table / answer store is not a model
    # and is rejected before any fidelity is measured.
    err = validate_model_file(args.model_file, recipe.format)
    if err:
        print(f"NOT A SUBMISSION: {err}")
        print("A recipe must be the actual model file, loadable and servable "
              "as the model (llama.cpp). An answer store is not a model.")
        return 2

    # House measures the TRUE model footprint — a miner's claim is advisory.
    size_bytes = Path(args.model_file).stat().st_size

    print(f"== compression race: {recipe.model} [{recipe.quant} / {recipe.format}] ==")
    print("gate: fidelity vs stored BF16 reference ...")
    fidelity = measure_vs_reference(args.reference_artifact, args.submission_url)
    ok_gate = accepted(fidelity.get("kl_mean"), fidelity.get("kl_max_component"))
    print(f"  top1_match={fidelity.get('top1_match'):.3f}  kl_mean={fidelity.get('kl_mean'):.4f}  "
          f"kl_p999={fidelity.get('kl_p999'):.4f}  (n={fidelity.get('n')} samples, "
          f"corpus_sha={fidelity.get('corpus_sha256')})")

    if not ok_gate:
        print(f"  REJECTED: top-1 < {0.99} or KL above bound — not recognizably itself "
              f"(needs top-1 >= {0.99} AND KL <= {0.20}). Not a valid submission; no reward.")
        # Still write a receipt so the attempt is on record, but it holds no crown.
        size_bytes = Path(args.model_file).stat().st_size
        receipt = build_receipt(recipe, fidelity, size_bytes, args.epoch, num_params=args.num_params)
        out = args.out or f"lanes/{recipe.model}/receipts/rejected_{receipt['receipt_sha256'][:16]}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(receipt, indent=2))
        print(f"receipt: {out}")
        return 0

    size_gb = size_bytes / (1024 ** 3)
    bpw = (size_bytes * 8 / args.num_params) if args.num_params else None
    print(f"  size={size_gb:.2f} GB ({size_bytes} bytes)"
          + (f", bpw={bpw:.2f}" if bpw else ""))

    crown = None
    if args.king_receipt:
        king = _king(args.king_receipt)
        dethroned, reason = crown_decision(fidelity, king)
        crown = {
            "dethroned": dethroned,
            "challenger_size_bytes": size_bytes,
            "king_size_bytes": king.get("size", {}).get("size_bytes"),
        }
        if reason:
            crown["hold_reason"] = reason
        verdict = "CROWN MOVED" if dethroned else "crown holds"
        print(f"  vs king ({king.get('size', {}).get('size_bytes')} bytes) -> {verdict}"
              + (f" ({reason})" if reason else ""))
    else:
        dethroned = True  # first accepted recipe becomes the crown
        crown = {"dethroned": True, "first_crown": True}
        print("  (no king yet — first accepted recipe becomes the crown)")

    receipt = build_receipt(recipe, fidelity, size_bytes, args.epoch,
                            num_params=args.num_params, crown=crown)
    ok = verify_receipt(receipt)
    print(f"accepted={receipt['accepted']} receipt_valid={ok} sha={receipt['receipt_sha256'][:16]}")

    out = args.out or f"lanes/{recipe.model}/receipts/receipt_{receipt['receipt_sha256'][:16]}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(receipt, indent=2))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
