"""gate_from_ladder.py — set the KL acceptance threshold from a ladder.

The ladder measures known quants (Q8 / Q6 / Q4) against the stored reference.
The gate is **field-anchored** (per Keg doctrine, never derived from our own
measurements): the near-lossless boundary comes from the field's published KLD
range (~0.025–0.09, degradation "drastic" above ~0.1 — Unsloth, silent-zone,
llama-perplexity community). The ladder *confirms* the separation: the
crown contenders (Q8/Q6) should sit well below the line and the lossy control
(Q4) well above it.

This tool reports where each quant lands relative to the field line and flags
whether the line is a clean separator, so the gate is finalized from evidence
rather than assumed.

Input (--results):
  {"Q8_0": {"kl": 0.020, "kl_max_component": 0.030, "size_gb": 27.6}, ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Field near-lossless / degradation boundary (nats). Model+corpus relative, so
# a quant's absolute KLD vs the field's published values is indicative, not a
# hard point — but the degradation line is where quality drops "drastic".
FIELD_LINE = 0.10


def analyze(results: dict) -> dict:
    items = sorted(results.items(), key=lambda kv: kv[1].get("kl_max_component", kv[1].get("kl", float("inf"))))
    rows = []
    for name, r in items:
        kl = r.get("kl_max_component", r.get("kl"))
        rows.append({
            "quant": name,
            "kl": round(kl, 4),
            "size_gb": r.get("size_gb"),
            "relative_to_field_line": "below" if kl < FIELD_LINE else "above",
        })
    # clean separator: at least one quant below the line and at least one above
    below = [r for r in rows if r["relative_to_field_line"] == "below"]
    above = [r for r in rows if r["relative_to_field_line"] == "above"]
    clean = bool(below) and bool(above)
    return {
        "field_line": FIELD_LINE,
        "quants": rows,
        "below_line": [r["quant"] for r in below],
        "above_line": [r["quant"] for r in above],
        "clean_separator": clean,
        "recommended_accept_kl": FIELD_LINE,
        "verdict": (
            "gate = field line; confirmed by ladder" if clean else
            "gate still pending — ladder does not straddle the field line; "
            "investigate before finalizing"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text())
    print(json.dumps(analyze(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
