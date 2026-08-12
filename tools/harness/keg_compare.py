"""keg_compare.py — compute the fidelity report (KL divergence per component,
top-1 agreement) from two single-pass top-k artifacts.

Consumes:
  --ref <reference_topk.json>      {"positions": {str(pos): {token: logprob}}, ...}
  --sub <submission_topk.json>     {"positions": {str(pos): {token: logprob}}, ...}
  --components <components.json>   {str(pos): "prose"|"code"|"multilingual"|"technical"}

Outputs the same report shape as fidelity.measure_vs_reference so the gate,
receipts, and posting pipeline consume it unchanged:
  {top1_match, kl_mean, kl_max_component, kl_by_component, kl_p999, n}

KL(ref || sub) is computed over the union of the two deep top-k supports, with
a floor for tokens missing from either model's top-k. The gate keys on the
worst component (kl_max_component); top-1 is reported but not gated.

usage:
  keg_compare.py --ref reference_topk.json --sub quant_topk.json \
      --components components.json [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_LOG_FLOOR = -20.0


def _kld(ref: dict[str, float], sub: dict[str, float]) -> float:
    """KL(ref || sub) over the union of the two deep top-k supports."""
    tokens = set(ref) | set(sub)
    rp = {t: math.exp(ref.get(t, _LOG_FLOOR)) for t in tokens}
    sp = {t: math.exp(sub.get(t, _LOG_FLOOR)) for t in tokens}
    zr, zs = sum(rp.values()), sum(sp.values())
    if zr <= 0 or zs <= 0:
        return float("inf")
    return sum(
        (rp[t] / zr) * (math.log(rp[t] / zr) - math.log(sp[t] / zs))
        for t in tokens if rp[t] > 0 and sp[t] > 0)


def _argmax(logp: dict[str, float]) -> str:
    return max(logp, key=logp.get) if logp else ""


def compare(ref_path: str, sub_path: str, comp_path: str) -> dict:
    ref = json.loads(Path(ref_path).read_text()).get("positions", {})
    sub = json.loads(Path(sub_path).read_text()).get("positions", {})
    comps = json.loads(Path(comp_path).read_text())

    hits = total = 0
    klds: list[float] = []
    by_comp: dict[str, list[float]] = {}
    for pos, r in ref.items():
        s = sub.get(pos)
        if not s:
            continue
        total += 1
        if _argmax(s) == _argmax(r):
            hits += 1
        k = _kld(r, s)
        klds.append(k)
        by_comp.setdefault(comps.get(pos, "prose"), []).append(k)

    klds_sorted = sorted(klds)
    comp_means = {c: sum(v) / len(v) for c, v in by_comp.items() if v}
    return {
        "top1_match": hits / total if total else 0.0,
        "kl_mean": sum(klds) / len(klds) if klds else float("inf"),
        "kl_max_component": max(comp_means.values()) if comp_means else float("inf"),
        "kl_by_component": comp_means,
        "kl_p999": (klds_sorted[min(len(klds_sorted) - 1, int(0.999 * len(klds_sorted)))]
                    if klds else float("inf")),
        "n": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--components", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    report = compare(args.ref, args.sub, args.components)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
