#!/usr/bin/env python3
"""Keg — PLAY MINER. Simulate the miner experience end-to-end, offline.

Walks the REAL keg machinery (public-corpus reference, fidelity gate,
receipts, crown dominance) against fidelity-aware mock model servers, so you
can feel what submitting a recipe is like — accepted, crowned, dethroned,
rejected — without a GPU.

This is a SIMULATION. Numbers are synthetic; the flow and receipts are real
code paths. Nothing here touches a lane's real board.

usage: python3 tools/play_miner.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keg.fidelity import load_corpus, measure_vs_reference, save_reference
from keg.receipt import build_receipt, verify_receipt
from keg.recipe import Recipe, accepted
from keg.run_race import crown_decision

# ---- fidelity-aware mock server (llama.cpp logprobs shape) ----
TOKENS = [" alpha", " beta", " gamma", " delta", " epsilon",
          " zeta", " eta", " theta", " iota", " kappa"]
_LOGITS = [3.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
_Z = max(_LOGITS)
_EXPS = [math.exp(l - _Z) for l in _LOGITS]
_LOGSUMEXP = math.log(sum(_EXPS)) + _Z


def _log_softmax(i: int) -> float:
    return _LOGITS[i] - _LOGSUMEXP


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, match_prob: float = 1.0, **kw):
        self.match_prob = match_prob
        super().__init__(*args, **kw)

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("prompt", "")
        logprobs = body.get("logprobs")
        if logprobs:
            # Deterministic per prompt: slot-0 token matches the reference
            # (" alpha") with probability match_prob, else differs.
            rnd = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
            slot0 = " alpha" if rnd.random() < self.match_prob else " beta"
            top = []
            for i, t in enumerate(TOKENS):
                tok = slot0 if i == 0 else t
                top.append({"token": tok, "logprob": _log_softmax(i),
                            "id": i, "bytes": list(tok.encode())})
            choices = [{"index": 0,
                        "logprobs": {"content": [{"top_logprobs": top}]},
                        "text": ""}]
        else:
            choices = [{"index": 0, "text": " word" * body.get("max_tokens", 1),
                        "logprobs": None}]
        resp = {"choices": choices, "usage": {"completion_tokens": 1}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_server(match_prob: float = 1.0):
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              partial(_Handler, match_prob=match_prob))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def url(srv):
    return f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> int:
    sim = Path("/tmp/keg_sim"); sim.mkdir(exist_ok=True)
    ref_artifact = sim / "reference.json"
    corpus = load_corpus()  # the public seed corpus
    N = 150                 # samples for the offline sim (real run uses ~5000)

    print("=" * 64)
    print("KEG — PLAY MINER  (simulation; real code paths, public corpus)")
    print("=" * 64)

    ref = start_server(match_prob=1.0)
    try:
        ref_sha = save_reference(url(ref), str(ref_artifact), corpus=corpus, n=N)
        print(f"\n[house] derived the lane's BF16 reference over the public corpus "
              f"({N} sampled positions, sha256={ref_sha})")
    finally:
        ref.shutdown()

    submissions = [
        ("alpha miner", "Q8_0",   0.996, 32.1, "gguf"),
        ("beta miner",  "NVFP4",  0.994, 30.2, "nvfp4"),
        ("gamma miner", "Q6_K",   0.991, 23.4, "gguf"),
        ("delta miner", "Q4_K_M", 0.974, 17.1, "gguf"),
        ("epsilon miner", "Q2_K", 0.915, 11.0, "gguf"),
    ]

    king = None
    print(f"\n{'miner':<14}{'quant':<9}{'size':>6}{'fid':>7}  verdict")
    print("-" * 60)

    for name, quant, fid, gb, fmt in submissions:
        sub = start_server(match_prob=fid)
        try:
            recipe = Recipe(
                model="muse-glimmer-30b",
                model_file=f"muse-glimmer-30b-{quant}.gguf",
                model_sha256=f"sim-{quant}",
                quant=quant, format=fmt, runtime="llama.cpp",
                note=f"{name} — simulating {quant}",
            )
            size_bytes = int(gb * (1024 ** 3))
            fidelity = measure_vs_reference(str(ref_artifact), url(sub), corpus=corpus)
            measured = fidelity.get("top1_match", 0.0)

            if not accepted(measured):
                receipt = build_receipt(recipe, fidelity, size_bytes, epoch="sim")
                print(f"{name:<14}{quant:<9}{gb:>6.1f}{measured:>7.3f}  REJECTED "
                      f"(holds {measured*100:.1f}% < 99%)")
                print(f"  {'':22}receipt valid={verify_receipt(receipt)} "
                      f"sha={receipt['receipt_sha256'][:10]}")
                continue

            receipt = build_receipt(recipe, fidelity, size_bytes, epoch="sim")
            dethroned, reason = crown_decision(receipt, king)
            if dethroned:
                king = receipt
                verdict = "CROWN"
            else:
                verdict = f"no crown ({reason})"
            print(f"{name:<14}{quant:<9}{gb:>6.1f}{measured:>7.3f}  ACCEPTED -> {verdict}")
            print(f"  {'':22}receipt valid={verify_receipt(receipt)} "
                  f"sha={receipt['receipt_sha256'][:10]}")
        finally:
            sub.shutdown()

    print("-" * 60)
    print("\nFINAL BOARD — crown (smallest accepted recipe):")
    if king:
        k = king
        print(f"  {k['recipe']['quant']:<9}{k['size']['size_gb']:>6.2f} GB   "
              f"top-1={k['fidelity']['top1_match']:.3f}   "
              f"holder={k['recipe']['note']}")
    else:
        print("  (no crown)")
    print("\n(simulation complete — no lane board was touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
