"""Keg — the gate: fidelity to the model reference.

A recipe is measured against the model's own BF16 reference over a fixed set
of sampled next-token positions drawn from a PUBLIC, diverse corpus, using the
metrics production serving stacks actually use:

- **KL divergence** (primary) — distribution shift vs the reference, bounded
  over a deep top-k. The field-standard fidelity metric (llama-perplexity,
  Fireworks, "Accuracy is Not All You Need"): it measures how much the quant's
  token distribution drifts from the base model's, and is highly correlated
  with answer flips.
- **top-1 agreement** (reported, not gated) — does the recipe's most-likely
  next token match the true model's? Human-readable companion to KLD.

Measurement matches the field's *long-mode* convention (llama-perplexity /
mlx-kld): positions are scored with a long preceding context (a ~2048-token
window), not short prefixes. Short prefixes flatter quants because early-
context predictions are mostly trivial; long context gives a realistic read.

Trustless by construction: the corpus is public and hashed, so anyone can
re-derive the reference from the model — no sealed corpus, no house authority.
Anti-gaming comes from the design, not secrecy: a broad corpus can't be
meaningfully overfit by calibration, and the "smallest faithful" metric
punishes memorization.

The reference artifact stores each sampled position's deep top-k log-probs,
so the gate replays from a compact, re-derivable artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests

CORPUS_VERSION = 3

_DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "seed.txt"

# Next-token positions to sample for the gate. Long-context positions are
# informative, so fewer suffice than the short-prefix design; N ~ 2500 keeps
# the gate's standard error well under 1% while keeping per-pass time sane.
DEFAULT_N = 2500

# Depth of the per-position top-k. Deep enough that the bounded KL closely
# approximates full-vocabulary KL: smcleod's mlx-kld measures K=1024 at ~2.3%
# error vs the dense (full-vocab) value, and the ordering is rank-preserving
# even at much smaller k.
TOP_K = 1024
_LOG_FLOOR = -20.0  # log-prob for tokens not in the returned top-k

# Long-context window (in characters, ~4 chars/token) scored before each
# position, matching the field's long-mode (llama-perplexity / mlx-kld use
# ~2048-token contexts). Positions nearer than this to the stream start are
# skipped (no long context available).
_CTX_CHARS = 8192  # ~2048 tokens


def load_corpus(path: str | None = None) -> List[str]:
    """Documents from the corpus file (one non-empty line = one document)."""
    path = path or os.environ.get("KEG_CORPUS_FILE") or str(_DEFAULT_CORPUS)
    docs = [l.rstrip() for l in Path(path).read_text().splitlines() if l.strip()]
    if not docs:
        raise ValueError(f"empty corpus: {path}")
    return docs


def _corpus_sha(corpus: List[str]) -> str:
    return hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16]


def build_stream(corpus: List[str]) -> str:
    """The corpus as one continuous text stream (docs joined), the basis for
    long-context next-token positions. Deterministic from the corpus."""
    return "\n\n".join(corpus)


def sample_positions(corpus: List[str], n: int) -> List[int]:
    """Deterministic set of char offsets in the stream with long context.

    Each offset has >= _CTX_CHARS of preceding text, so the prompt is a long,
    non-trivial context window (the field's long-mode). A pure function of the
    corpus (seeded by its hash), so the reference and every submission measure
    the exact same points — reproducible by anyone.
    """
    stream = build_stream(corpus)
    if len(stream) <= _CTX_CHARS + 2:
        # too short for long context; fall back to shallow offsets
        return list(range(_CTX_CHARS, len(stream) - 1))[:n]
    rng = random.Random(_corpus_sha(corpus))
    lo, hi = _CTX_CHARS, len(stream) - 1
    # sample with a modest step to spread positions across the stream, then
    # jitter deterministically so the set isn't trivially periodic
    step = max(1, (hi - lo) // max(1, n))
    offsets = [o for o in range(lo, hi, step)]
    if len(offsets) > n:
        rng.shuffle(offsets)
        offsets = offsets[:n]
    return sorted(offsets)


def _topk(base_url: str, prompt: str, top_k: int = TOP_K) -> Dict[str, float]:
    """The model's top-k next-token log-probs for a prefix, OpenAI-compatible."""
    r = requests.post(
        f"{base_url}/completions",
        json={"prompt": prompt, "max_tokens": 1, "logprobs": top_k,
              "echo": False, "temperature": 0.0},
        timeout=180,
    )
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]
    content = lp.get("content") or []
    tops = (content[0].get("top_logprobs") or []) if content else []
    out: Dict[str, float] = {}
    for e in tops:
        if "token" in e and "logprob" in e:
            out[e["token"]] = float(e["logprob"])
    return out


def _argmax(logp: Dict[str, float]) -> str:
    return max(logp, key=logp.get) if logp else ""


def _kld(ref_logp: Dict[str, float], sub_logp: Dict[str, float]) -> float:
    """KL(ref || sub) over the union of the two deep top-k supports.

    Tokens missing from a side are floored. Deep top-k makes this a close
    approximation of full-vocabulary KLD. Direction matches the field
    standard (weight divergence by the reference's probabilities).
    """
    tokens = set(ref_logp) | set(sub_logp)
    rp = {t: math.exp(ref_logp.get(t, _LOG_FLOOR)) for t in tokens}
    sp = {t: math.exp(sub_logp.get(t, _LOG_FLOOR)) for t in tokens}
    zr = sum(rp.values())
    zs = sum(sp.values())
    if zr <= 0 or zs <= 0:
        return float("inf")
    kld = 0.0
    for t in tokens:
        r, s = rp[t] / zr, sp[t] / zs
        if r > 0 and s > 0:
            kld += r * (math.log(r) - math.log(s))
    return kld


def _prompt_for(stream: str, offset: int) -> str:
    """The long-context window immediately before `offset` (the field's
    long-mode: a ~_CTX_CHARS window ending at the position)."""
    return stream[max(0, offset - _CTX_CHARS):offset]


def save_reference(url: str, out: str, corpus: List[str] | None = None,
                   n: int = DEFAULT_N) -> str:
    """Probe the model (the BF16 reference) over the sampled long-context
    positions and store the artifact: {offset: {token: logprob}}. Hash-bound."""
    corpus = corpus or load_corpus()
    stream = build_stream(corpus)
    positions = sample_positions(corpus, n)
    raw: Dict[str, Dict[str, float]] = {}
    for off in positions:
        try:
            lp = _topk(url, _prompt_for(stream, off))
            if lp:
                raw[str(off)] = lp
        except requests.HTTPError:
            raise
        except Exception:
            continue
    artifact = {
        "schema": "keg/reference-v2",
        "corpus_version": CORPUS_VERSION,
        "corpus_sha256": _corpus_sha(corpus),
        "top_k": TOP_K,
        "ctx_chars": _CTX_CHARS,
        "n": len(raw),
        "positions": raw,
    }
    Path(out).write_text(json.dumps(artifact, indent=2))
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[:16]


def load_reference(path: str) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema") != "keg/reference-v2":
        raise ValueError("not a keg reference-v2 artifact")
    return artifact


def measure_vs_reference(reference_path: str, submission_url: str,
                         corpus: List[str] | None = None) -> dict:
    """Fidelity of a submission vs the stored BF16 reference: KL divergence
    (primary) and top-1 agreement (reported), over the same long-context
    positions. Deterministic, no LLM in the loop."""
    corpus = corpus or load_corpus()
    stream = build_stream(corpus)
    artifact = load_reference(reference_path)
    positions = sample_positions(corpus, artifact["n"])
    refs = artifact["positions"]
    hits = 0
    total = 0
    klds: List[float] = []
    for off in positions:
        ref = refs.get(str(off))
        if ref is None:
            continue
        try:
            sub = _topk(submission_url, _prompt_for(stream, off))
        except requests.HTTPError:
            raise
        except Exception:
            continue
        if not sub:
            continue
        total += 1
        if _argmax(sub) == _argmax(ref):
            hits += 1
        klds.append(_kld(ref, sub))
    klds_sorted = sorted(klds)
    report = {
        "top1_match": hits / total if total else 0.0,
        "kl_mean": sum(klds) / len(klds) if klds else float("inf"),
        "kl_p999": (klds_sorted[min(len(klds_sorted) - 1,
                                    int(0.999 * len(klds_sorted)))]
                    if klds else float("inf")),
        "n": total,
        "corpus_version": artifact["corpus_version"],
        "corpus_sha256": artifact["corpus_sha256"],
        "reference_sha256": hashlib.sha256(
            Path(reference_path).read_bytes()).hexdigest()[:16],
    }
    return report
