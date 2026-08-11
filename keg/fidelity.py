"""Keg — the gate: fidelity to the model reference.

A recipe is measured against the model's own BF16 reference over a fixed set
of sampled positions drawn from a PUBLIC, diverse corpus, using the two
metrics production serving stacks actually use:

- **top-1 agreement** (primary) — does the recipe's most-likely next token
  match the true model's? This is the direct "is the model recognizably
  itself" signal (equivalently, the EAR / answer-flip rate).
- **KL divergence** (secondary) — distribution shift vs the reference,
  bounded over the matched top-k (the pattern used by sparkinfer's eval and
  by GGUF serving tools), catching tail/drift that top-1 alone can miss.

Trustless by construction: the corpus is public and hashed, so anyone can
re-derive the reference from the model — no sealed corpus, no house
authority. Anti-gaming comes from the design, not secrecy:

- a broad corpus can't be meaningfully overfit by calibration, and
- the "smallest faithful" metric punishes memorization (encoding thousands
  of positions into a recipe costs size, which loses the race).

The reference artifact stores each sampled position's top-k log-probs, so
both metrics replay from a small artifact.
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

# Next-token positions to sample for the gate. With N ~ 5000 the standard
# error near p=0.99 is ~0.2% — cleanly separating accept/reject.
DEFAULT_N = 5000
_STRIDE = 64  # char stride for candidate positions within a document

# Top-k depth stored for the reference and used for the bounded KL. Deep
# enough that the submission's top token is almost always covered (sparkinfer
# dumps a deep top-k so the tail isn't floored).
TOP_K = 32
_LOG_FLOOR = -20.0  # log-prob for tokens not in the returned top-k


def load_corpus(path: str | None = None) -> List[str]:
    """Documents from the corpus file (one non-empty line = one document)."""
    path = path or os.environ.get("KEG_CORPUS_FILE") or str(_DEFAULT_CORPUS)
    docs = [l.rstrip() for l in Path(path).read_text().splitlines() if l.strip()]
    if not docs:
        raise ValueError(f"empty corpus: {path}")
    return docs


def _corpus_sha(corpus: List[str]) -> str:
    return hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16]


def sample_positions(corpus: List[str], n: int) -> List[Tuple[int, int]]:
    """Deterministic set of (doc_index, char_offset) next-token positions.

    A pure function of the corpus (seeded by its hash), so the reference and
    every submission measure the exact same points — reproducible by anyone.
    """
    rng = random.Random(_corpus_sha(corpus))
    candidates: List[Tuple[int, int]] = []
    for di, doc in enumerate(corpus):
        for off in range(_STRIDE, len(doc) - 8, _STRIDE):
            candidates.append((di, off))
    if len(candidates) > n:
        rng.shuffle(candidates)
        candidates = candidates[:n]
    return sorted(candidates)


def _pos_key(di: int, off: int) -> str:
    return f"{di}:{off}"


def _topk(base_url: str, prompt: str, top_k: int = TOP_K) -> Dict[str, float]:
    """The model's top-k next-token log-probs for a prefix, OpenAI-compatible."""
    r = requests.post(
        f"{base_url}/completions",
        json={"prompt": prompt, "max_tokens": 1, "logprobs": top_k,
              "echo": False, "temperature": 0.0},
        timeout=120,
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
    """Bounded KL(ref || sub) over the union of the two top-k supports.

    Tokens missing from a side are floored. Matched-depth so the reference
    tail isn't floored (the sparkinfer pattern). Base is whatever the server
    returns; consistent across ref and submission, so the value is comparable
    and the threshold is set by the calibration ladder.
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


def save_reference(url: str, out: str, corpus: List[str] | None = None,
                   n: int = DEFAULT_N) -> str:
    """Probe the model (the BF16 reference) over the sampled positions and
    store the artifact: {position_key: {token: logprob}}. Hash-bound."""
    corpus = corpus or load_corpus()
    positions = sample_positions(corpus, n)
    raw: Dict[str, Dict[str, float]] = {}
    for di, off in positions:
        try:
            lp = _topk(url, corpus[di][:off])
            if lp:
                raw[_pos_key(di, off)] = lp
        except requests.HTTPError:
            raise
        except Exception:
            continue
    artifact = {
        "schema": "keg/reference-v1",
        "corpus_version": CORPUS_VERSION,
        "corpus_sha256": _corpus_sha(corpus),
        "top_k": TOP_K,
        "n": len(raw),
        "positions": raw,
    }
    Path(out).write_text(json.dumps(artifact, indent=2))
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[:16]


def load_reference(path: str) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema") != "keg/reference-v1":
        raise ValueError("not a keg reference artifact")
    return artifact


def measure_vs_reference(reference_path: str, submission_url: str,
                         corpus: List[str] | None = None) -> dict:
    """Fidelity of a submission vs the stored BF16 reference: top-1 agreement
    (primary) and bounded KL (secondary), over the same sampled positions.

    Deterministic, no LLM in the loop. Returns top1_match (0..1), kl_mean,
    kl_p999, the sample count, and the corpus + reference hashes so receipts
    bind to the exact yardstick.
    """
    corpus = corpus or load_corpus()
    artifact = load_reference(reference_path)
    positions = sample_positions(corpus, artifact["n"])  # same points as ref
    refs = artifact["positions"]
    hits = 0
    total = 0
    klds: List[float] = []
    for di, off in positions:
        key = _pos_key(di, off)
        ref = refs.get(key)
        if ref is None:
            continue
        try:
            sub = _topk(submission_url, corpus[di][:off])
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
