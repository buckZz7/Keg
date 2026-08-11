"""Keg — the gate: fidelity to the model reference (public-corpus edition).

A recipe is measured by TOP-1 NEXT-TOKEN AGREEMENT against the model's own
BF16 reference, over a fixed set of sampled positions drawn from a PUBLIC,
diverse corpus. Trustless by construction: the corpus is public and hashed,
so anyone can re-derive the reference from the model — no sealed corpus, no
house authority. Anti-gaming comes from the design, not secrecy:

- a broad corpus can't be meaningfully overfit by calibration, and
- the "smallest faithful" metric punishes memorization (encoding thousands
  of positions into a recipe costs size, which loses the race).

The reference artifact stores only each sampled position's top-1 token, so it
stays small even for a large corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests

CORPUS_VERSION = 3

_DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "seed.txt"

# Next-token positions to sample for the top-1 gate. With N ~ 5000 the
# standard error near p=0.99 is ~0.2% — cleanly separating accept/reject.
DEFAULT_N = 5000
_STRIDE = 64  # char stride for candidate positions within a document


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


def _top1(base_url: str, prompt: str) -> str:
    """The model's top-1 next token for a prefix, via an OpenAI-compatible API."""
    r = requests.post(
        f"{base_url}/completions",
        json={"prompt": prompt, "max_tokens": 1, "logprobs": 1,
              "echo": False, "temperature": 0.0},
        timeout=120,
    )
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]
    content = lp.get("content") or []
    tops = (content[0].get("top_logprobs") or []) if content else []
    for e in tops:
        if "token" in e:
            return e["token"]
    return ""


def save_reference(url: str, out: str, corpus: List[str] | None = None,
                   n: int = DEFAULT_N) -> str:
    """Probe the model (the BF16 reference) over the sampled positions and
    store the artifact: {position_key: top1_token}. Hash-bound. Returns sha."""
    corpus = corpus or load_corpus()
    positions = sample_positions(corpus, n)
    raw: Dict[str, str] = {}
    for di, off in positions:
        try:
            tok = _top1(url, corpus[di][:off])
            if tok:
                raw[_pos_key(di, off)] = tok
        except requests.HTTPError:
            raise
        except Exception:
            continue
    artifact = {
        "schema": "keg/reference-v1",
        "corpus_version": CORPUS_VERSION,
        "corpus_sha256": _corpus_sha(corpus),
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
    """Top-1 next-token agreement of the submission vs the stored BF16
    reference, over the same sampled positions. Deterministic, no LLM.

    Returns top1_match (0..1), the sample count, and the corpus + reference
    hashes so receipts bind to the exact yardstick.
    """
    corpus = corpus or load_corpus()
    artifact = load_reference(reference_path)
    positions = sample_positions(corpus, artifact["n"])  # same points as ref
    refs = artifact["positions"]
    hits = 0
    total = 0
    for di, off in positions:
        key = _pos_key(di, off)
        ref = refs.get(key)
        if ref is None:
            continue
        try:
            sub = _top1(submission_url, corpus[di][:off])
        except requests.HTTPError:
            raise
        except Exception:
            continue
        total += 1
        if sub == ref:
            hits += 1
    report = {
        "top1_match": hits / total if total else 0.0,
        "n": total,
        "corpus_version": artifact["corpus_version"],
        "corpus_sha256": artifact["corpus_sha256"],
        "reference_sha256": hashlib.sha256(
            Path(reference_path).read_bytes()).hexdigest()[:16],
    }
    return report
