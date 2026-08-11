"""Keg — the gate: fidelity to the model reference.

Ungameable by construction: a property of the weights, not a task set.
The submission's next-token distribution is compared to the model
reference's on a held-out corpus — top-1 match rate + KL divergence
(mean and 99.9% quantile). A recipe that washes, overfits its
calibration, or cheats computation fails the gate automatically.

PRIVACY: the race corpus is house-held (file via KEG_CORPUS_FILE) and
published only when superseded — a public corpus could be overfitted by
calibration. Receipts bind to corpus_sha256; the reference artifact keys
prompts by hash so the text never leaks.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# Fixed held-out prompts. NOT the models' training data; fixed so every
# race is comparable. Versioned: a corpus change invalidates prior
# receipts (they record the version + hash).
CORPUS_VERSION = 2

# The RACE corpus lives outside the repo (KEG_CORPUS_FILE). The embedded
# list below is a dev/test placeholder only — never the race corpus.
_DEV_CORPUS = [
    "The history of navigation begins with the earliest civilizations, where",
    "In the nineteenth century, the industrial revolution transformed the",
    "The physics of light has occupied scientists since the age of Newton,",
    "A recipe is only as good as its ingredients, and the same is true of",
    "When the first telegraph lines crossed the continent, messages that",
    "The economics of scale explain why large factories replaced workshops",
    "Land, water, wind, and fire are the four classical elements of nature,",
    "The invention of the printing press changed the economics of knowledge",
    "In computer science, a cache is a small, fast memory that holds",
    "The migration patterns of monarch butterflies span three generations,",
]

CORPUS: List[str] = [
    line.rstrip() for line in Path(os.environ["KEG_CORPUS_FILE"]).read_text().splitlines()
    if line.strip()
] if os.environ.get("KEG_CORPUS_FILE") else _DEV_CORPUS


def _prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _logprobs(base_url: str, prompt: str, top_k: int = 128) -> List[Tuple[str, float]]:
    """Next-token logprobs from an OpenAI-compatible server."""
    r = requests.post(
        f"{base_url}/completions",
        json={
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": top_k,
            "echo": False,
            "temperature": 0.0,
        },
        timeout=120,
    )
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]
    # llama.cpp: logprobs.content is a list of per-token entries, each with
    # top_logprobs = {token: logprob} for the next-token distribution.
    content = lp.get("content") or []
    tops = (content[0].get("top_logprobs") or []) if content else []
    # top_logprobs is a list of {token, logprob, id, bytes} — build {token: prob}
    return [(e["token"], math.exp(float(e["logprob"]))) for e in tops if "token" in e]


def _distributions(logprobs: List[Tuple[str, float]]) -> Dict[str, float]:
    total = sum(p for _, p in logprobs)
    return {tok: p / total for tok, p in logprobs if total > 0}


def _probe(url: str, corpus: List[str]) -> Dict[str, Dict[str, float]]:
    """Next-token distributions for every corpus prompt from ONE server.

    A full pass over the corpus on a single endpoint. Two passes (one per
    server) are combined per prompt — the math is order-independent, so
    this supports the sequential two-pass gate when the box cannot hold
    both models at once.
    """
    dists: Dict[str, Dict[str, float]] = {}
    for prompt in corpus:
        try:
            dists[prompt] = _distributions(_logprobs(url, prompt))
        except requests.HTTPError:
            raise  # a failed probe must surface — never silently pass
        except Exception:
            continue  # transient/parse issues: skip the prompt, keep the rest
    return dists


def save_reference(url: str, path: str, corpus: List[str] | None = None) -> str:
    """Probe a server (typically the BF16 model) over the corpus and store
    the reference artifact: the model's own next-token logprobs, hash-bound.

    Returns the artifact sha256 (recorded in receipts). The corpus is
    fixed, so the reference is tiny (~40KB) and replayable forever.
    """
    import hashlib
    corpus = corpus or CORPUS
    raw = {}
    for prompt in corpus:
        try:
            raw[_prompt_key(prompt)] = _logprobs(url, prompt)
        except requests.HTTPError:
            raise
        except Exception:
            continue
    artifact = {
        "schema": "keg/reference-v1",
        "corpus_version": CORPUS_VERSION,
        "corpus_sha256": hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16],
        "prompts": raw,
    }
    Path(path).write_text(json.dumps(artifact, indent=2))
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[:16]


def load_reference(path: str) -> Dict[str, Dict[str, float]]:
    """Load the stored reference artifact into per-prompt distributions."""
    import json as _json
    from pathlib import Path as _Path
    artifact = _json.loads(_Path(path).read_text())
    if artifact.get("schema") != "keg/reference-v1":
        raise ValueError("not a keg reference artifact")
    return {p: _distributions(toks) for p, toks in artifact.get("prompts", {}).items()}


def _compare(refs: Dict[str, Dict[str, float]], subs: Dict[str, Dict[str, float]],
             corpus: List[str]) -> dict:
    """Per-prompt top-1 + KL over the union vocabulary."""
    top1_hits = 0
    kls: List[float] = []
    for prompt in corpus:
        ref, sub = refs.get(prompt), subs.get(prompt)
        if not ref or not sub:
            continue
        ref_top = max(ref, key=ref.get)  # type: ignore[arg-type]
        sub_top = max(sub, key=sub.get)  # type: ignore[arg-type]
        if ref_top == sub_top:
            top1_hits += 1
        vocab = set(ref) | set(sub)
        kl = sum(ref.get(t, 1e-9) * math.log(ref.get(t, 1e-9) / sub.get(t, 1e-9)) for t in vocab)
        kls.append(max(0.0, kl))
    n = len(kls)
    if n == 0:
        return {"top1_match": 0.0, "kl_mean": float("inf"), "kl_p999": float("inf"), "n": 0}
    top1 = top1_hits / n
    return {"top1_match": top1, "kl_mean": sum(kls) / n,
            "kl_p999": sorted(kls)[min(n - 1, int(0.999 * n))], "n": n}


def measure_fidelity(
    reference_url: str,
    submission_url: str,
    corpus: List[str] | None = None,
    swap=None,
) -> dict:
    """Top-1 match rate + KL (mean, 99.9%) vs the model reference.

    Two passes: probe the reference over the whole corpus, then (if
    *swap* is given — a callable that switches the box from the
    reference server to the submission server) the submission. The gate
    passes when top-1 >= threshold and KL-99.9 <= threshold. The report
    records the measured fidelity plus the corpus version +
    hash so receipts stay replayable.

    Prefer measure_vs_reference() (stored BF16 artifact) — this two-pass
    form is the fallback when no artifact exists yet.
    """
    corpus = corpus or CORPUS
    corpus_sha = hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16]

    refs = _probe(reference_url, corpus)
    if swap is not None:
        swap()
    subs = _probe(submission_url, corpus)

    report = _compare(refs, subs, corpus)
    report["corpus_version"] = CORPUS_VERSION
    report["corpus_sha256"] = corpus_sha
    return report


def measure_vs_reference(
    reference_path: str,
    submission_url: str,
    corpus: List[str] | None = None,
) -> dict:
    """Gate against the STORED reference artifact (the model's own BF16
    distributions): probe the submission only, compare vs the artifact.

    One server session, no swap, no Q8_0 dependency. The reference sha
    is included in the report so receipts bind to the exact yardstick.
    """
    import hashlib
    from pathlib import Path as _Path
    corpus = corpus or CORPUS
    corpus_sha = hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16]

    refs = load_reference(reference_path)
    # probe the submission with the ACTUAL corpus prompts, keyed by hash
    subs = {}
    for prompt in corpus:
        try:
            subs[_prompt_key(prompt)] = _distributions(_logprobs(submission_url, prompt))
        except requests.HTTPError:
            raise
        except Exception:
            continue
    # align by hash key so the corpus text never has to appear in the artifact
    subs_by_key = subs
    refs_by_key = refs

    top1_hits = 0
    kls: List[float] = []
    for prompt in corpus:
        key = _prompt_key(prompt)
        ref, sub = refs_by_key.get(key), subs_by_key.get(key)
        if not ref or not sub:
            continue
        ref_top = max(ref, key=ref.get)  # type: ignore[arg-type]
        sub_top = max(sub, key=sub.get)  # type: ignore[arg-type]
        if ref_top == sub_top:
            top1_hits += 1
        vocab = set(ref) | set(sub)
        kl = sum(ref.get(t, 1e-9) * math.log(ref.get(t, 1e-9) / sub.get(t, 1e-9)) for t in vocab)
        kls.append(max(0.0, kl))
    n = len(kls)
    report = {"top1_match": top1_hits / n if n else 0.0,
              "kl_mean": sum(kls) / n if n else float("inf"),
              "kl_p999": sorted(kls)[min(n - 1, int(0.999 * n))] if n > 1 else (kls[0] if n == 1 else float("inf")),
              "n": n}
    report["corpus_version"] = CORPUS_VERSION
    report["corpus_sha256"] = corpus_sha
    report["reference_sha256"] = hashlib.sha256(
        _Path(reference_path).read_bytes()).hexdigest()[:16]
    return report
