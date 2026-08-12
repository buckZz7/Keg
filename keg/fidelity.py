"""Keg — the gate: fidelity to the model reference.

A recipe is measured against the model's own BF16 reference over a fixed set
of sampled next-token positions drawn from a PUBLIC, diverse, versioned corpus,
using the metrics production serving stacks actually use:

- **KL divergence** (primary) — distribution shift vs the reference, bounded
  over a deep top-k. The field-standard fidelity metric (llama-perplexity,
  Fireworks, "Accuracy is Not All You Need"): highly correlated with answer
  flips, and the metric that separates near-lossless quants from lossy ones on
  a hard, diverse corpus.
- **top-1 agreement** (reported, not gated) — human-readable companion to KLD.

Measurement matches the field's *long-mode* convention (llama-perplexity /
mlx-kld): positions are scored with a long preceding context (a ~2048-token
window), not short prefixes. Short prefixes flatter quants because early-
context predictions are mostly trivial.

Anti-overfit design (this is what makes the gate hard to game):
- **Stratified sampling.** Positions are sampled *per component* (prose, code,
  multilingual, technical) with a floor, so no component is a thin weak spot a
  miner could overfit or ignore. Uniform sampling would under-represent the
  smaller code/multilingual components.
- **Per-component gate.** A recipe is accepted only if EVERY component's mean
  KL passes — not just the overall mean. This closes the "excel on the
  dominant component, let the rest fail" attack (smcleod: "a quant that holds
  up on prose but falls apart on code would look fine here" — unless you gate
  per component).
- **Large N.** N positions make the eval statistically representative of the
  corpus's behavior, so there's no gap between eval and true performance to
  overfit. An aggressive quant's intrinsic loss (it physically lacks the bits)
  shows through on diverse, representative data no matter how it's calibrated.

Trustless by construction: the corpus is public and hashed, so anyone can
re-derive the reference from the model — no sealed corpus, no house authority.
The reference artifact stores each sampled position's deep top-k log-probs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests

CORPUS_VERSION = 3

_DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "seed.txt"

# Next-token positions to sample for the gate. Large so the eval is a
# representative sample of the corpus's behavior (no overfit gap); each pass is
# ~N * ~1s of prefill on the eval box.
DEFAULT_N = 5000

# Minimum positions sampled from each component, so per-component gating is
# reliable even for the smaller code/multilingual components.
MIN_PER_COMPONENT = 200

# Depth of the per-position top-k. Deep enough that the bounded KL closely
# approximates full-vocabulary KL: smcleod's mlx-kld measures K=1024 at ~2.3%
# error vs the dense (full-vocab) value, rank-preserving.
TOP_K = 1024
_LOG_FLOOR = -20.0  # log-prob for tokens not in the returned top-k

# Long-context window (in characters, ~4 chars/token) scored before each
# position, matching the field's long-mode (~2048-token contexts).
_CTX_CHARS = 8192  # ~2048 tokens
_STRIDE = 64       # char stride for candidate positions within a document


def load_corpus(path: str | None = None) -> List[str]:
    """Documents from the corpus file (one non-empty line = one document)."""
    path = path or os.environ.get("KEG_CORPUS_FILE") or str(_DEFAULT_CORPUS)
    docs = [l.rstrip() for l in Path(path).read_text().splitlines() if l.strip()]
    if not docs:
        raise ValueError(f"empty corpus: {path}")
    return docs


def load_components(path: str | None = None) -> List[str]:
    """One component label per document (aligned with the corpus file order).
    Falls back to all 'prose' if no components file exists (e.g. seed corpus)."""
    path = path or os.environ.get("KEG_COMPONENTS_FILE") or ""
    if not path:
        cpath = Path(os.environ.get("KEG_CORPUS_FILE") or str(_DEFAULT_CORPUS))
        cpath = cpath.with_suffix(".components.txt")
        path = str(cpath)
    docs = load_corpus()
    try:
        labels = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return ["prose"] * len(docs)
    if len(labels) != len(docs):
        raise ValueError(f"components file {path}: {len(labels)} labels for {len(docs)} docs")
    return labels


def _corpus_sha(corpus: List[str]) -> str:
    return hashlib.sha256("\n".join(corpus).encode()).hexdigest()[:16]


def build_stream(corpus: List[str]) -> str:
    """The corpus as one continuous text stream (docs joined)."""
    return "\n\n".join(corpus)


def _doc_ranges(corpus: List[str]) -> List[Tuple[int, int, int]]:
    """(doc_index, start_char, end_char) for each doc in the stream."""
    ranges = []
    start = 0
    for i, doc in enumerate(corpus):
        end = start + len(doc)
        ranges.append((i, start, end))
        start = end + 2  # "\n\n" separator
    return ranges


def _component_for_offset(offset: int, ranges: List[Tuple[int, int, int]],
                          components: List[str]) -> str:
    for i, start, end in ranges:
        if start <= offset < end:
            return components[i]
    return "prose"


def sample_positions(corpus: List[str], n: int = DEFAULT_N,
                     components: List[str] | None = None) -> List[Tuple[int, str]]:
    """Deterministic, STRATIFIED set of (char_offset, component) positions.

    Each offset has >= _CTX_CHARS of preceding text (long mode). Sampling is
    per component with a floor (MIN_PER_COMPONENT) so every component is
    represented and no component is a thin weak spot. A pure function of the
    corpus + components (seeded by their hashes), so the reference and every
    submission measure the same points — reproducible by anyone.
    """
    components = components or load_components()
    stream = build_stream(corpus)
    if len(stream) <= _CTX_CHARS + 2:
        return [(o, "prose") for o in range(_CTX_CHARS, len(stream) - 1)][:n]

    ranges = _doc_ranges(corpus)
    rng = random.Random(_corpus_sha(corpus) + ":" + hashlib.sha256(
        "\n".join(components).encode()).hexdigest()[:16])

    # candidate offsets per component (offset >= CTX, not at stream end)
    by_comp: Dict[str, List[int]] = {}
    for i, start, end in ranges:
        comp = components[i]
        bucket = by_comp.setdefault(comp, [])
        for off in range(max(start, _CTX_CHARS), end - 1, _STRIDE):
            bucket.append(off)

    comps = sorted(by_comp)
    counts = {c: len(by_comp[c]) for c in comps}
    total = sum(counts.values())
    selected: List[Tuple[int, str]] = []
    for c in comps:
        k = max(MIN_PER_COMPONENT, round(n * counts[c] / total))
        bucket = list(by_comp[c])
        rng.shuffle(bucket)
        for off in bucket[:k]:
            selected.append((off, c))

    # cap to n, preserving component floors (drops the surplus proportionally)
    if len(selected) > n:
        surplus = len(selected) - n
        # drop surplus from the components that exceed their floor
        drop = []
        for c in comps:
            n_c = sum(1 for _, cc in selected if cc == c)
            over = n_c - MIN_PER_COMPONENT
            if over > 0:
                drop.extend([(off, cc) for off, cc in selected if cc == c][:over])
        rng.shuffle(drop)
        drop_set = set(drop[:surplus])
        selected = [(o, c) for o, c in selected if (o, c) not in drop_set]

    return sorted(selected)


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
    """KL(ref || sub) over the union of the two deep top-k supports."""
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
    """The long-context window immediately before `offset`."""
    return stream[max(0, offset - _CTX_CHARS):offset]


def save_reference(url: str, out: str, corpus: List[str] | None = None,
                   n: int = DEFAULT_N) -> str:
    """Probe the model (the BF16 reference) over the sampled long-context
    positions and store the artifact: {offset: {token: logprob}}. Hash-bound."""
    corpus = corpus or load_corpus()
    stream = build_stream(corpus)
    components = load_components()
    positions = sample_positions(corpus, n, components)
    raw: Dict[str, Dict[str, float]] = {}
    comps: Dict[str, str] = {}
    for off, comp in positions:
        try:
            lp = _topk(url, _prompt_for(stream, off))
            if lp:
                raw[str(off)] = lp
                comps[str(off)] = comp
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
        "components": comps,
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
    (primary, per-component), top-1 agreement (reported), over the same
    stratified long-context positions. Deterministic, no LLM in the loop."""
    corpus = corpus or load_corpus()
    stream = build_stream(corpus)
    components = load_components()
    artifact = load_reference(reference_path)
    positions = sample_positions(corpus, artifact["n"], components)
    refs = artifact["positions"]
    hits = 0
    total = 0
    klds: List[float] = []
    by_comp: Dict[str, List[float]] = {}
    for off, comp in positions:
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
        k = _kld(ref, sub)
        klds.append(k)
        by_comp.setdefault(comp, []).append(k)
    klds_sorted = sorted(klds)
    comp_means = {c: (sum(v) / len(v)) for c, v in by_comp.items() if v}
    report = {
        "top1_match": hits / total if total else 0.0,
        "kl_mean": sum(klds) / len(klds) if klds else float("inf"),
        "kl_max_component": max(comp_means.values()) if comp_means else float("inf"),
        "kl_by_component": comp_means,
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
