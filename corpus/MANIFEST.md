# Corpus

The fidelity corpus is **public, versioned, and diverse** — by design, not by
default. Trustlessness comes from the corpus being open: anyone can re-derive
the reference from the model over the same corpus. Anti-gaming comes from
breadth (a broad corpus can't be meaningfully overfit by calibration) and from
the "smallest faithful" metric (memorizing thousands of positions costs size).

## Seed corpus (`seed.txt`)

A small, varied dev/test corpus used by the machinery and the simulation. It is
**not** the production corpus — it is a stand-in so the code paths are runnable
offline. One line = one document; next-token positions are sampled across all
documents.

## Production corpus

The production reference must be measured over a **larger, public-domain,
multi-domain** corpus so the top-1 gate resolves to well under 1% error. Candidates:

- **Common Corpus** (Pleias) — the largest truly open public-domain dataset
  (500B words; English 180B; multilingual; reasoning-rich books). Ideal: fully
  open and auditable.
- **RedPajama / SlimPajama** — open reproduction of LLaMA training data:
  Wikipedia + GitHub (code) + arXiv + StackExchange + books + news.
- For glimmer-30b (multimodal, code-capable) include code and structured text
  alongside prose.

**Size target:** ~100k–500k tokens, sampled at a stride to yield on the order
of **3,000–10,000 next-token positions**. With N≈5,000, the standard error on a
top-1 match rate near 0.99 is ~0.2% — cleanly separating an accepted recipe
(≥99%) from a rejected one (~97%).

**Pinning:** the production corpus is pinned by its sha256 (recorded in every
receipt) and pulled pod-side; the exact text need not live in the repo. The
reference artifact stores only each sampled position's top-1 token, so it stays
small even for a large corpus.

## Why public (no secrecy machinery)

- **Trustless by construction** — no sealed corpus, no commit-reveal, no
  house authority. Anyone re-derives the reference.
- **Calibration overfitting is bounded** — gains from tuning the calibration
  dataset to a *broad* corpus are minor (the imatrix author calls them "very
  minor"; a published example moved WikiText-2 PPL only 6.92→6.83).
- **The size metric punishes the residual attack (memorization)** — encoding
  thousands of diverse positions into a recipe costs size, which loses the race.
