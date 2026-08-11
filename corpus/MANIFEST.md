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
reference artifact stores each sampled position's top-k log-probs, so it stays
small even for a large corpus.

## Why public (no secrecy machinery)

- **Trustless by construction** — no sealed corpus, no commit-reveal, no
  house authority. Anyone re-derives the reference.
- **Calibration overfitting is bounded** — gains from tuning the calibration
  dataset to a *broad* corpus are minor (the imatrix author calls them "very
  minor"; a published example moved WikiText-2 PPL only 6.92→6.83).
- **The size metric punishes the residual attack (memorization)** — encoding
  thousands of diverse positions into a recipe costs size, which loses the race.

## References

The gate uses the two metrics production serving stacks already trust, so we
build on the field rather than claim a novel metric:

- **Top-1 agreement + KL divergence vs the reference model** is the standard
  fidelity signal. sparkinfer's SN74 eval gates on exactly this pair
  (`top-1 ≥ 0.90` AND `KL ≤ 0.20` vs llama.cpp), and providers such as
  Fireworks and most GGUF serving tools treat KL as the primary quality
  signal. llama.cpp's community converged on "KL + answer-flip rate vs FP16"
  as more informative than perplexity.

- **Helcig, Kurtic, Alistarh, *"Statistically-Lossless Quantization of Large
  Language Models"*, arXiv:2605.02404 (2026)** — formalizes **EAR (Expected
  Acceptance Rate)** = token-agreement probability between the original and
  quantized model, explicitly framing "EAR ≥ 0.99 means 99% agreement." This
  is the same top-1 agreement we use as the primary metric, open-sourced at
  `github.com/IST-DASLab/SLQ`. Supporting citation, not the anchor.

- **Nikolić, Zadeh, Torres, Moshovos, *"Displacement Is Not Direction:
  Evaluating Fidelity Metrics for Quantized LLM Deployment"*, arXiv:2606.19558
  (2026).** Shows that *every* distributional fidelity metric — KLD, perplexity,
  **and top-1 agreement alike** — loses its correlation with downstream
  benchmark quality inside the near-baseline "silent zone" (the collapse is
  metric-invariant; e.g. top-1 full-cohort ρ≈+0.70 → silent-zone ρ≈0). In their
  cohort the silent zone spans Q8_0 down through Q4_K_M (composite ~0.69–0.70),
  with Q2_K_L the lossy boundary (~0.635).

  This supports Keg in two ways. First, because fidelity can no longer *rank*
  near-lossless quants by quality once they're accepted, the only meaningful
  discriminator among accepted recipes is **size** — which is exactly what Keg
  races on ("smallest faithful wins"). Second, our ≥99% top-1 gate coincides
  with the silent-zone boundary: it admits the near-lossless cluster (Q4→Q8)
  and rejects the lossy tier (Q2-class), which is the desired behavior. Keg
  does not rank within the zone, so the collapse is irrelevant to its gate.


