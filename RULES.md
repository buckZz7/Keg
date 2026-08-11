# Keg — Rules

The smallest recipe that still *is* the model wins. A recipe is accepted — and
competes — only if it holds **≥99%** of the model's true behavior. Everything
is measured, nothing is claimed.

## The reference (the truth, public and re-derivable)

Every recipe is measured against **the model's own next-token behavior** —
generated once from the 16-bit (BF16) weights, over a fixed set of positions
sampled from a **public, diverse, versioned corpus**. Anyone can re-derive the
reference by running the model over the same corpus.

- **Public by design.** The corpus is open and hash-bound; the reference
  artifact stores only each sampled position's top-1 token. No sealed corpus,
  no house authority — trustlessness comes from openness, not secrecy.
- **Deterministic.** The sampled positions are a pure function of the corpus
  (seeded by its hash), so the reference and every submission measure the same
  points. Repro­ducible by anyone.
- **No LLM in the loop.** Fidelity is computed from token predictions, not
  judged by a model.

## The submission (a recipe)

A recipe is the exact model file (by sha256), its format/quant, and the runtime
that produced it. A recipe the house cannot reproduce is not a submission. The
race metric is **size** — the model's footprint — measured by the house from
the real file, never from the miner's word.

## The gate — acceptance

Fidelity = **top-1 next-token agreement** of the recipe vs the reference, over
the sampled positions. Deterministic. **A recipe is accepted only if it holds
≥0.99 top-1**; below that it is rejected.

The threshold is **calibrated by a ladder** — on a new model the house first
measures where the known quants (Q8 → Q2) land against the reference, then sets
the bar from data, not by fiat.

## The score — size

The house measures the model file's footprint in bytes (and bpw, normalized).
Smaller is better.

## The crown — dominance

A challenger takes a lane's crown **only if it is smaller than the incumbent
king AND accepted (≥0.99 top-1)**. A smaller-but-lossier recipe is rejected, not
rewarded. Both sides must share the same corpus and reference, else the crown
holds. There is no seed: the first accepted recipe establishes the crown;
anyone smaller takes it.

## Rewards

Rewards are listed per lane in `board.md` under **Rewards**. Today there is one:
the **crown** (smallest accepted recipe). More tiers can be added as rows in the
lane's Rewards table — no new machinery.

## Receipts

Every run produces one receipt: hash-bound (sha256 over all fields), replayable,
valid only against the current reference. It records the measured fidelity, the
house-measured size, the corpus + reference hashes, and the box fingerprint.
**Receipts are the source of truth for the board.** A receipt that doesn't
replay is not a receipt. Rejected attempts get a receipt too.

## Why this isn't gamed

- **Breadth kills calibration.** The corpus is broad and diverse; calibrating a
  quant to a broad corpus just makes a generally-better quant — the honest
  behavior. Narrow-corpus overfitting is bounded.
- **The size metric kills memorization.** Encoding thousands of diverse
  positions into a recipe costs size, which loses the race.
- **Public = verifiable.** Anyone re-derives the reference and checks a receipt.
