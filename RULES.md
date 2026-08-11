# Keg — Rules

The full mechanism. Short version: **smallest recipe that still holds the
model's behavior wins. Everything is measured, nothing is claimed.**

## The reference (the truth, and the referee)

Every recipe in a lane is measured against one yardstick: **the model's own
next-token distributions, generated once from the 16-bit (BF16) weights** over
a fixed held-out corpus, stored as a hash-bound artifact. Anyone can re-derive
it from the weights — the truth is a computation, not a held opinion.

- **Immutability.** The reference never changes while a lane is live. A change
  invalidates every prior receipt.
- **Non-trainability.** It is a property of the weights, not a task set. There
  is nothing to memorize, nothing to calibrate against.
- **Self-authentication.** Corpus + reference are hash-bound; the corpus text
  is house-held and never appears in the repo (a public corpus could be
  overfitted by calibration). Receipts bind to both hashes.
- **No LLM in the loop.** Fidelity is computed from token distributions, not
  judged by a model.

## The submission (a recipe)

A recipe is the exact model file (by sha256), its format/quant, and the runtime
that produced it. A recipe the house cannot reproduce is not a submission. The
race metric is **size** — the model's footprint — measured by the house from
the real file, never taken from the miner's word.

## The gate — fidelity

Fidelity = top-1 token match (+ KL) of the recipe's output distribution vs the
reference, over the held-out corpus. Deterministic. Bands are MEASURED, not
named — a recipe lands where its measurement puts it, whatever the format
claims:

| Band | Top-1 vs reference | Meaning |
|---|---|---|
| **A** | ≥ 0.99 | indistinguishable (Q6 / Q8 / FP8 class) |
| B | 0.97–0.99 | acceptable (Q5 / Q4_K_M class) |
| C | 0.90–0.97 | the cliff (Q3 / Q2 class) |
| *rejected* | < 0.90 | below the floor — the model is no longer itself (Q1 / IQ1) |

Thresholds are provisional, calibrated by a ladder measurement on the box, and
revised only by measurement — never by fiat.

## The score — size

The house measures the model file's footprint in bytes (and bpw, normalized).
Smaller is better.

## The crown — dominance

A challenger takes a lane's crown **only if it is smaller than the incumbent
king AND still holds band A (≥0.99 top-1)**. A smaller-but-lossier recipe cannot
take the crown; it is listed on the board as a lossy alternative. Both sides
must share the same reference artifact and corpus version, else the crown holds
("reference mismatch").

## The seed — killing the free-ride

The house seeds each lane with the current best-known faithful quants (e.g.
Q8_0, Q6_K, Q5_K_M, Q4_K_M) as baseline receipts. Submitting those earns
nothing — the house already holds them. Miners are rewarded **only** for
producing something strictly smaller that still holds band A. The only path to
the crown is real compression innovation.

## Receipts

Every run produces one receipt: hash-bound (sha256 over all fields),
replayable, valid only against the current reference set. It records the
measured band, the house-measured size, the box fingerprint, and the corpus +
reference hashes. **Receipts are the source of truth for the board.** A receipt
that doesn't replay is not a receipt.

## Why miners can't farm it

- **Can't memorize** — the reference is immutable and not a task set.
- **Can't submit existing stuff** — the seed already owns it.
- **Can't game a judge** — there is no judge; the reference is a computation.
- **Can't adapt to game it** — a recipe is a *fixed file*, not an agent.

## Adding a model

Each lane is self-contained: its own reference, seed, receipts, and board,
sharing only this ruleset and the machinery under `keg/`. Copy the template
(`lanes/_TEMPLATE/`), generate the new model's reference from its BF16 weights,
seed it, and the lane is live. Cross-lane comparisons are meaningless (a Q6 of
one model vs a Q6 of another) and are never shown.
