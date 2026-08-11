# Keg — Rules

The full mechanism. Short version: **the smallest recipe that still *is* the
model wins. A recipe is accepted — and competes — only if it holds ≥99% of the
model's true behavior.** Everything is measured, nothing is claimed. There are
no fidelity tiers: you are either still the model, or you aren't.

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

## The gate — acceptance

Fidelity = top-1 token match (+ KL) of the recipe's output distribution vs the
reference, over the held-out corpus. Deterministic. **A recipe is accepted only
if it holds ≥0.99 top-1.** Below that it is rejected — the model is no longer
recognizably itself.

This bar is the anti-free-ride. Holding ≥99% is not free: most off-the-shelf
low quants (Q4_K_M class) land at 97–98% and get rejected. You cannot submit
something found lying around; you must produce something that is genuinely
still the model.

## The score — size

The house measures the model file's footprint in bytes (and bpw, normalized).
Smaller is better.

## The crown — dominance

A challenger takes a lane's crown **only if it is smaller than the incumbent
king AND accepted (≥0.99 top-1)**. A smaller-but-lossier recipe is rejected,
not rewarded. Both sides must share the same reference artifact and corpus
version, else the crown holds ("reference mismatch"). There is no seed: the
first accepted recipe establishes the crown, and anyone who beats it is
rewarded for genuinely better compression.

## Rewards

Rewards are listed per lane in `board.md` under **Rewards**. Today there is one:
the **crown** (smallest accepted recipe). If more reward tiers are ever wanted,
they are added as rows in the lane's Rewards table — no new machinery. Nothing
in the reference or the gate changes when a tier is added.

## Receipts

Every run produces one receipt: hash-bound (sha256 over all fields),
replayable, valid only against the current reference set. It records the
measured fidelity, the house-measured size, the box fingerprint, and the
corpus + reference hashes. **Receipts are the source of truth for the board.** A
receipt that doesn't replay is not a receipt. Rejected attempts get a receipt
too, so every run is on record.

## Why miners can't farm it

- **Can't memorize** — the reference is immutable and not a task set.
- **Can't submit junk** — below ≥0.99 it's rejected outright; holding 99% is not free.
- **Can't game a judge** — there is no judge; the reference is a computation.
- **Can't adapt to game it** — a recipe is a *fixed file*, not an agent.

## Adding a model

Each lane is self-contained: its own reference, receipts, and board, sharing
only this ruleset and the machinery under `keg/`. Copy the template
(`lanes/_TEMPLATE/`), generate the new model's reference from its BF16 weights,
and the lane is live. Cross-lane comparisons are meaningless (a Q6 of one model
vs a Q6 of another) and are never shown.
