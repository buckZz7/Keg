# Keg — Rules

The smallest recipe that still holds the model's behavior wins. A recipe is
accepted — and competes — only if it holds **≥99%** of the model's true
behavior. Everything is measured, nothing is claimed.

## The reference

Every recipe is measured against the model's own next-token distributions,
generated once from the 16-bit (BF16) weights over a fixed held-out corpus,
stored as a hash-bound artifact.

- **Immutable.** The reference never changes while a lane is live; a change
  invalidates every prior receipt.
- **Non-trainable.** It is a property of the weights, not a task set.
- **Self-authenticating.** Corpus and reference are hash-bound; the corpus text
  is house-held and never appears in the repo. Receipts bind to both hashes.
- **No LLM in the loop.** Fidelity is computed from token distributions, not
  judged by a model.

## The submission (a recipe)

A recipe is the exact model file (by sha256), its format/quant, and the runtime
that produced it. A recipe the house cannot reproduce is not a submission. The
race metric is **size** — the model's footprint — measured by the house from
the real file, never from the miner's word.

## The gate — acceptance

Fidelity = top-1 token match (+ KL) of the recipe's output distribution vs the
reference, over the held-out corpus. Deterministic. **A recipe is accepted only
if it holds ≥0.99 top-1**; below that it is rejected.

## The score — size

The house measures the model file's footprint in bytes (and bpw, normalized).
Smaller is better.

## The crown — dominance

A challenger takes a lane's crown **only if it is smaller than the incumbent
king AND accepted (≥0.99 top-1)**. A smaller-but-lossier recipe is rejected, not
rewarded. Both sides must share the same reference artifact and corpus version,
else the crown holds. There is no seed: the first accepted recipe establishes
the crown; anyone smaller takes it.

## Rewards

Rewards are listed per lane in `board.md` under **Rewards**. Today there is one:
the **crown** (smallest accepted recipe). More tiers can be added as rows in the
lane's Rewards table — no new machinery.

## Receipts

Every run produces one receipt: hash-bound (sha256 over all fields), replayable,
valid only against the current reference set. It records the measured fidelity,
the house-measured size, the box fingerprint, and the corpus + reference hashes.
**Receipts are the source of truth for the board.** A receipt that doesn't
replay is not a receipt. Rejected attempts get a receipt too, so every run is on
record.
