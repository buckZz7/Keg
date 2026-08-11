# Lane: <model-family>

The compression race for **<Model Name>** (<one-line description>).

**The crown:** the smallest recipe that still holds **band A (≥0.99 top-1)**
fidelity against the lane's BF16 reference. See the [rules](../../RULES.md).

## The reference

The lane's reference is the model's own next-token distributions, generated
once from the BF16 weights over a fixed held-out corpus, hash-bound, and
house-held. See `reference/MANIFEST.md`.

> Generate it: `tools/add_lane.sh <model-family>` scaffolds this lane. Then
> produce the reference on a box that can hold the BF16 weights (or a CPU pass)
> and fill `reference/MANIFEST.md`. The reference never changes while the lane
> is live.

## Seeded baseline

The house pre-claims the current best-known faithful quants for this model.
Submitting these earns nothing; only a strictly-smaller recipe that still holds
band A can take the crown. See `seed/MANIFEST.md`.

## Board

See `board.md` — receipts are the source of truth.

## Submitting

Open a PR adding `lanes/<model-family>/recipes/<handle>.json` — one recipe,
exact model file by sha256. We measure it, post the receipt, and update the
board.
