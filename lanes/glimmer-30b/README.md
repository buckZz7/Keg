# Lane: glimmer-30b

The compression race for **Muse Glimmer 30B** (Meta's open-weight 30B dense
vision model, Apache 2.0). The anchor lane — this is where the mechanism is
proven before it spreads to more models.

**The crown:** the smallest recipe that still holds **band A (≥0.99 top-1)**
fidelity against the lane's BF16 reference. See the [rules](../../RULES.md).

## The reference

The lane's reference is the model's own next-token distributions, generated
once from the BF16 weights over a fixed held-out corpus, hash-bound, and
house-held. See `reference/MANIFEST.md`. The 16-bit weights need ~64 GB and do
not fit a single 5090, so the reference is produced by a one-time CPU pass on
a larger machine — it is ~40 KB and replayable forever.

## Seeded baseline (the free-ride killer)

The house pre-claims the current best-known faithful quants for this model.
Submitting these earns nothing; only a strictly-smaller recipe that still holds
band A can take the crown. See `seed/MANIFEST.md`.

## Board

See `board.md` — receipts are the source of truth.

## Model context (cited, not run by us)

Model-level benchmark numbers are from the publisher; they describe the model,
not any serving configuration. Source: [Unsloth, Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer) (2026-08-10).

| Benchmark | Muse Glimmer-30B |
|---|---|
| MCP Atlas (public) | 75.5 |
| SWE-Bench Verified | 76.0 |
| SWE-Bench Pro | 51.2 |
| AIME 2026 | 94.7 |
| GPQA Diamond (AA) | 83.5 |
| OSWorld-Verified | 65.9 |
| MMMU Pro | 74 |

## Submitting

Open a PR adding `lanes/glimmer-30b/recipes/<handle>.json` — one recipe, exact
model file by sha256. We measure it, post the receipt, and update the board.
