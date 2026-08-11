# Lane seed — glimmer-30b

The house pre-claims the current best-known faithful quants for this model so
that submitting them earns nothing. **The only path to the crown is producing
something strictly smaller that still holds band A (≥0.99 top-1).**

Seeded baseline (house-held; file hashes recorded once generated):

| Quant | Approx size (30B) | Why it's in the seed |
|---|---|---|
| Q8_0 | ~32 GB | near-lossless reference-class; already the practical ceiling on a 5090 |
| Q6_K | ~23 GB | ~99%+ top-1; the serving-grade floor |
| Q5_K_M | ~20 GB | ~98-99%; strong quality-per-bit |
| Q4_K_M | ~17 GB | the ecosystem "acceptable" sweet spot |

These are the incumbent recipes. To take the crown a submission must be
**smaller than the smallest seeded band-A recipe and still ≥0.99 top-1** — i.e.
real compression, not a re-submission of what the house already holds.

> **Status: seed hashes pending.** Baseline quant files are house-generated and
> hashed into receipts on the box; the scaffold does not fabricate them.
