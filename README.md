# Keg

A compression benchmark: **how small can you make a model while it still *is*
the model?**

One repo, one lane per model. Submit a quantized recipe; the **smallest recipe
that still holds ≥99% of the model's true behavior** leads the lane's board.

## How fidelity is measured

Every recipe is measured against the **model's own next-token behavior** — the
model's BF16 reference over a **public, diverse corpus**. The corpus is open
and hash-bound, so anyone can re-derive the reference by running the model
over it. Fidelity is a measurement, not a judgment: a recipe either holds
**≥99% top-1 agreement and stays within the KL bound** against the model's true
behavior, or it doesn't. Below that it's rejected. No judge model, no sealed
corpus, no house authority.

## Lanes

| Lane | Model | Current best (smallest accepted) |
|---|---|---|
| [glimmer-30b](lanes/glimmer-30b/) | Muse Glimmer 30B | — |

## How a lane works

1. **Submit a recipe** — the exact model file (by sha256), format/quant, runtime.
2. **We measure it** — top-1 next-token agreement + KL divergence vs the
   model's BF16 reference over the public corpus (accepted only at ≥99% top-1
   within the KL bound), plus the true file size.
3. **The board** — accepted recipes rank by size; the smallest holds the top spot.

Full mechanism: [RULES.md](RULES.md)

## Why this isn't a speed benchmark

Raw speed is owned by the runtime incumbents (vLLM, sparkinfer). This benchmark
is about *compression science* — quantization, calibration — which they don't
optimize. It's the question every edge deployer asks: how small can I go and
keep the model real?
