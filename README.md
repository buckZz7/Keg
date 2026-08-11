# Keg

A compression race: **how small can you make a model while it still *is* the
model?**

One repo, one lane per model. Miners submit quantized recipes; the **smallest
recipe that still holds ≥99% of the model's true behavior wins** the lane's
crown.

## Lanes

| Lane | Model | Crown (smallest accepted recipe) |
|---|---|---|
| [glimmer-30b](lanes/glimmer-30b/) | Muse Glimmer 30B | — |

## How a lane works

1. **Submit a recipe** — the exact model file (by sha256), format/quant, runtime.
2. **We measure it** — fidelity vs the lane's BF16 reference (accepted only at ≥99%), plus the true file size.
3. **The crown** — the smallest accepted recipe wins. Dethrone only if *smaller AND accepted*.

Full mechanism: [RULES.md](RULES.md)
