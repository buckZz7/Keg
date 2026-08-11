# Keg

Keg is a compression race: **how small can you make a model while it still *is* the model?**

It matters because every edge deployer hits the same wall — RTX Spark, a 5090, a phone: memory is fixed, and a smaller faithful model is cheaper to host and fits hardware the full model can't. Keg is where that question gets answered, honestly.

One repo, one lane per model. Miners submit quantized recipes; the **smallest recipe that still holds ≥99% of the model's true behavior wins** the lane's crown.

## How fidelity is measured

Every recipe is compared against the **model's own BF16 next-token distributions** — generated once, hash-bound, deterministic. Fidelity is a measurement, not a judgment: a recipe either holds ≥99% top-1 against the model's true behavior or it doesn't. Below that, it's rejected. No judge model, no test set to memorize.

## Lanes

| Lane | Model | Crown (smallest accepted recipe) |
|---|---|---|
| [glimmer-30b](lanes/glimmer-30b/) | Muse Glimmer 30B | — |

## How a lane works

1. **Submit a recipe** — the exact model file (by sha256), format/quant, and runtime.
2. **We measure it** — fidelity vs the lane's BF16 reference (accepted only at ≥99%), plus the true file size.
3. **The crown** — the smallest accepted recipe wins. Dethrone only if *smaller AND accepted*; a smaller-but-lossier recipe is rejected.

## Compete

Open a pull request adding `lanes/<model>/recipes/<handle>.json` — one recipe, exact model file by sha256. We measure it, post the receipt, and update the board. Receipts are the source of truth.

Full mechanism and rules: [RULES.md](RULES.md)
