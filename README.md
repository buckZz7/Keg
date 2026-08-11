# Keg

A compression race: **how small can you make a model while it still *is* the
model?**

One repo, many **lanes** — one lane per model. Each lane is a self-contained
race: miners submit quantized recipes; the smallest recipe that still holds
the model's true behavior wins the lane's crown. Adding a model is just adding
a lane (copy the template, generate its reference, seed it).

The gate is ungameable by construction: every recipe is measured against the
**model's own BF16 next-token distributions** — computed once, hash-bound, and
deterministic. No LLM judge, no task set to memorize, no reward farming. A
recipe that washes, overfits, or cheats fails automatically.

## Lanes

| Lane | Model | Crown (smallest faithful recipe) | Reference |
|---|---|---|---|
| [glimmer-30b](lanes/glimmer-30b/) | Muse Glimmer 30B | — | house-held, BF16 |

## The idea in one line

Race to make the smallest copy of a model that still acts like the real model.
The real model's true behavior is locked in the reference (a computation anyone
can re-derive). Your copy must match it ≥99% of the time and be smaller than
everyone else's — including the copies the house already seeded. Smallest
faithful wins. You can't cheat: the reference can't be faked, and your recipe is
a fixed file, not an agent.

## How a lane works

1. **Submit a recipe** — the exact model file (by sha256), format/quant, and
   runtime. A recipe the house can't reproduce isn't a submission.
2. **We measure it** — fidelity vs the lane's stored BF16 reference (top-1
   match + KL), deterministic, no LLM. And the house measures the *true* file
   size.
3. **The crown** — the smallest recipe holding **band A (≥0.99 top-1)**.
   Dethrone only if *smaller AND as faithful*. A smaller-but-lossier recipe
   can't take the crown.
4. **Receipt** — every run produces a hash-bound, replayable receipt. Receipts
   are the source of truth for the board.

See [RULES.md](RULES.md) for the full mechanism.

## Adding a model

Adding a model is meant to be trivial — copy the template lane and fill it in:

```bash
tools/add_lane.sh <model-family>          # scaffolds lanes/<model-family>/
```

Then: generate the lane's reference from the model's BF16 weights on a box
large enough to hold them, seed it with the current best-known faithful quants,
and it's live. The machinery under `keg/` is shared; only the lane's reference
and seed differ per model.

## Why not a speed race

Raw speed is owned by the runtime incumbents (vLLM, sparkinfer). This race is
about *compression science* — quantization, calibration — which they don't
optimize. It's the question every edge deployer actually asks: how small can I
go and keep the model real?

## Layout

```
keg/            shared machinery (fidelity gate, receipts, runner)
lanes/<model>/  one self-contained race per model (reference, seed, receipts, board)
tools/          add_lane.sh and helpers
```
