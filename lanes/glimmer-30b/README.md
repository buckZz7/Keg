# Lane: glimmer-30b

The compression race for **Muse Glimmer 30B** (Meta's open-weight 30B dense
vision model, Apache 2.0). The anchor lane — where the mechanism is proven
before it spreads to more models.

**The crown:** the smallest recipe that still holds **≥0.99 top-1** fidelity
against the lane's BF16 reference over the public corpus. Only accepted
(≥0.99) recipes compete. See the [rules](../../RULES.md).

## The reference

The lane's reference is the model's own next-token behavior, generated once
from the BF16 weights over the **public corpus** — open and re-derivable by
anyone. See `reference/MANIFEST.md` and [corpus/](../../corpus/MANIFEST.md).
The 16-bit weights need ~64 GB and do not fit a single 5090, so the reference
is produced by a one-time CPU pass on a larger machine; it stays small because
it stores each sampled position's top-k log-probs.

## Board and rewards

See `board.md`. The crown is the only reward today; adding reward tiers is a
row in the Rewards table ("lanes within lanes").

## Submitting

Open a PR adding `lanes/glimmer-30b/recipes/<handle>.json` — one recipe, exact
model file by sha256. We measure it, post the receipt, and update the board.
