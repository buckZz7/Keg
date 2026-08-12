# Lane: <model-family>

The compression race for **<Model Name>** (<one-line description>).

**The crown:** the smallest recipe that still holds the model's behavior — its
KL stays within the bound in every component vs the model's BF16 reference.
Only accepted recipes compete. See the [rules](../../RULES.md).

## The reference

The lane's reference is the model's own next-token behavior, generated once
from the BF16 weights over the **public corpus** — open and re-derivable by
anyone. See `reference/MANIFEST.md` and [corpus/](../../corpus/MANIFEST.md).

> Generate it: `tools/add_lane.sh <model-family>` scaffolds this lane. Then
> produce the reference on a box that can hold the BF16 weights (or a CPU pass)
> and fill `reference/MANIFEST.md`. The reference never changes while the lane
> is live.

## Board and rewards

See `board.md`. The crown is the only reward today; adding reward tiers is a
row in the Rewards table ("lanes within lanes").

## Submitting

Open a PR adding `lanes/<model-family>/recipes/<handle>.json` — one recipe,
exact model file by sha256. We measure it, post the receipt, and update the
board.
