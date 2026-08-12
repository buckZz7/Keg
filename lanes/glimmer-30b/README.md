# Lane: glimmer-30b

The compression benchmark for **Muse Glimmer 30B** (Meta's open-weight 30B dense
vision model, Apache 2.0). The anchor lane — where the mechanism is proven
before it spreads to more models.

**The crown:** the smallest recipe that stays within the **KL bound** of the
lane's BF16 reference — measured under the field's long-mode KLD (deep top-k
over long-context positions). Only accepted (within the KL bound) recipes
compete. See the [rules](../../RULES.md).

## The reference

The lane's reference is the model's own next-token behavior, generated once
from the BF16 weights over the **public corpus** — open and re-derivable by
anyone. See `reference/MANIFEST.md` and [corpus/](../../corpus/MANIFEST.md).
The 16-bit weights need ~64 GB and do not fit a single 5090, so the reference
is produced by a one-time pass on a box that can hold them (e.g. an A100-80GB);
it stays small because it stores each sampled position's deep top-k log-probs
(top-1024), enough to replay the gate's KLD to within ~2% of full-vocabulary.

## Board and rewards

See `board.md`. The crown is the only reward today; adding reward tiers is a
row in the Rewards table ("lanes within lanes").

## Submitting

Open a PR adding `lanes/glimmer-30b/recipes/<handle>.json` — one recipe, exact
model file by sha256. We measure it, post the receipt, and update the board.
