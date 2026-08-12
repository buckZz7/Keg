#!/usr/bin/env bash
# Scaffold a new Keg lane (a per-model compression race).
#
#   tools/add_lane.sh <model-family>
#
# Creates lanes/<model-family>/ (README, reference/, receipts/, board.md)
# from the template. It does NOT generate the reference — that is a house
# step on a box that can hold the model's BF16 weights.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <model-family>" >&2
  exit 1
fi

L="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANE="$ROOT/lanes/$L"

if [ -e "$LANE" ]; then
  echo "lane already exists: $LANE" >&2
  exit 1
fi

mkdir -p "$LANE/reference" "$LANE/receipts"
cp "$ROOT/lanes/_TEMPLATE/README.md" "$LANE/README.md"
sed -i "s/<model-family>/$L/g" "$LANE/README.md"

cat > "$LANE/reference/MANIFEST.md" <<'EOF'
# Lane reference — <model-family>

The lane's yardstick: the model's own next-token behavior, generated once
from the **BF16 weights** over a fixed set of positions sampled from the
**public corpus**. Anyone can re-derive it by running the model over the same
corpus.

- `reference.json` — the artifact, storing only each sampled position's top-1
  token. The corpus is public and pinned by its sha256 (see corpus/MANIFEST.md).
- `reference.sha256` — the artifact's hash, bound into every receipt.

**Immutability:** the reference never changes while the lane's submissions are
live. Any change invalidates every prior receipt. The KL bound is set by a
calibration ladder on the box, anchored to the field's near-lossless range.

> **Status: pending house generation.** Produce the artifact from the real
> BF16 weights over the production corpus on the box; the scaffold does not
> fabricate it.
EOF

cat > "$LANE/board.md" <<'EOF'
# Board — <model-family>

Receipts are the source of truth. Only accepted recipes (KL within the bound in
every component vs the lane's BF16 reference) appear here; rejected attempts are
recorded under
`receipts/` but hold nothing.

## Rewards

Adding a reward tier is a new row here — "lanes within lanes." No new machinery.

| Reward | Current holder | Quant / format | Size (GB) | bpw | Fidelity (top-1) | Receipt |
|---|---|---|---|---|---|---|
| **Crown** (smallest accepted) | — | — | — | — | — | — |

## Recent receipts

| Quant / format | Size (GB) | Fidelity (top-1) | Verdict | Receipt |
|---|---|---|---|---|
EOF

sed -i "s/<model-family>/$L/g" "$LANE/reference/MANIFEST.md" "$LANE/board.md"

echo "lane scaffolded: $LANE"
echo "next: generate the reference from the model's BF16 weights, then it's live."
