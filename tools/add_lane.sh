#!/usr/bin/env bash
# Scaffold a new Keg lane (a per-model compression race).
#
#   tools/add_lane.sh <model-family>
#
# Creates lanes/<model-family>/ (README, reference/, seed/, receipts/,
# board.md) from the template. It does NOT generate the reference — that is a
# house step on a box that can hold the model's BF16 weights.
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

mkdir -p "$LANE/reference" "$LANE/seed" "$LANE/receipts"
cp "$ROOT/lanes/_TEMPLATE/README.md" "$LANE/README.md"

# Fill the placeholders.
sed -i "s/<model-family>/$L/g" "$LANE/README.md"

cat > "$LANE/reference/MANIFEST.md" <<'EOF'
# Lane reference — <model-family>

The lane's yardstick: the model's own next-token distributions, generated once
from the **BF16 weights** over a fixed held-out corpus, hash-bound, and
house-held.

- `reference.json` — the artifact (prompts keyed by hash; text never in repo).
- `reference.sha256` — the artifact's hash, bound into every receipt.

**Immutability:** the reference never changes while the lane is live. Any
change invalidates every prior receipt.

> **Status: pending house generation.** Produce the artifact from the real
> BF16 weights on the box; the scaffold does not fabricate it.
EOF

cat > "$LANE/seed/MANIFEST.md" <<'EOF'
# Lane seed — <model-family>

The house pre-claims the current best-known faithful quants for this model so
submitting them earns nothing. Only a strictly-smaller recipe that still holds
band A (≥0.99 top-1) can take the crown.

| Quant | Approx size | Why it's in the seed |
|---|---|---|

> **Status: seed hashes pending.** Baseline quant files are house-generated and
> hashed into receipts on the box.
EOF

cat > "$LANE/board.md" <<'EOF'
# Board — <model-family>

The crown: the **smallest recipe that holds band A (≥0.99 top-1)** vs the
lane's BF16 reference. Receipts are the source of truth.

| Crown | Quant / format | Size (GB) | bpw | Fidelity (top-1) | Receipt |
|---|---|---|---|---|---|

Lossy alternatives (band B/C) that are smaller but can't take the crown:

| Quant / format | Size (GB) | Fidelity (top-1) | Band | Receipt |
|---|---|---|---|---|
EOF

sed -i "s/<model-family>/$L/g" "$LANE/reference/MANIFEST.md" \
  "$LANE/seed/MANIFEST.md" "$LANE/board.md"

echo "lane scaffolded: $LANE"
echo "next: generate the reference from the model's BF16 weights, then seed it."
