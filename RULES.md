# Keg — Rules

The smallest recipe that still *is* the model leads the board. A recipe is
accepted — and competes — only if its token distribution stays within a KL
bound of the model's (near-lossless). Everything is measured, nothing is
claimed.

## The reference (the truth, public and re-derivable)

Every recipe is measured against **the model's own next-token behavior** —
generated once from the 16-bit (BF16) weights, over a fixed set of positions
sampled from a **public, diverse, versioned corpus**. Anyone can re-derive the
reference by running the model over the same corpus.

- **Public by design.** The corpus is open and hash-bound; the reference
  artifact stores each sampled position's top-k log-probs (enough to replay
  both metrics below). No sealed corpus, no house authority — trustlessness
  comes from openness, not secrecy.
- **Deterministic.** The sampled positions are a pure function of the corpus
  (seeded by its hash), so the reference and every submission measure the same
  points. Repro­ducible by anyone.
- **No LLM in the loop.** Fidelity is computed from token predictions, not
  judged by a model.

## Eval hardware — what we're after

**Keg is for the models people actually run, on hardware they actually own.**
The house reference is generated once on a big box that can hold the lane's
BF16 weights (e.g. an A100) — that's just the calibration artifact, not the
target. **Submissions are evaluated on a consumer card (e.g. a 5090)**, because
the goal is compression that holds the model on the edge / self-hosted /
consumer hardware the benchmark is for.

- **The gate is hardware-independent.** Size is bytes; KL is the model's
  next-token distribution, identical whichever GPU loads it (provided the box
  reproduces the reference). The race's validity does not depend on the eval GPU.
- **Reproduction is checked, not assumed.** Before an eval box is trusted, the
  house re-runs the **self-check** (the reference vs itself → KL 0.0) on that
  box. If it reproduces the stored reference, eval there is sound.
- **tps is measured on the eval box** and labeled with it, so the reported
  speed is relevant to the hardware the market runs — not a datacenter card.

**Why a consumer card (e.g. the RTX 5090) is the eval box.** The 5090 — 32 GB
GDDR7, ~1.8 TB/s memory bandwidth, ~$2k — is the practical ceiling of
*single-card hardware people actually own*; beyond it you are in datacenter /
workstation rental territory (A100/H100). A 32 GB card fits the entire
quantization ladder of a 30B model (Q4 ≈16 GB → Q8 ≈30 GB), so the whole race of
a lane is playable on the target hardware. And because decode is memory-
bandwidth-bound, a 5090 tps is what a self-hoster genuinely gets — an A100 tps
would overstate the market's real speed. The frontier this race rewards —
"how small can I go and still hold the model" — is precisely what decides which
VRAM tier a model fits, i.e. which card you need to buy.

So: **reference on a big box (once), eval on a consumer card (always).** We are
after the smallest recipe that is still the model *on hardware you can buy*.

## The submission (a recipe)

A recipe is the exact model file (by sha256), its format/quant, and the runtime
that produced it. The metric is **size** — the model's footprint —
measured by the house from the real file, never from the miner's word.

**A recipe must be a real, loadable model file of the lane's architecture.**
The house rejects anything it cannot serve as the model (currently **GGUF**,
loaded with llama.cpp). This is the anti-memorization gate: an answer store /
lookup table that merely replays the corpus is **not a model**, so it is
rejected before any fidelity is measured. A recipe the house cannot reproduce
is not a submission.

## The gate — acceptance

Fidelity is measured vs the model's own reference, over the sampled positions,
using the field's **single-pass long-mode** measurement (the llama-perplexity /
mlx-kld method: the corpus stream is processed once in context windows with KV
reuse, and the top-k log-probs at each sampled position are recorded):

- **KL divergence** (primary) — how much the recipe's next-token distribution
  drifts from the model's. KL is the field's fidelity metric of record
  (*"Accuracy is Not All You Need,"* arXiv:2407.09141; Fireworks; llama-
  perplexity): it is highly correlated with answer flips, and it is the metric
  that separates near-lossless quants from lossy ones on a hard, diverse corpus.
- **top-1 agreement** (reported, not gated) — the recipe's most-likely token vs
  the model's, for human readability.

**Positions are sampled per component** (prose, code, multilingual, technical)
with a floor, so no component is a thin weak spot a miner could overfit or
ignore — and **a recipe is accepted only if EVERY component's mean KL passes
the bound**, not just the overall mean. Gating on the worst component closes the
"excel on the dominant component, let the rest fail" attack (smcleod: "a quant
that holds up on prose but falls apart on code would look fine here" — unless
you gate per component). top-1 is reported but is not a pass/fail gate.

The threshold is **calibrated by a ladder** — on a new model the house measures
where the known quants (Q8 → Q4) land against the reference, then sets the bar
from data, anchored to the field's near-lossless KLD range (≈0.05–0.1 nats;
Fireworks production-quality deployments < 7e-3). Not by fiat.

Keg gates on KL (still the model, or not) and then ranks by size — the right
discriminator in the near-baseline cluster where fidelity metrics lose the
power to rank. (*"Displacement Is Not Direction,"* arXiv:2606.19558.)

## The score — size

The house measures the model file's footprint in bytes (and bpw, normalized).
Smaller is better.

## The crown — current best

A recipe becomes the current best **only if it is smaller than the incumbent
AND accepted (within the KL bound)**. A smaller-but-lossier
recipe is rejected, not rewarded. Both sides must share the same corpus and
reference, else the crown holds. There is no seed: the first accepted recipe
establishes the crown; anyone smaller takes it.

## Rewards

Rewards are listed per lane in `board.md` under **Rewards**. Today there is one:
the **crown** (smallest accepted recipe). More tiers can be added as rows in the
lane's Rewards table — no new machinery.

## Receipts

Every run produces one receipt: hash-bound (sha256 over all fields), replayable,
valid only against the current reference. It records the measured fidelity, the
house-measured size, the corpus + reference hashes, and the box fingerprint.
**Receipts are the source of truth for the board.** A receipt that doesn't
replay is not a receipt. Rejected attempts get a receipt too.

## Why it's trustworthy

- **Breadth kills calibration.** The corpus is broad and diverse; calibrating a
  quant to a broad corpus just makes a generally-better quant — the honest
  behavior. Narrow-corpus overfitting is bounded.
- **A recipe must be a real model, not an answer store.** The anti-memorization
  gate rejects any submission that isn't a loadable model file of the lane's
  architecture — a lookup table replaying the corpus is not a model and can't
  rank by size. Memorizing a broad corpus is impossible anyway: encoding
  thousands of diverse positions costs size, which loses out.
- **Public = verifiable.** Anyone re-derives the reference and checks a receipt.
