# Lane reference — glimmer-30b

The lane's yardstick: the model's own next-token behavior, generated once from
the **BF16 weights** over a fixed set of positions sampled from the **public
corpus**. Anyone can re-derive it by running the model over the same corpus.

**What is stored here:**
- `reference.json` — the artifact: `{positions: {<token_index>: {token:
  logprob}}, n}`. It stores each sampled position's **top-k** log-probs
  (enough to replay top-1 and KL), so it stays small even for a large corpus.
- `components.json` — per-position component label
  (`prose` / `code` / `multilingual` / `technical`) used for the per-component
  gate (a recipe must pass the KL bound in *every* component, not the average).
- The **corpus is public** (see [corpus/](../../corpus/MANIFEST.md)) and pinned
  by `corpus_sha256`, recorded in every receipt.

**Production** (house, single-pass harness on a box that can hold the model):

```
# 1. tokenize the stream + sample stratified positions (vocab-only, ~seconds)
keg_sample --model <bf16-or-any-gguf> --corpus corpus/production.txt \
  --components corpus/production.components.txt --outdir <dir> --n 5000
# 2. extract top-k at those positions in one streaming pass (~minutes)
keg_extract --model <bf16.gguf> --tokens <dir>/tokens.txt \
  --positions <dir>/positions.txt --out reference.json --n-ctx 8192 --top-k 1024
```

The harness (`keg_sample.cpp`, `keg_extract.cpp`) is compiled against the
model's own llama.cpp build so the tokenizer matches exactly. Both the BF16
reference and every submission use the **same** token positions and the same
extractor, so their KL is directly comparable.

**Immutability / calibration:** the reference never changes while a lane's
submissions are live; any change invalidates prior receipts. The KL acceptance
bar is set by a **calibration ladder** (measure Q8→Q4 against this reference on
the box first), anchored to the field's near-lossless KLD range — not assumed.

## Generation record (real)

- **Date:** 2026-08-12
- **Method:** single-pass `keg_extract` (C++ harness against the custom
  llama.cpp with muse-glimmer support) on the BF16 glimmer-30b, A100-80GB,
  8192-token windows, KV reuse, top-k 1024.
- **Artifact:** `reference.json` — sha256 `e3f47fb78db36d8b` (file, 101 MB).
- **corpus_version:** 3 · **corpus_sha256:** `757502cd9a63dddc` ·
  **components_sha256:** `3f525c7d40986e7c` · **top_k:** 1024 · **n:** 4,999
- **Positions:** token indices (not char offsets), ≥2048 tokens of in-window
  context, sampled per component (all four represented).
- **Verified:** self-check — BF16 measured against its own reference gives
  `top1 = 1.0`, `KL = 0.0` across all four components (n=4999).
- **Re-derivable:** run `keg_sample` + `keg_extract` (above) with the same
  pinned corpus, components, n, n_ctx and top_k to reproduce this artifact.
