# Lane reference — glimmer-30b

The lane's yardstick: the model's own next-token behavior, generated once from
the **BF16 weights** over a fixed set of positions sampled from the **public
corpus**. Anyone can re-derive it by running the model over the same corpus.

**What is stored here:**
- `reference.json` — the artifact: `{schema, corpus_version, corpus_sha256,
  top_k, n, positions: {<doc:offset>: {token: logprob}}}`. It stores each
  sampled position's top-k log-probs (enough to replay top-1 and KL), so it
  stays small even for a large corpus.
- The **corpus is public** (see [corpus/](../../corpus/MANIFEST.md)) and pinned
  by `corpus_sha256`, recorded in every receipt.

**Production** (house, on a box that can hold the BF16 weights, or a CPU pass):

```
python -c "from keg.fidelity import load_corpus, save_reference; \
  print(save_reference('http://<bf16-server>/v1', \
  'lanes/glimmer-30b/reference/reference.json', load_corpus()))"
```

**Immutability / calibration:** the reference never changes while a lane's
submissions are live; any change invalidates prior receipts. The ≥99% acceptance
bar is set by a **calibration ladder** (measure Q8→Q2 against this reference on
the box first), not assumed.

> **Status: pending house generation.** The artifact is produced from the real
> BF16 weights over the production corpus on the box; the scaffold does not
> fabricate it.
