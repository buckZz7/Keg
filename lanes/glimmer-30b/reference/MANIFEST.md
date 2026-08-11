# Lane reference — glimmer-30b

The lane's yardstick: the model's own next-token distributions, generated once
from the **BF16 weights** over a fixed held-out corpus.

**What is stored here** (house-side, hash-bound):

- `reference.json` — the artifact: `{schema: keg/reference-v1, corpus_version,
  corpus_sha256, prompts: {<prompt_hash>: [{token, prob}, ...]}}`. Prompts are
  keyed by hash — the corpus **text never appears** in the repo.
- `reference.sha256` — the artifact's hash, bound into every receipt.

**Production** (house, on a box that can hold the BF16 weights, or a CPU pass):

```
python -c "from keg.fidelity import save_reference; \
  print(save_reference('http://<bf16-server>/v1', 'lanes/glimmer-30b/reference/reference.json'))"
```

The corpus (`KEG_CORPUS_FILE`) is house-held and published only when
superseded. A public corpus could be overfitted by calibration.

**Immutability:** the reference never changes while the lane is live. Any
change invalidates every prior receipt. Receipts bind to `reference_sha256`
and `corpus_sha256`.

> **Status: pending house generation.** The artifact is produced from the real
> BF16 weights on the box; the scaffold does not fabricate it.
