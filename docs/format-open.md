# Opening the competition to all model formats — roadmap

## The core decision

GGUF-only is correct for launch, but it caps the race at the GGUF frontier
(we proved: Q6_K is the smallest passing GGUF). To measure the *absolute*
frontier — and to make the benchmark relevant to the actual self-hosted market —
the competition must accept the formats people actually run.

**The efficient way is NOT "a loader per format."** It's a small set of
**runtime adapters**, each of which loads *many* formats through one path. Three
runtimes cover essentially the whole mainstream landscape:

| Runtime | Formats it loads | Status |
|---|---|---|
| **llama.cpp** | GGUF (incl. Blackwell FP4 types: Q4_0_4_4/4_8/8_8, MXFP4) | ✅ built |
| **vLLM** | GPTQ, AWQ, FP8 (W8A8), **NVFP4**, safetensors | ➕ to build |
| ExLlama | EXL2 | 🚧 only if demand |

vLLM alone covers the majority of real submissions (GPTQ/AWQ are the standard
GPU quantizations; safetensors is the universal container; NVFP4 rides inside it).

## The runtime-adapter contract

Every adapter implements ONE thing: **given a loaded model, produce the top-k
next-token logits at the reference's sampled positions** — exactly what the
single-pass extraction already does for llama.cpp. The gate, the reference, the
receipt, and the size metric are all format-agnostic already.

```
measure(model, positions, stream) -> topk_logits  # identical output shape every runtime
```

So a runtime adapter = (loader for that runtime) + (this extraction). The
heavy machinery (KL vs reference, per-component gate, receipt, board) is shared
and unchanged.

## The per-runtime self-check (non-negotiable before trust)

Each runtime must **reproduce the reference** before any submission is trusted on
it. The house runs the runtime on a known artifact (e.g. the BF16 itself) and
confirms KL ≈ 0.0 against the stored reference. Different runtimes have
different numerics, so this must be validated per runtime — not assumed.

## Reference policy (staged / empirical — not assumed)

The reference is the model's BF16 next-token behavior over the sampled positions.
Whether ONE reference can serve every runtime is an *empirical* question, not an
assumption — floating-point is non-associative, so different kernels produce
slightly different floats. **Research (arXiv 2506.09501) confirms this is the
crux of opening the competition:** BF16-compute has substantial variance across
configs; only FP32 accumulation with BF16 storage ("LayerCast") gives
near-perfect determinism (~2.2% sample divergence). Our current reference is
**llama.cpp BF16-compute** — a vLLM-measured BF16 model will NOT reproduce it
exactly, so without an upgrade, every runtime would need its own reference and
cross-format crown comparison would be impossible.

The policy is decided by measurement, never by fiat:

1. **Prerequisite: upgrade the reference to FP32 compute** (BF16 storage, FP32
   accumulation). This is the single highest-leverage move — it is what makes
   ONE reference reproducible across runtimes (llama.cpp and vLLM), which is
   what makes the crown cross-format. It must be done before any second runtime
   is trusted, and it must be re-validated to confirm it doesn't inflate the
   existing GGUF ladder (FP32 vs BF16 compute can shift numbers, forcing
   re-calibration).
2. **Default after upgrade: one FP32 reference** (the current corpus/positions,
   runtime-agnostic, re-derived deterministically from the corpus hash). The
   gate, receipt, and crown then score every format against the same truth.
3. **When a new runtime is added (e.g. vLLM), run the cross-reference
   self-check first:** generate that runtime's FP32 reference and compare to the
   stored one.
   - **Reproduces within tolerance** (far below the KL gate, 0.02) → **one
     reference serves all formats.** No proliferation; cross-format comparison
     stays clean.
   - **Does not reproduce** → that runtime gets its **own reference** (generated
     once, public, hash-bound, re-derivable). Only runtimes that genuinely need
     it get one.

**The per-runtime self-check is non-negotiable for every runtime regardless** —
the whole design rests on "this runtime genuinely reproduces the reference we're
scoring against." That reproduction tolerance (validated, far below the gate) is
the trust guarantee, and it's where the effort goes.

Net: **FP32-compute reference upgrade first, single reference by default,
per-runtime only when a self-check proves it needed.** The gate, receipt, and
crown are unaffected — they all just score against whatever reference the
runtime reproduces.

## Anti-memorization gate per format

The gate ("must be a real, loadable model of the lane's architecture, not a
lookup table") becomes format-specific only in *inspection*:
- **GGUF**: parse the binary header → tensor names/types (done, `gguf_tensors.py`).
- **safetensors**: read the **header only** (tensor names/shapes/dtypes are stored
  separately from the bytes) → no need to load the full weights. Cheap.
- **EXL2**: safetensors container → same header inspection.

Then the house loads it in the adapter and measures — if it isn't a real model,
the loader fails or the fidelity is obviously wrong.

## NVFP4 note

NVFP4 is a *precision scheme*, not a file format. It ships inside safetensors
(vLLM / TensorRT-LLM) or as llama.cpp Blackwell FP4 GGUF types. It needs a runtime
with native FP4 kernels (Blackwell / 5090). Its fidelity is genuinely debated, so
the self-check + the KL gate are exactly what will decide whether it's "still the
model" or a lossy loss. No special-casing required.

## Incremental rollout (each gated on self-check passing)

1. Keep llama.cpp → GGUF (current), measured against the current BF16 reference.
2. **Regenerate the reference with FP32 compute** (BF16 storage, FP32
   accumulation). Re-validate the GGUF ladder against it — confirm it doesn't
   shift the existing Q6_K crown / acceptance boundary. This is the
   prerequisite for any cross-runtime work; without it, no second runtime can
   share a reference.
3. **Build the vLLM adapter** → validates GPTQ/AWQ/FP8/NVFP4/safetensors. Note
   this is heavier than a loader: our gate needs **deep top-k (1024) logprobs at
   arbitrary stream positions**, which vLLM only exposes at full-vocab cost in
   library mode (`logprobs=-1`; the OpenAI-compat `max_logprobs` defaults to 20
   and is capped). Self-check the adapter against the FP32 reference, then open
   those formats.
4. Add **ExLlama → EXL2** later if there's demand.

## What does NOT change

- The gate (worst-component KL ≤ 0.02) applies to every format.
- The crown is the smallest accepted recipe, regardless of format.
- Receipts bind sha256 + house-measured size + house-inspected tensor structure
  — all format-agnostic, all unfakeable.
- No LLM judge; trustlessness from the public, re-derivable reference.

## Cost of opening

The real cost is the verification surface (more runtimes, more self-checks, more
"is this even the model" surface), not the format count. Keeping the per-runtime
adapter model bounds that cost to ~2-3 adapters for the whole market.
