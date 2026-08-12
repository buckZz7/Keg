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

1. Keep llama.cpp → GGUF (current).
2. **Build the vLLM adapter** → validates GPTQ/AWQ/FP8/NVFP4/safetensors on the
   5090 eval box. Self-check, then open those formats.
3. Add **ExLlama → EXL2** later if there's demand.

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
