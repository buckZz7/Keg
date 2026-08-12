# Submissions — glimmer-30b

Each submission is one **recipe**: a JSON file naming an exact, real model file
of the lane's architecture, byte-addressed by `model_sha256`, plus the `source`
where the house fetches it. Add it as `recipes/<handle>.json` in a PR.
See [`EXAMPLE.json`](EXAMPLE.json).

## Format

| Field | Meaning |
|---|---|
| `model` | Model family, e.g. `muse-glimmer-30b` |
| `model_file` | The model filename |
| `model_sha256` | SHA-256 of the **exact file** — verified byte-for-byte by the house |
| `quant` | Format / quant level, e.g. `Q6_K`, `Q8_0` |
| `format` | Container: `gguf` at launch (other formats arrive via `docs/format-open.md`) |
| `runtime` | Runtime that serves it, e.g. `llama.cpp` |
| `runtime_version` | Exact runtime commit, for reproducibility |
| `source` | URL or repo path the house fetches the file from — **verified** (fetched + hashed, must match `model_sha256`) |
| `note` | Free-form (no weight in scoring) |

## What the house does (verify, don't trust)

1. **Fetches** the file from `source`, **hashes** it — must match `model_sha256`
   byte-for-byte, else rejected.
2. **Validates the file** — must be a real, loadable model of this architecture;
   a lookup table / answer store is rejected ("NOT A SUBMISSION").
3. **Inspects the true structure** — reads the per-tensor quantization layout
   from the actual file (not a claim) and records it house-side.
4. **Measures fidelity** against the lane's BF16 reference (single-pass, long-mode,
   n=4,999 stratified positions) on the eval box (a consumer card, e.g. a 5090).
5. **Posts a receipt** (hash-bound, replayable) and updates the board.

Only house-verified facts enter the receipt — the file hash, the measured size,
the measured fidelity, the inspected tensor layout, and the (verified) source.
Miner-claimed fields like imatrix or quantization command are NOT trusted or
scored.

## Acceptance

A recipe is **accepted** if its **worst-component mean KL** is within the bound
(`ACCEPT_KL = 0.02`) vs the reference — gated per component (prose, code,
multilingual, technical), so no component is a thin weak spot. **top-1 agreement
is reported, not gated.** The **crown** = the smallest accepted recipe; a
challenger dethrones only if smaller *and* accepted.

## Format expansion

Launch is GGUF / llama.cpp. The competition opens to other formats via
per-runtime adapters (vLLM → GPTQ/AWQ/FP8/NVFP4, ExLlama → EXL2), each gated on
that runtime reproducing the reference. See [`docs/format-open.md`](../../../docs/format-open.md).
