# Submissions — glimmer-30b

Each submission is one **recipe**: a JSON file naming an exact, real model file
of the lane's architecture, byte-addressed by `model_sha256`. Add it as
`recipes/<handle>.json` in a PR. See `EXAMPLE.json`.

## Format

| Field | Meaning |
|---|---|
| `model` | Model family, e.g. `muse-glimmer-30b` |
| `model_file` | The model filename (GGUF for launch) |
| `model_sha256` | SHA-256 of the **exact file** you submit — verified byte-for-byte |
| `quant` | Format / quant level, e.g. `Q6_K`, `Q8_0` |
| `format` | Container: `gguf` for launch (other formats come later) |
| `runtime` | Runtime that produced/serves it, e.g. `llama.cpp` |
| `runtime_version` | Exact runtime commit, for reproducibility |
| `note` | Free-form (no weight in scoring) |

## What the house does

1. **Validates the file** — must be a real, loadable GGUF of this model; a
   lookup table / answer store is rejected ("NOT A SUBMISSION").
2. **Serves it** and measures fidelity against the lane's BF16 reference
   (top-1 + KL over the public corpus).
3. **Posts a receipt** (hash-bound, replayable) and updates the board.

Accepted = top-1 ≥ 0.99 **and** within the KL bound. The **crown** = the
smallest accepted recipe; a challenger dethrones only if smaller *and* accepted.
