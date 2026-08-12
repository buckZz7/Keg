# Adversarial testing — beating the crown (2026-08-12)

Goal: from a miner's perspective, find any quant recipe **smaller than the crown**
(UD-Q6_K_XL, 26.3 GB) that passes the gate (worst-component KL ≤ 0.02).

Method: every candidate measured against the stored BF16 reference with the
single-pass harness (`tools/harness/`), n=4,999 stratified long-context positions,
all on an A100-80GB. Candidates are real `unsloth/Muse-Glimmer-30B-GGUF` files
(intake-recorded sha256 + bytes).

## Result: the crown holds. No candidate passes the gate.

| Quant | Size (GB) | kl_mean | **kl_max (worst comp)** | Pass 0.02? |
|---|---|---|---|---|
| **UD-Q6_K_XL (crown)** | 26.3 | 0.0093 | **0.0165** (multilingual) | ✅ |
| UD-Q5_K_XL | 21.8 | 0.0275 | 0.0309 (prose) | ❌ |
| UD-Q5_K_L | 19.8 | 0.0420 | 0.0477 (prose) | ❌ |
| UD-Q5_K_M | 19.2 | 0.0464 | 0.0524 (prose) | ❌ |
| UD-Q4_K_XL | 15.9 | 0.1519 | 0.1745 (prose) | ❌ |
| ours-calib-Q5_K_M (imatrix on our corpus) | 19.8 | 0.0459 | 0.0515 (prose) | ❌ |

## The four miner strategies

1. **Drop a tier (Q5/Q4).** All standard Q5 and Q4 quants fail the gate. The gate
   cleanly separates Q6 (0.0165, pass) from Q5_K_XL (0.0309, fail) — the crown is
   effectively the smallest standard quant that still holds the model.
2. **KLD-guided low-bitwidth (SLQ).** Assessed infeasible at the *KL* gate. Q4
   measures 0.17 (~9× over). The field's "task-lossless at 3.3 bpw" (SLQ)
   optimizes task accuracy (MMLU), not KL on a high-entropy corpus; KL is far more
   sensitive than task accuracy, so a Q4-class model cannot approach the gate.
3. **Corpus-calibrated quantization (imatrix on our corpus).** Produced a ~2% KL
   change vs Unsloth's own calibration — effectively no edge. KL divergence is
   intrinsic to the quantization error, not fixable by calibration-data choice.
   Still ~2.5× over the gate.
4. **Iterate (free submissions).** The ladder itself is the iteration; no smaller
   candidate approached the gate.

## Observations

- The **binding (worst) component shifts to prose** as quants degrade
  (multilingual was worst at Q8/Q6; prose is worst at Q5/Q4). The per-component
  gate is doing real work — a miner can't win by shoring up code/technical and
  letting prose drift.
- **Stock llama.cpp cannot produce the `_XL` quants** the crown uses (Q6_K_XL
  isn't a llama-quantize ftype) — that's Unsloth's tooling. So the crown's exact
  recipe isn't trivially reproducible with vanilla tooling.

## Update (same day): the crown was beatable — and now it's Q6_K

The first crown (UD-Q6_K_XL, 26.3 GB) was **not the smallest passing recipe**:
a **standard Q6_K (22.9 GB) passes the gate** (worst-comp KL 0.0199, thin margin)
and is 3.4 GB smaller. So the house baseline was sub-optimal — it crowned the
only 6-bit in the vendor repo without testing the plain Q6_K. The crown moves to
**Q6_K (22.9 GB)**.

Also tested and rejected: **mixed-precision / error-placement** (F16 embedding +
output projection, quantized Q5/Q4 body) — it was *worse* than the uniform quants
(Q5 body 0.066, Q4 body 0.20) and larger; the body drives prose degradation, so
protecting emb/output tensors doesn't help on this model.

**Beating the new king (Q6_K, 22.9 GB)** requires a smaller passing recipe
(< 22.9 GB, worst-comp KL ≤ 0.02). The next standard tier (Q5_K_XL, 21.8 GB)
fails at 0.031, calibration gives no gain, so it needs genuine craft — a smarter
per-tensor precision mix, a Q5.5-class recipe, a research-grade method (SLQ /
GPTQ), or lossless packing. That is the innovation the benchmark is built to
reward.

## Conclusion

The gate is sound and the leaderboard is honest. The crown was beatable by a
standard quant (Q6_K) — the house just hadn't tested it. Beating *that* now
requires real quantization craft, which is exactly the race the benchmark exists
to reward.
