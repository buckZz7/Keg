# Board — glimmer-30b

Receipts are the source of truth. Only accepted recipes (KL within the bound in
every component vs the lane's BF16 reference) appear here; rejected attempts are
recorded under `receipts/` but hold nothing.

## Rewards

Today there is one reward: the **crown** (smallest accepted recipe). If more
reward tiers are ever added, they become new rows here — no new machinery.

| Reward | Current holder | Quant / format | Size (GB) | Fidelity (KL) | Fidelity (top-1) | Receipt |
|---|---|---|---|---|---|---|
| **Crown** (smallest accepted) | Muse Glimmer 30B | UD-Q6_K_XL / GGUF | 26.3 | 0.0093 | 0.974 | [ud-q6_k_xl.receipt.json](receipts/ud-q6_k_xl.receipt.json) |

## Recent receipts

| Quant / format | Size (GB) | Fidelity (KL) | Fidelity (top-1) | Verdict | Receipt |
|---|---|---|---|---|---|
| UD-Q6_K_XL / GGUF | 26.3 | 0.0093 | 0.974 | accepted (crown) | [ud-q6_k_xl.receipt.json](receipts/ud-q6_k_xl.receipt.json) |
| Q8_0 / GGUF | 29.6 | 0.0049 | 0.983 | accepted | [q8_0.receipt.json](receipts/q8_0.receipt.json) |

The gate is `ACCEPT_KL = 0.02` (worst-component mean KL), field-anchored and
calibrated by the ladder: the near-lossless tier (Q8 0.005, Q6 0.009) sits well
below it, and the lossy tier is rejected. Measured single-pass, long-mode,
n=4,999 stratified positions against the BF16 reference.
