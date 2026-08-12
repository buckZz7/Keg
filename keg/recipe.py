"""Keg — the submission: a compression recipe.

A recipe is everything the house needs to reproduce the run: the exact
model file (by hash), the format/quant, and the runtime that produced it.
The race metric is SIZE — how small the model footprint is while still
holding the model's behavior. The house MEASURES the real file size into
the receipt; a miner's claim is advisory only.

A recipe is accepted — and competes for the crown — only if its next-token
distribution stays within a KL bound of the model's own BF16 reference,
measured under the field-standard long-mode KLD (deep top-k, long context).
KL divergence is the field's fidelity metric of record ("Accuracy is Not All
You Need", Fireworks, llama-perplexity): it is highly correlated with answer
flips, and is the one that separates near-lossless quants from lossy ones on a
hard, diverse corpus. top-1 agreement is reported in the receipt (human-
readable) but is not a pass/fail gate. Below the KL bound you are still the
model; above it, you aren't. There are no fidelity tiers.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class Recipe(BaseModel):
    """A self-contained compression recipe (the miner's submission)."""

    model: str = Field(description="Model family, e.g. muse-glimmer-30b")
    model_file: str = Field(description="Model filename (GGUF / NVFP4 / ...)")
    model_sha256: str = Field(description="SHA-256 of the model file")
    quant: str = Field(description="Format / quant level, e.g. Q6_K, Q8_0, NVFP4, MXFP4")
    format: str = Field(default="gguf", description="Container format. GGUF-only for launch (one runtime the house can load/serve/reproduce); other formats (fp8, nvfp4, mxfp4, safetensors) can be added as their runtimes + verification are.")
    runtime: str = Field(default="llama.cpp", description="Runtime that produced/serves it")
    runtime_version: str = Field(default="", description="Exact runtime version/commit")
    note: str = Field(default="", description="Free-form miner note (no weight in scoring)")

    def fingerprint(self) -> str:
        """Hash the recipe — the receipt binds to it."""
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# Fidelity threshold for ACCEPTANCE. A recipe is a valid submission — and only
# competes for the crown — if its mean KL divergence from the model's own BF16
# reference stays within this bound, measured under long-mode KLD (deep
# top-k, long context). KL is the primary gate (no top-1 floor).
#
# The value is anchored to the field's near-lossless KLD range (~0.05-0.1 nats:
# smcleod/mlx-kld "typical 4-bit" 1e-2..5e-2, "substantial" >1e-1; Fireworks
# high-quality deployments < 7e-3) AND finalized by the calibration ladder on
# the box — set where the near-lossless cluster (Q8/Q6) separates from the
# lossy control (Q4). PENDING final ladder number; do not ship receipts on a
# placeholder.
ACCEPT_KL = 0.10


def accepted(top1_match: float | None, kl_mean: float | None) -> bool:
    """A recipe is accepted (and eligible for the crown) only if its mean KL
    stays within the bound. KL is the precision gate; top-1 is reported but
    not gated (redundant given KL's tight correlation with flips)."""
    if kl_mean is None:
        return False
    return kl_mean <= ACCEPT_KL
