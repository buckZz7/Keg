"""Keg — the submission: a compression recipe.

A recipe is everything the house needs to reproduce the run: the exact
model file (by hash), the format/quant, and the runtime that produced it.
The race metric is SIZE — how small the model footprint is while still
holding the model's behavior. The house MEASURES the real file size into
the receipt; a miner's claim is advisory only.

A recipe is accepted — and competes for the crown — only if it holds >= 0.99
top-1 against the model's own BF16 reference. Below that it is rejected.
There are no fidelity tiers: you are either still the model, or you aren't.
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
    format: str = Field(default="gguf", description="Container: gguf, nvfp4, mxfp4, fp8, safetensors")
    runtime: str = Field(default="llama.cpp", description="Runtime that produced/serves it")
    runtime_version: str = Field(default="", description="Exact runtime version/commit")
    note: str = Field(default="", description="Free-form miner note (no weight in scoring)")

    def fingerprint(self) -> str:
        """Hash the recipe — the receipt binds to it."""
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# Fidelity threshold for ACCEPTANCE and the CROWN. A recipe is only a valid
# submission — and only competes for the crown — if it holds >= this top-1
# match against the model's own BF16 reference. Below it, rejected.
#
# This is the anti-free-ride: holding the model's behavior is not free. Most
# off-the-shelf low quants (Q4_K_M class) land at 97-98% and get rejected.
ACCEPT_MIN = 0.99


def accepted(top1_match: float) -> bool:
    """A recipe is accepted (and eligible for the crown) only at >= 0.99."""
    return top1_match >= ACCEPT_MIN
