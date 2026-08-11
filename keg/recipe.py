"""Keg — the submission: a compression recipe.

A recipe is everything the house needs to reproduce the run: the exact
model file (by hash), the format/quant, and the runtime that produced it.
The race metric is SIZE — how small the model footprint is while still
holding the model's behavior. The house MEASURES the real file size into
the receipt; a miner's claim is advisory only.
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
    quant: str = Field(description="Format / quant level, e.g. Q4_K_M, Q6_K, NVFP4, MXFP4")
    format: str = Field(default="gguf", description="Container: gguf, nvfp4, mxfp4, fp8, safetensors")
    runtime: str = Field(default="llama.cpp", description="Runtime that produced/serves it")
    runtime_version: str = Field(default="", description="Exact runtime version/commit")
    note: str = Field(default="", description="Free-form miner note (no weight in scoring)")

    def fingerprint(self) -> str:
        """Hash the recipe — the receipt binds to it."""
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# Fidelity bands vs the model reference (top-1 match over the held-out
# corpus, measured against the model's own BF16 next-token distributions).
# Provisional thresholds — recalibrated from a ladder measurement on the
# box (the first job on a new model). Revised only by measurement, never
# by fiat. The CROWN requires band A: the smallest recipe that is still
# essentially the real model.
#
#   A  >= 0.99  indistinguishable  (Q6 / Q8 / FP8 class — serving-grade)
#   B  0.97–0.99  acceptable        (Q5 / Q4_K_M class — the download king)
#   C  0.90–0.97  the cliff         (Q3 / Q2 class — enthusiast floor)
#   below 0.90   rejected           (Q1 / IQ1 = broken)
BANDS = {
    "A": {"top1_min": 0.99, "label": "indistinguishable"},
    "B": {"top1_min": 0.97, "label": "acceptable"},
    "C": {"top1_min": 0.90, "label": "the cliff"},
}

# Eligibility floor: below this top-1 a recipe is not a valid submission
# (the model is no longer recognizably itself — the Q1/broken tier).
FLOOR = 0.90


def band_for(top1_match: float) -> str:
    """Band from a measured top-1 vs the model reference."""
    for name in ("A", "B", "C"):
        if top1_match >= BANDS[name]["top1_min"]:
            return name
    return "R"  # R = rejected (below the floor)


def eligible(top1_match: float) -> bool:
    """A recipe above the floor is accepted (scored); below it is not."""
    return top1_match >= FLOOR
