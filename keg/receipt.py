"""Keg — the receipt: the run, hash-bound and replayable.

The receipt is the trustless record: recipe hash, model hash, the fidelity
gate result, the HOUSE-MEASURED size (the race metric), the eval box
fingerprint, and the epoch. A receipt that doesn't replay is not a receipt.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from typing import Any, Dict, Optional

from .recipe import Recipe, accepted


def _box_fingerprint() -> str:
    """Fingerprint of the eval box (GPU + driver + host)."""
    info = {"machine": platform.machine(), "node": platform.node()}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        info["gpu"] = out
    except Exception:
        pass
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]


def build_receipt(
    recipe: Recipe,
    fidelity: Dict[str, Any],
    size_bytes: int,
    epoch: str,
    num_params: Optional[int] = None,
    crown: Optional[Dict[str, Any]] = None,
) -> dict:
    """Assemble the receipt for a completed compression race run.

    size_bytes is MEASURED by the house from the actual model file. bpw is
    computed for a normalized cross-format comparison.
    """
    size_gb = size_bytes / (1024 ** 3)
    bpw = (size_bytes * 8 / num_params) if num_params else None

    receipt = {
        "schema": "keg/receipt-v1",
        "receipt_sha256": "",
        "epoch": epoch,
        "box": _box_fingerprint(),
        "recipe": recipe.model_dump(),
        "recipe_fingerprint": recipe.fingerprint(),
        "fidelity": fidelity,
        "accepted": accepted(fidelity.get("kl_mean"),
                             fidelity.get("kl_max_component")),
        "size": {
            "size_bytes": size_bytes,
            "size_gb": round(size_gb, 3),
            "bpw": round(bpw, 3) if bpw is not None else None,
        },
        "gate_passed": accepted(fidelity.get("kl_mean"),
                                fidelity.get("kl_max_component")),
        "ts": time.time(),
    }
    if crown is not None:
        receipt["crown"] = crown
    payload = json.dumps({k: v for k, v in receipt.items() if k != "receipt_sha256"}, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return receipt


def verify_receipt(receipt: dict) -> bool:
    """Replay the receipt: recompute the hash and compare."""
    payload = json.dumps({k: v for k, v in receipt.items() if k != "receipt_sha256"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest() == receipt.get("receipt_sha256")
