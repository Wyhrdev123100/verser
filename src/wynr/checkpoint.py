"""Wynr-1.0 checkpoint save / load with atomic write and full state."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover
    GradScaler = None  # type: ignore


def _new_scaler() -> Optional[Any]:
    try:
        return GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[Any],
    global_step: int,
    best_validation_loss: Optional[float],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    tokenizer_info: Dict[str, Any],
    source: str = "local",
) -> Dict[str, Any]:
    """Atomically save a full training checkpoint (atomic write via tmp + rename)."""
    ckpt: Dict[str, Any] = {
        "wynr_version": "1.0.0",
        "global_step": int(global_step),
        "best_validation_loss": best_validation_loss,
        "epoch": None,
        "model_config": model_config,
        "training_config": training_config,
        "tokenizer": tokenizer_info,
        "source": source,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": None,
            "numpy": None,
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": None,
        },
        "timestamp": float(time.time()),
    }
    import random as _random

    gauss_next = getattr(_random.Random(), "gauss_next", None)
    ckpt["rng"]["python"] = {
        "random": [int(x) for x in _random.Random().getstate()[1][:32]],
        "gauss_next": gauss_next,
    }
    try:
        ckpt["rng"]["numpy"] = [int(x) for x in np.random.get_state()[1][:32]]
    except Exception:
        ckpt["rng"]["numpy"] = None
    if torch.cuda.is_available():
        ckpt["rng"]["torch_cuda"] = torch.cuda.get_rng_state().tolist()

    path = str(path)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    info = {
        "path": path,
        "global_step": int(global_step),
        "best_validation_loss": best_validation_loss,
        "size_bytes": os.path.getsize(path),
    }
    return info


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt
