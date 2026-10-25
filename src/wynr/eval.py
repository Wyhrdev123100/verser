"""Wynr-1.0 evaluation helpers: loss, perplexity, throughput."""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_loss(model, ids: torch.Tensor, pad_token_id: int = 3) -> float:
    logits, _ = model(ids, return_aux=True)
    loss = model.cross_entropy(logits, ids, pad_token_id=pad_token_id)
    return float(loss.item())


def perplexity(loss: float) -> float:
    return float(torch.exp(torch.tensor(loss)).item()) if loss < 700 else float("inf")


def measure_tokens_per_second(loss: float, elapsed_s: float, num_tokens: int) -> float:
    if elapsed_s <= 0 or num_tokens <= 0:
        return 0.0
    return float(num_tokens) / float(elapsed_s)
