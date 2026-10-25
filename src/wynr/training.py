"""Wynr-1.0 PyTorch training loop: train step, validation, optimizer schedule."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.wynr.config import WynrConfig


class Trainer:
    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        scaler: Optional[Any],
        config: WynrConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.device = device

    def train_step(self, ids: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        logits, aux = self.model(ids, return_aux=True)
        loss = self.model.cross_entropy(logits, ids) + aux * self.model.config.moe_loss_weight
        if self.config.training.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        return {
            "loss": float(loss.detach().item()),
            "aux_loss": float(aux.item()) if torch.is_tensor(aux) else float(aux),
        }

    def validate_step(self, ids: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            logits, aux = self.model(ids, return_aux=True)
            loss = self.model.cross_entropy(logits, ids)
        return {
            "val_loss": float(loss.item()),
            "aux_loss": float(aux.item()) if torch.is_tensor(aux) else float(aux),
        }
