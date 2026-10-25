"""Wynr-1.0 decoder-only Transformer with Mixture of Experts."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wynr.config import ModelConfig


class MoEExpert(nn.Module):
    """Independent expert MLP."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """MoE layer: router -> top-k experts -> weighted combine + aux load loss."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        moe_loss_weight: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_loss_weight = moe_loss_weight
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([MoEExpert(d_model, d_ff) for _ in range(num_experts)])
        self.n_heads = n_heads
        self.d_head = max(d_model // max(n_heads, 1), 1)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_gate = nn.Linear(d_model, d_ff)
        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_down = nn.Linear(2 * d_ff, d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.moe_drop = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        hd = self.d_head
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn = attn.masked_fill(~causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        return x + out  # pre-norm: LayerNorm applied to input before attention

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln2(x)
        g = F.gelu(self.ffn_gate(h))
        u = F.gelu(self.ffn_up(h))
        h = self.ffn_down(torch.cat([g, u], dim=-1))
        return x + h  # pre-norm: LayerNorm applied to input before FFN

    def _moe(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        router_logits = self.router(x)  # B, T, E
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # B, T, top_k
        weighted = topk_probs * topk_probs.clamp_min(1e-20)
        expert_outs = torch.zeros_like(x)
        aux = 0.0
        uniform = torch.full((1, self.num_experts,), 1.0 / self.num_experts, device=x.device, dtype=x.dtype)
        for b in range(B):
            for t in range(T):
                acc = torch.zeros(D, device=x.device, dtype=x.dtype)
                tok_probs = topk_probs[b, t]
                tok_idx = topk_idx[b, t]
                for j in range(self.top_k):
                    e = int(tok_idx[j])
                    out_e = self.experts[e](x[b, t : t + 1])
                    acc = acc + tok_probs[j] * out_e[0]
                expert_outs[b, t] = acc
                prob_tensor = probs[b, t]
                aux = aux + float((-(uniform[0] * (prob_tensor + 1e-9).log()).sum()).detach())
        aux_loss = (aux / max(B * T, 1)) * self.moe_loss_weight
        return expert_outs, aux_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        attn_out = self._attention(x)
        h = x + attn_out
        h = h + self._ffn(h)
        expert_out, aux_loss = self._moe(h)
        return expert_out, aux_loss

    def expert_stats(self, x: torch.Tensor) -> Dict[str, object]:
        """Return token counts routed per expert."""
        B, T, D = x.shape
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        counts = torch.zeros(self.num_experts, device=x.device)
        for b in range(B):
            for t in range(T):
                for j in range(self.top_k):
                    counts[int(topk_idx[b, t][j])] += 1
        return {
            "counts": [int(c) for c in counts.tolist()],
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_tokens": int(B * T),
        }


class WynrMoE(nn.Module):
    """Decoder-only Transformer with pre-norm, causal attention and MoE FFN."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.config = model_config
        vocab_size = model_config.vocab_size
        d_model = model_config.d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(model_config.max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [
                MoELayer(
                    d_model=d_model,
                    n_heads=model_config.n_heads,
                    d_ff=model_config.d_ff,
                    dropout=model_config.dropout,
                    num_experts=model_config.moe_num_experts,
                    top_k=model_config.moe_top_k,
                    moe_loss_weight=model_config.moe_loss_weight,
                )
                for _ in range(model_config.n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.pad_token_id = 3

    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
        return x

    def forward(
        self,
        ids: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Return (logits_for_loss, aux_loss); logits shape (B, T, V), position t predicts token t+1."""
        x = self._encode(ids)
        aux_total = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            aux_total = aux_total + aux
        logits = self.lm_head(x)
        if not return_aux:
            return logits
        return logits, aux_total

    def predict_next_token_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Return LM head output for last position (B, V)."""
        x = self._encode(ids)
        for layer in self.layers:
            x, _ = layer(x)
        return self.lm_head(x[:, -1, :])

    def cross_entropy(self, logits: torch.Tensor, ids: torch.Tensor, pad_token_id: int = 3) -> torch.Tensor:
        """Causal LM loss: logits[:, :-1] predicts ids[:, 1:]."""
        if ids.shape[1] < 2:
            return logits.sum() * 0.0
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_loss = F.cross_entropy(
            log_probs.reshape(-1, log_probs.size(-1)),
            ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
        )
        return token_loss

    def get_expert_stats(self, x: torch.Tensor) -> Dict[str, object]:
        if len(self.layers) == 0:
            return {"num_experts": 0, "top_k": 0, "counts": []}
        return self.layers[0].expert_stats(x)

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
