"""A tiny causal LM for smoke testing and offline vertical slices."""
from __future__ import annotations

import torch
import torch.nn as nn


class ToyGPT(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 32, n_layers: int = 2, n_heads: int = 2, max_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        # causal mask
        mask = torch.triu(torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, loss=loss)

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            import json
            json.dump({"vocab_size": self.vocab_size, "d_model": self.tok_emb.embedding_dim, "max_len": self.max_len}, f)

    @classmethod
    def from_pretrained(cls, path):
        import json, os
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        m = cls(vocab_size=c["vocab_size"], d_model=c["d_model"], max_len=c["max_len"])
        m.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu"))
        return m
