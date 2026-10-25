"""Wynr-1.0 text generation with temperature, top-k, top-p."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.wynr.tokenizer import WynrTokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer: WynrTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = "cpu",
    max_context: int = 256,
) -> Tuple[str, List[int]]:
    """Generate continuation. Returns (generated_text, full_token_ids)."""
    model.eval()
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id

    seq = [i for i in tokenizer.encode(prompt) if i != pad_id]
    generated: List[int] = []
    for _ in range(max_new_tokens):
        if len(seq) > max_context:
            seq = seq[-max_context:]
        input_ids = torch.tensor([seq], device=device, dtype=torch.long)
        logits = model.predict_next_token_logits(input_ids)[0]
        logits = logits / max(float(temperature), 1e-8)
        if top_k is not None:
            kth = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, kth, dim=-1)
            threshold = values[..., -1:]
            logits = torch.where(logits >= threshold, logits, torch.full_like(logits, float("-inf")))
        probs = F.softmax(logits, dim=-1)
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            keep[..., -1] = True
            indices = torch.where(keep)[0]
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(1, indices.unsqueeze(1), sorted_probs.gather(1, indices.unsqueeze(1)))
            probs = new_probs
        next_id = int(torch.multinomial(probs, 1).item())
        if next_id in (bos_id, eos_id, pad_id, unk_id):
            break
        generated.append(next_id)
        seq.append(next_id)
        if len(generated) >= max_new_tokens:
            break
    full_ids = list(tokenizer.encode(prompt))
    return tokenizer.decode(generated), full_ids
