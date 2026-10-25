# Wynr-1.0 — Model Card (Draft)

## Wynr-1.0

- Type: decoder-only Transformer with Mixture of Experts (custom, random init).
- Parameters: smoke config ~ small (d_model=64, 1 layer, vocab ~40 chars).
- Training data: synthetic Vietnamese (v1).
- Tokenizer: custom char vocabulary (BOS/EOS/PAD/UNK), version 1.
- Objective: causal LM (cross-entropy) + MoE auxiliary load-balancing loss.

## Status

- Smoke phase complete. Not a general-purpose AI assistant yet.
- No pretrained LLM weights used.

## Intended use

- Vietnamese conversational model research / fine direction.
