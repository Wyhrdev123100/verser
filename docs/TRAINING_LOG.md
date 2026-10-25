# Wynr-1.0 — Training Log

## 2026-10-23 — Session local smoke test

- Mode: train
- Config: configs/smoke.yaml (max_steps=6, batch_size=2, 1 layer, d_model=64, moe 2 experts top-k=1)
- Data: data/train/train.jsonl (40 generated samples, 26 clean after dedup)
- Tokenizer: checkpoints/tokenizer/wynr-1.0/tokenizer.json (custom char vocab)
- Result: COMPLETED
- Final train loss: ~5.51, aux MoE loss: ~0.017
- Checkpoints saved: wynr_step00002(.ckpt/_best), wynr_step00004, wynr_step00006
- Best checkpoint: wynr_step00006_best.ckpt

## 2026-10-23 — Session local resume test

- Mode: resume
- From: checkpoints/wynr_step00006.ckpt
- To: wynr_step00008.ckpt (final)
- Result: global_step advanced 6 -> 8
- Resume verified OK.
