# Wynr-1.0 — Project Status

## Current Status (2026-10-23)

- **Phase**: Giai đoạn A (Smoke test) — phần lớn hoàn thành.
- **Environment**: local Linux (CPU-only, PyTorch 2.14.1+cpu, no CUDA). Kaggle GPU chưa được sử dụng trong phiên này.
- **Model**: WynrMoE (decoder-only Transformer + Mixture of Experts), random init (no pretrained weights).
- **Tokenizer**: custom character vocabulary (BOS=1, EOS=2, PAD=3, UNK=4, chars start at 5) trained from project data.

## Completed Items

| # | Item | Status |
|---|------|--------|
| 1 | Source code Transformer + MoE (`src/wynr/model.py`) | PASS |
| 2 | Random init weights, no pretrained | PASS |
| 3 | Custom tokenizer pipeline | PASS |
| 4 | Sample Vietnamese data generation / validation / dedup | PASS |
| 5 | Training loop + validation + checkpoint | PASS |
| 6 | Generation CLI (temperature / top-k / top-p) | PASS |
| 7 | Checkpoint save / resume / verify | PASS |
| 8 | Unit tests (`pytest tests/`) | PASS (20 passed) |
| 9 | Smoke train run (6 steps) + resume run (6 -> 8) | PASS |

## Smoke Test Results (config: `configs/smoke.yaml`)

- Training: 6 steps completed. Final train loss ~5.51, aux loss ~0.017.
- Checkpoints: `checkpoints/wynr_step00002.ckpt`, `wynr_step00004.ckpt`, `wynr_step00006.ckpt` (+ `_best` variants).
- Best checkpoint: `checkpoints/wynr_step00006_best.ckpt`.
- Tokenizer: `checkpoints/tokenizer/wynr-1.0/tokenizer.json`.
- Resume: from `wynr_step00006.ckpt` continued to `wynr_step00008.ckpt` (global_step 6 -> 8) — PASS.
- Verify: `python scripts/verify_checkpoint.py --checkpoint checkpoints/wynr_step00006.ckpt` — OK.

## Not Yet Done (requires Kaggle / user action)

- [ ] Run smoke train on Kaggle free GPU (Tesla T4).
- [ ] Phase B/C training with larger config (`configs/base.yaml`).
- [ ] Backup artifact to durable storage (Kaggle Dataset / local disk).
- [ ] Kaggle notebook execution in fresh kernel.

## Next Step

Xem `docs/NEXT_SESSION.md`.
