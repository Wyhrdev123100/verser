# Wynr-1.0 — Resume Guide

## How to resume training

```bash
python scripts/train.py --config configs/smoke.yaml --mode resume --checkpoint checkpoints/wynr_step00006.ckpt
```

## Requirements for a valid resume

1. Checkpoint file exists.
2. Model config compatible (architecture fields must match).
3. Tokenizer version/hash compatible (custom char vocab in `checkpoints/tokenizer/wynr-1.0/tokenizer.json`).
4. Optimizer / scheduler / scaler state restored.
5. global_step and best_validation_loss restored.

## If checkpoint is missing

- Do NOT silently retrain from scratch.
- Find backup: `scripts/save_session.py`, Kaggle dataset version, or local disk.
- Restore: `python scripts/restore_session.py --backup-dir artifacts/backups/<...>`

## If only inference checkpoint exists

- Cannot fully restore optimizer state from inference checkpoint; retrain or use training checkpoint.
