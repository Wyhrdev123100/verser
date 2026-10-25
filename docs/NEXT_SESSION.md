# Wynr-1.0 — Next Session

## Pick-up Instructions (2026-10-23)

1. Read `docs/PROJECT_STATUS.md`.
2. Read `docs/TRAINING_LOG.md`.
3. Read `docs/RESUME_GUIDE.md`.
4. Check Git status and current commit.
5. Check `checkpoints/` and `checkpoints/tokenizer/wynr-1.0/` contents.
6. Do NOT re-init tokenizer or re-train from scratch unless instructed.

## Current State

- Smoke training completed: 6 steps (config `configs/smoke.yaml`), best checkpoint at step 6.
- Resume test passed: continued from step 6 to step 8.
- All unit tests pass (20 tests).

## Immediate Next Actions

1. Choose next config: keep `configs/smoke.yaml` for smoke, use `configs/base.yaml` for prototype.
2. If GPU available (Kaggle), run:
   `python scripts/train.py --config configs/base.yaml --mode train`
3. If resuming here: `python scripts/train.py --config configs/smoke8.yaml --mode resume --checkpoint checkpoints/wynr_step00006.ckpt`
4. After training: run `python scripts/evaluate.py --checkpoint ...`, `python scripts/generate.py --checkpoint ...`.

## Environment Notes

- Local workspace: no GPU (CPU only).
- Do not claim training on GPU unless actually run on Kaggle.
