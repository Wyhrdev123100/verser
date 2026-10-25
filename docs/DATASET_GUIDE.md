# Wynr-1.0 — Dataset Guide

## Synthetic data

- Generator: `scripts/generate_data.py`
- Validator: `scripts/validate_data.py`
- Deduplicator: `scripts/deduplicate_data.py`
- Current data: `data/train/train.jsonl`, `data/processed/train.jsonl`, `data/stats.json`

## Format

- JSONL, UTF-8, fields: `{"text": "...", "topic": "...", "source": "synthetic_v1"}`.

## Manifest

- See `data/manifest.json` (to be finalized) and `artifacts/manifest.json`.

## Split

- train/validation split: deterministic seed 42 (see `src/wynr/data_utils.split_train_val`).
- Current smoke: train_file `data/train/train.jsonl`.
