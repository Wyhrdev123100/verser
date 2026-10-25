# Checkpoints

Default cadence: every 500 optimizer steps (configurable via `training.checkpoint_every`).

Structure: `runs/<run_id>/checkpoints/checkpoint-<step>/`

Files: model.pt, optimizer.pt, scheduler.pt, scaler.pt (optional), rng.pt, tokenizer/,
config.json, dataset_fingerprint.txt, manifest.json.

Manifest contains run_id, global_step, model_arch, model_config, backend, library_versions,
dataset_fingerprint, training_config, file checksums, status.

Safety: write to temp dir, verify, atomic rename, update `latest.txt` only after verification.
No overwrite of other runs' checkpoints. Retention configurable. Pinned checkpoints not auto-deleted.