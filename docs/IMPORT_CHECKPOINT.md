# Import Checkpoint

Supports folder, ZIP, Hugging Face repo (via storage helpers). Safe extraction with path traversal protection.

Compatibility check before resume: model_arch, model_config, manifest status, checksums.
If only model weights present, offer fine-tuning from weights with warning (optimizer state not restored).