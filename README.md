# Universal AI Training Platform (UATP)

Open-source, multi-model AI training platform with reliable checkpointing
and resume. Competes on experience, GPU efficiency and training management.

## Features

- Multi-model adapters (toy GPT for smoke, Hugging Face causal/seq2seq/encoder/vision)
- GPU/CPU hardware profiling and config recommendations (safe/balanced/performance)
- Checkpoint manager with manifest + checksums, atomic save, retention
- Resume from local checkpoints, imported checkpoints, Colab durable storage
- Dataset manager (local JSONL/CSV, Hugging Face datasets)
- Export model weights + tokenizer
- Hugging Face Hub publishing
- CLI and minimal web dashboard

## Install

```bash
pip install -e .
```

## Quick start (CPU smoke)

```bash
mkdir -p data/train
python -c "import json; [open('data/train/train.jsonl','a').write(json.dumps({'text':f'sentence {i}'})+'\n') for i in range(64)]"
uatp hardware
uatp train configs/smoke.yaml
uatp checkpoints list --run-id <run_id>
uatp train configs/smoke.yaml --resume runs/<run_id>/checkpoints/checkpoint-<step> --max-steps <n>
```

## CLI

```bash
uatp hardware
uatp models
uatp datasets --inspect-path data/train/train.jsonl
uatp runs
uatp checkpoints inspect --path runs/.../checkpoint-4
uatp checkpoints validate --path runs/.../checkpoint-4
uatp import-checkpoint --source ckpt.zip --run-id myrun
uatp export --ckpt runs/.../checkpoint-4 --out-dir artifacts/export
uatp publish --folder artifacts/export --repo-id user/model
uatp colab
```

## Tests

```bash
pytest tests/
```

## Colab

See `notebooks/UATP_Colab.ipynb`.

## Status

Implemented: hardware profiler, dataset manager, model adapters, training engine,
checkpoint manager, run manager, storage, colab helpers, export, publish, CLI,
minimal web UI, tests.

Not yet: full Gradio UI, multi-user queue/isolation, comprehensive benchmarks,
all model families beyond listed adapters.
