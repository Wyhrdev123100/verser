# Architecture

Modules:
- `uatp.config`: RunConfig and sub-configs.
- `uatp.hardware`: detection + recommendations.
- `uatp.models`: registry + adapters.
- `uatp.datasets`: DatasetManager + bundles.
- `uatp.training.engine`: training loop.
- `uatp.checkpoint`: manifest + manager.
- `uatp.runs`: run records.
- `uatp.storage`: local/gdrive/hf.
- `uatp.colab`: workflow helpers.
- `uatp.export`: export manager.
- `uatp.publish`: HF publishing.
- `uatp.ui`: FastAPI dashboard.
- `uatp.cli`: typer CLI.