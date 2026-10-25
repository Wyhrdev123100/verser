"""CLI for Universal AI Training Platform."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
import yaml

from . import __version__
from .config import RunConfig
from .hardware import detect, recommend_config
from .models import REGISTRY
from .datasets.manager import DatasetManager
from .runs.manager import RunManager
from .checkpoint.manager import CheckpointManager
from .storage.manager import StorageManager
from .training.engine import TrainingEngine
from .export.manager import ExportManager
from .publish.manager import PublishManager
from .colab.workflow import detect_runtime, prepare_storage, sync_checkpoint, export_zip, find_latest_checkpoint

app = typer.Typer(help="Universal AI Training Platform CLI", add_completion=False)


@app.command()
def version():
    typer.echo(f"uatp {__version__}")


@app.command()
def hardware(mode: str = "balanced"):
    hw = detect()
    typer.echo(json.dumps(hw.to_dict() if hasattr(hw, "to_dict") else hw.__dict__, indent=2))
    rec = recommend_config(hw, mode=mode)
    typer.echo("\nRecommended:")
    typer.echo(json.dumps(rec, indent=2))


@app.command("models")
def models_cmd(action: str = typer.Argument("list")):
    if action == "list":
        for m in REGISTRY.list_models():
            typer.echo(f"{m.name}\t{m.family}\t{m.adapter}\t{m.description}")
        typer.echo("\nAdapters:")
        for a in REGISTRY.list_adapters():
            typer.echo(a)


@app.command("datasets")
def datasets_cmd(action: str = typer.Argument("inspect"), inspect_path: Optional[str] = None, hf: Optional[str] = None, text_column: str = "text"):
    dm = DatasetManager()
    if inspect_path:
        b = dm.load_local(inspect_path, text_column=text_column)
        typer.echo(json.dumps(b.to_dict_summary(), indent=2))
    elif hf:
        b = dm.load_hf(hf, text_column=text_column)
        typer.echo(json.dumps(b.to_dict_summary(), indent=2))
    else:
        typer.echo("Provide --inspect-path or --hf")


@app.command("train")
def train_cmd(config: str, mode: Optional[str] = None, resume: Optional[str] = None, max_steps: Optional[int] = None, runs_root: str = "./runs"):
    cfg = RunConfig.from_yaml(config)
    if mode:
        cfg.training.mode = mode
    engine = TrainingEngine(runs_root=runs_root)
    rid = engine.train(cfg, resume_from=resume, max_steps_override=max_steps)
    typer.echo(f"Run completed: {rid}")


@app.command("runs")
def runs_cmd(action: str = typer.Argument("list"), runs_root: str = "./runs"):
    rm = RunManager(runs_root)
    if action == "list":
        for r in rm.list_runs():
            typer.echo(f"{r.run_id}\t{r.status}\tstep={r.global_step}\tmodel={r.model}")


@app.command("checkpoints")
def checkpoints_cmd(action: str = typer.Argument("list"), run_id: Optional[str] = None, path: Optional[str] = None, runs_root: str = "./runs"):
    cm = CheckpointManager(runs_root)
    if action == "list":
        cks = cm.list_checkpoints(run_id=run_id)
        for c in cks:
            typer.echo(f"{c.get('path')}\tstep={c.get('global_step')}\tvalid={c.get('valid')}\ttype={c.get('checkpoint_type')}\tsize={c.get('size_bytes')}")
    elif action == "inspect":
        if not path:
            typer.echo("path required")
            raise typer.Exit(1)
        typer.echo(json.dumps(cm.inspect(path), indent=2))
    elif action == "validate":
        if not path:
            typer.echo("path required")
            raise typer.Exit(1)
        ok, errs = cm.validate(path)
        typer.echo(f"valid={ok}")
        if errs:
            for e in errs:
                typer.echo(f"  - {e}")
        if not ok:
            raise typer.Exit(1)
    elif action == "resume":
        typer.echo("Use `train --resume PATH` to resume.")
    elif action == "export":
        if not path:
            typer.echo("path required")
            raise typer.Exit(1)
        out = path.rstrip("/") + "_export"
        em = ExportManager(runs_root)
        em.export(path, out)
        typer.echo(f"exported to {out}")
    else:
        typer.echo(f"unknown action {action}")


@app.command("import-checkpoint")
def import_ckpt_cmd(source: str, run_id: Optional[str] = None, runs_root: str = "./runs", step: Optional[int] = None):
    cm = CheckpointManager(runs_root)
    rm = RunManager(runs_root)
    if run_id is None:
        # create a run record placeholder
        rid = rm.new_run_id()
    else:
        rid = run_id
    if not rm.exists(rid):
        rm.create({"imported": True}, run_id=rid)
    target = cm.import_checkpoint(source, rid, step_override=step)
    typer.echo(f"imported to {target}")


@app.command("publish")
def publish_cmd(folder: str, repo_id: str, token: Optional[str] = None, private: bool = True, runs_root: str = "./runs"):
    pm = PublishManager()
    rid = pm.publish(folder, repo_id, token=token, private=private)
    typer.echo(f"published: {rid}")


@app.command("colab")
def colab_cmd(action: str = typer.Argument("status"), durable_dir: Optional[str] = None):
    typer.echo(f"runtime: {detect_runtime()}")
    if durable_dir:
        latest = find_latest_checkpoint(durable_dir)
        typer.echo(f"latest checkpoint: {latest}")
    else:
        gd = StorageManager.gdrive_root()
        typer.echo(f"gdrive_root: {gd}")


@app.command("export")
def export_cmd(ckpt: str, out_dir: str, runs_root: str = "./runs"):
    em = ExportManager(runs_root)
    em.export(ckpt, out_dir)
    typer.echo(f"exported to {out_dir}")


if __name__ == "__main__":
    app()
