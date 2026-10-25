"""Minimal web dashboard (FastAPI)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ..hardware import detect
from ..runs.manager import RunManager
from ..checkpoint.manager import CheckpointManager

app = FastAPI(title="Universal AI Training Platform")


@app.get("/api/hardware")
def api_hardware():
    hw = detect()
    return hw.to_dict() if hasattr(hw, "to_dict") else hw.__dict__


@app.get("/api/runs")
def api_runs(runs_root: str = "./runs"):
    rm = RunManager(runs_root)
    return [r.__dict__ for r in rm.list_runs()]


@app.get("/api/checkpoints")
def api_checkpoints(runs_root: str = "./runs", run_id: str | None = None):
    cm = CheckpointManager(runs_root)
    return cm.list_checkpoints(run_id=run_id)


@app.get("/", response_class=HTMLResponse)
def index():
    hw = detect()
    rm = RunManager("./runs")
    runs = rm.list_runs()
    rows = "".join(f"<tr><td>{r.run_id}</td><td>{r.status}</td><td>{r.global_step}</td><td>{r.model}</td></tr>" for r in runs)
    return f"""<!doctype html><html><head><title>UATP</title></head><body>
<h1>Universal AI Training Platform</h1>
<p>Device: {hw.device} | CPU cores: {hw.cpu_count} | RAM: {hw.total_ram_gb}GB</p>
<h2>Runs</h2>
<table border=1><tr><th>Run</th><th>Status</th><th>Step</th><th>Model</th></tr>{rows}</table>
<p>Use CLI for training. This dashboard is read-only.</p>
</body></html>"""


def create_app():
    return app
