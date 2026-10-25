"""Run management."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    status: str = "created"  # created | running | paused | completed | failed
    config: Dict[str, Any] = field(default_factory=dict)
    global_step: int = 0
    model: str = ""
    dataset: str = ""
    parent_run_id: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    resumed_from: Optional[str] = None


class RunManager:
    def __init__(self, runs_root: str = "./runs"):
        self.runs_root = os.path.abspath(runs_root)
        os.makedirs(self.runs_root, exist_ok=True)

    def new_run_id(self) -> str:
        return "run-" + uuid.uuid4().hex[:12]

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.runs_root, run_id)

    def create(self, config: Dict[str, Any], run_id: Optional[str] = None, parent_run_id: Optional[str] = None, resumed_from: Optional[str] = None) -> RunRecord:
        rid = run_id or self.new_run_id()
        rr = RunRecord(
            run_id=rid,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=config,
            model=(config.get("model", {}) or {}).get("model_id", ""),
            dataset=(config.get("dataset", {}) or {}).get("dataset_id", ""),
            parent_run_id=parent_run_id,
            resumed_from=resumed_from,
        )
        d = self.run_dir(rid)
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(d, "logs"), exist_ok=True)
        self.save(rr)
        return rr

    def save(self, rr: RunRecord) -> None:
        d = self.run_dir(rr.run_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump(asdict(rr), f, indent=2)

    def load(self, run_id: str) -> RunRecord:
        with open(os.path.join(self.run_dir(run_id), "run.json")) as f:
            d = json.load(f)
        return RunRecord(**d)

    def exists(self, run_id: str) -> bool:
        return os.path.isfile(os.path.join(self.run_dir(run_id), "run.json"))

    def list_runs(self) -> List[RunRecord]:
        res = []
        if not os.path.isdir(self.runs_root):
            return res
        for name in sorted(os.listdir(self.runs_root)):
            rp = os.path.join(self.runs_root, name, "run.json")
            if os.path.isfile(rp):
                try:
                    res.append(self.load(name))
                except Exception:
                    pass
        return res

    def update_status(self, run_id: str, status: str, global_step: Optional[int] = None, error: Optional[str] = None):
        rr = self.load(run_id)
        rr.status = status
        if global_step is not None:
            rr.global_step = global_step
        if error:
            rr.errors.append(error)
        self.save(rr)
        return rr

    def append_metric(self, run_id: str, metric: Dict[str, Any]):
        rr = self.load(run_id)
        rr.metrics.append(metric)
        self.save(rr)

    def append_checkpoint(self, run_id: str, path: str):
        rr = self.load(run_id)
        if path not in rr.checkpoints:
            rr.checkpoints.append(path)
        self.save(rr)
