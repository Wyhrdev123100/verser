import os
import yaml
from typer.testing import CliRunner
from uatp.cli import app
from uatp.config import RunConfig
from uatp.training.engine import TrainingEngine


def test_resume_step_increases(tmp_path):
    # prepare data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    p = data_dir / "train.jsonl"
    import json
    with open(p, "w") as f:
        for i in range(32):
            f.write(json.dumps({"text": f"sentence {i}"}) + "\n")
    cfg_dict = {
        "model": {"source": "toy", "model_id": "toy-gpt", "model_type": "causal_lm", "dtype": "fp32"},
        "dataset": {"source": "local", "dataset_id": str(p), "text_column": "text", "max_samples": 32},
        "training": {"mode": "pretrain", "epochs": 1, "max_steps": 4, "micro_batch_size": 4,
                     "gradient_accumulation": 2, "sequence_length": 8, "learning_rate": 0.001,
                     "checkpoint_every": 2, "log_every": 1},
        "storage": {"root": str(tmp_path / "runs")},
        "hardware": {"mode": "balanced", "force_cpu": True},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f)
    cfg = RunConfig.from_yaml(cfg_path)
    engine = TrainingEngine(runs_root=str(tmp_path / "runs"))
    rid1 = engine.train(cfg)
    # find latest checkpoint
    from uatp.checkpoint.manager import CheckpointManager
    cm = CheckpointManager(str(tmp_path / "runs"))
    ckpt = cm.find_latest_valid(run_id=rid1)
    assert ckpt is not None
    step1 = cm.inspect(ckpt)["global_step"]
    # resume
    cfg2 = RunConfig.from_yaml(cfg_path)
    rid2 = engine.train(cfg2, resume_from=ckpt, max_steps_override=step1 + 2)
    rr_path = tmp_path / "runs" / rid2 / "run.json"
    import json as _j
    d = _j.load(open(rr_path))
    assert d["global_step"] > step1
    assert d["status"] == "completed"