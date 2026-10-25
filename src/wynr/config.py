"""Configuration management for Wynr-1.0."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "model": {
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 256,
        "vocab_size": 512,
        "max_seq_len": 128,
        "dropout": 0.1,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_loss_weight": 0.01,
    },
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "max_steps": 2000,
        "eval_every_n_steps": 100,
        "warmup_steps": 100,
        "scheduler": "cosine",
        "mixed_precision": True,
        "gradient_clipping": 1.0,
        "seed": 42,
    },
    "data": {
        "max_seq_len": 128,
        "train_file": "data/train.jsonl",
        "validation_file": "data/val.jsonl",
    },
    "tokenizer": {
        "type": "sentencepiece_bpe",
        "version": 1,
        "vocab_size": 512,
        "source": "synthetic_v1",
    },
}


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    vocab_size: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_loss_weight: float = 0.01


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 2000
    eval_every_n_steps: int = 100
    warmup_steps: int = 100
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    max_seq_len: int = 128
    train_file: str = "data/train.jsonl"
    validation_file: str = "data/val.jsonl"


@dataclass
class TokenizerConfig:
    type: str = "sentencepiece_bpe"
    version: int = 1
    vocab_size: int = 512
    source: str = "synthetic_v1"


@dataclass
class WynrConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "tokenizer": asdict(self.tokenizer),
        }


def _build(datad: Dict[str, Any]) -> WynrConfig:
    model = ModelConfig(**{k: v for k, v in datad.get("model", {}).items()})
    training = TrainingConfig(**{k: v for k, v in datad.get("training", {}).items()})
    data = DataConfig(**{k: v for k, v in datad.get("data", {}).items()})
    tokenizer = TokenizerConfig(**{k: v for k, v in datad.get("tokenizer", {}).items()})
    return WynrConfig(model=model, training=training, data=data, tokenizer=tokenizer)


def load_config(path: Optional[str] = None) -> WynrConfig:
    """Load Wynr-1.0 configuration from YAML, falling back to safe defaults."""
    cfg_path = Path(path) if path else Path("configs/smoke.yaml")
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            raw = loaded
    merged = copy.deepcopy(DEFAULTS)
    for section in ("model", "training", "data", "tokenizer"):
        if section in raw and isinstance(raw[section], dict):
            merged[section].update(raw[section])
    return _build(merged)


def save_config(config: WynrConfig, path: str) -> None:
    payload = {"wynr_version": "1.0.0", "config": config.as_dict()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
    tmp.replace(p)


def config_compatible(saved: Dict[str, Any], current: WynrConfig) -> tuple:
    """Return (compatible, reason). Checks that non-architectural settings may
    differ, while architectural fields must match."""
    saved_model = saved.get("model_config") or saved.get("model") or {}
    saved_training = saved.get("training_config") or saved.get("training") or {}

    arch_keys = [
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "vocab_size",
        "max_seq_len",
        "moe_num_experts",
        "moe_top_k",
    ]
    for key in arch_keys:
        saved_val = saved_model.get(key)
        cur_val = getattr(current.model, key, None)
        if saved_val is None or cur_val is None or int(saved_val) != int(cur_val):
            return False, f"Model config mismatch on field '{key}': saved={saved_val} current={cur_val}"

    return True, "ok"
