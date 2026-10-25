"""Configuration models for runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    source: str = "huggingface"  # huggingface | local | custom
    model_id: str = ""
    model_type: str = "causal_lm"  # causal_lm | seq2seq | encoder | vision | custom
    adapter: Optional[str] = None  # peft adapter type: lora, qlora, None
    dtype: str = "auto"  # auto | fp32 | fp16 | bf16
    trust_remote_code: bool = False
    revision: Optional[str] = None
    architecture: Optional[str] = None
    tokenizer_id: Optional[str] = None


@dataclass
class DatasetConfig:
    source: str = "huggingface"  # huggingface | local | custom
    dataset_id: str = ""
    config_name: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    instruction_column: Optional[str] = None
    response_column: Optional[str] = None
    validation_split: Optional[float] = None  # fraction for val
    streaming: bool = False
    max_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    mode: str = "finetune"  # finetune | continued_pretrain | pretrain | peft
    epochs: int = 1
    max_steps: Optional[int] = None
    micro_batch_size: Optional[int] = None
    gradient_accumulation: Optional[int] = None
    sequence_length: Optional[int] = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "linear"
    mixed_precision: str = "auto"  # auto | none | fp16 | bf16
    gradient_checkpointing: bool = False
    dataloader_workers: int = 0
    seed: int = 42
    checkpoint_every: int = 500
    eval_every: Optional[int] = None
    log_every: int = 10


@dataclass
class StorageConfig:
    root: str = "./runs"
    durable: Optional[str] = None  # path to durable storage (gdrive etc.)
    hf_repo: Optional[str] = None
    keep_last: int = 3
    pin_checkpoints: List[str] = field(default_factory=list)


@dataclass
class HardwareConfig:
    mode: str = "balanced"  # safe | balanced | performance
    force_cpu: bool = False


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        def _dc(dc_cls, dd):
            if dd is None:
                return dc_cls()
            return dc_cls(**{k: v for k, v in dd.items() if k in dc_cls.__dataclass_fields__})

        return cls(
            run_id=d.get("run_id"),
            model=_dc(ModelConfig, d.get("model")),
            dataset=_dc(DatasetConfig, d.get("dataset")),
            training=_dc(TrainingConfig, d.get("training")),
            storage=_dc(StorageConfig, d.get("storage")),
            hardware=_dc(HardwareConfig, d.get("hardware")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)