"""Model registry and adapter abstractions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class ModelSpec:
    name: str
    family: str  # causal_lm, seq2seq, encoder, vision, custom
    adapter: str  # adapter key
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter:
    """Base adapter interface."""
    name: str = "base"

    def load(self, cfg, device="cpu"):
        raise NotImplementedError

    def supports_training_modes(self) -> List[str]:
        return ["finetune"]

    def supports_peft(self) -> bool:
        return False

    def num_params(self, model) -> int:
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0


class AdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, Type[ModelAdapter]] = {}
        self._models: Dict[str, ModelSpec] = {}

    def register_adapter(self, key: str, cls: Type[ModelAdapter]):
        self._adapters[key] = cls

    def register_model(self, spec: ModelSpec):
        self._models[spec.name] = spec

    def get_adapter(self, key: str) -> ModelAdapter:
        if key not in self._adapters:
            raise KeyError(f"Unknown adapter: {key}. Available: {list(self._adapters)}")
        return self._adapters[key]()

    def get_model(self, name: str) -> ModelSpec:
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}. Available: {list(self._models)}")
        return self._models[name]

    def list_adapters(self) -> List[str]:
        return sorted(self._adapters)

    def list_models(self) -> List[ModelSpec]:
        return list(self._models.values())


REGISTRY = AdapterRegistry()
