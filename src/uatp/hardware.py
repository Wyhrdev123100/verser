"""Hardware detection and profiling."""
from __future__ import annotations

import os
import platform as _platform
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class HardwareInfo:
    device: str
    gpu_name: Optional[str]
    gpu_count: int
    total_vram_gb: float
    free_vram_gb: float
    cuda_version: Optional[str]
    bf16_supported: bool
    fp16_supported: bool
    cpu_count: int
    total_ram_gb: float
    free_disk_gb: float
    platform: str
    python: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect() -> HardwareInfo:
    gpu_name = None
    gpu_count = 0
    total_vram = 0.0
    free_vram = 0.0
    cuda_version = None
    bf16 = False
    fp16 = False
    device = "cpu"

    if torch is not None and torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024 ** 3)
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            cuda_version = torch.version.cuda
            # bf16 support heuristic
            major = getattr(props, "major", 0)
            bf16 = major >= 8
            fp16 = True
        except Exception:
            pass
    elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        # approximate
        try:
            import psutil
            total_vram = psutil.virtual_memory().total / (1024 ** 3)
            free_vram = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            pass

    cpu_count = os.cpu_count() or 1
    total_ram = 0.0
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    free_disk = 0.0
    try:
        free_disk = shutil.disk_usage(".").free / (1024 ** 3)
    except Exception:
        pass

    return HardwareInfo(
        device=device,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_gb=round(total_vram, 3),
        free_vram_gb=round(free_vram, 3),
        cuda_version=cuda_version,
        bf16_supported=bf16,
        fp16_supported=fp16,
        cpu_count=cpu_count,
        total_ram_gb=round(total_ram, 3),
        free_disk_gb=round(free_disk, 3),
        platform=_platform.platform(),
        python=_platform.python_version(),
    )


def recommend_config(hw: HardwareInfo, model_params_billions: float = 0.1, mode: str = "balanced") -> Dict[str, Any]:
    """Return suggested training knobs based on hardware."""
    vram = hw.total_vram_gb if hw.device != "cpu" else max(hw.total_ram_gb * 0.5, 4.0)
    # base heuristic
    if vram >= 24:
        micro = 8
        seq = 2048
    elif vram >= 16:
        micro = 4
        seq = 1024
    elif vram >= 8:
        micro = 2
        seq = 512
    else:
        micro = 1
        seq = 256

    if mode == "safe":
        micro = max(1, micro // 2)
        seq = max(128, seq // 2)
        grad_ckpt = True
    elif mode == "performance":
        grad_ckpt = False
    else:
        grad_ckpt = vram < 12

    accum = 1
    target_effective = 16
    while micro * accum < target_effective and accum < 64:
        accum *= 2

    precision = "bf16" if hw.bf16_supported else ("fp16" if hw.fp16_supported and hw.device != "cpu" else "none")
    if mode == "safe":
        precision = "fp16" if (hw.fp16_supported and hw.device != "cpu") else "none"
        if hw.bf16_supported and mode != "safe":
            precision = "bf16"

    return {
        "micro_batch_size": micro,
        "gradient_accumulation": accum,
        "sequence_length": seq,
        "mixed_precision": precision,
        "gradient_checkpointing": grad_ckpt,
        "dataloader_workers": min(4, hw.cpu_count),
        "checkpoint_every": 500,
    }