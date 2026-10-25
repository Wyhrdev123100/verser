"""Export manager."""
from __future__ import annotations

import os
import shutil
from typing import Optional

import torch

from ..checkpoint.manager import CheckpointManager


class ExportManager:
    def __init__(self, runs_root: str = "./runs"):
        self.ckpts = CheckpointManager(runs_root)

    def export(self, ckpt_path: str, out_dir: str, include_tokenizer: bool = True) -> str:
        os.makedirs(out_dir, exist_ok=True)
        out = self.ckpts.export_model_only(ckpt_path, out_dir)
        # write model card stub
        card = os.path.join(out_dir, "MODEL_CARD.md")
        with open(card, "w") as f:
            f.write("# Exported Model\n\nExported from checkpoint: %s\n" % ckpt_path)
        # verify load
        mp = os.path.join(out, "model.pt")
        if os.path.isfile(mp):
            sd = torch.load(mp, map_location="cpu")
            # basic check
            if not isinstance(sd, dict):
                raise RuntimeError("exported weights invalid")
        return out
