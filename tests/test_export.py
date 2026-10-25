import os
import torch
from uatp.checkpoint.manager import CheckpointManager
from uatp.export.manager import ExportManager
from uatp.models.toy import ToyGPT


def test_export(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 3, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    out = tmp_path / "export"
    em = ExportManager(str(tmp_path / "runs"))
    em.export(path, str(out))
    assert os.path.isfile(out / "model.pt")
    # verify load
    sd = torch.load(out / "model.pt", map_location="cpu")
    assert isinstance(sd, dict)