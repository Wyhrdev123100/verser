import os
import torch
import pytest

from uatp.checkpoint.manager import CheckpointManager
from uatp.models.toy import ToyGPT


def test_save_validate_list(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 5, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    assert os.path.isdir(path)
    ok, errs = cm.validate(path)
    assert ok, errs
    cks = cm.list_checkpoints(run_id="run1")
    assert len(cks) == 1
    assert cks[0]["global_step"] == 5
    assert cks[0]["valid"] is True


def test_checksum_corruption(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 5, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    # corrupt model.pt
    mp = os.path.join(path, "model.pt")
    with open(mp, "ab") as f:
        f.write(b"x")
    ok, errs = cm.validate(path)
    assert not ok
    assert any("checksum" in e for e in errs)


def test_find_latest_valid(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    for step in (2, 4):
        state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
        cm.save("run1", step, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    latest = cm.find_latest_valid(run_id="run1")
    assert latest.endswith("checkpoint-4")


def test_compatible_incompatible(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 5, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    ok, errs = cm.compatible(path, "OtherArch", {})
    assert not ok
    ok2, _ = cm.compatible(path, "ToyGPT", {})
    assert ok2


def test_import_zip(tmp_path):
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 7, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    # zip it
    import zipfile
    zip_path = tmp_path / "ckpt.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(path):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, os.path.dirname(path))
                z.write(full, arc)
    # import into new run
    target = cm.import_checkpoint(str(zip_path), "run2", step_override=7)
    assert os.path.isdir(target)
    ok, errs = cm.validate(target)
    assert ok, errs


def test_model_only_import(tmp_path):
    # a folder with just model.pt (no manifest) should be importable as model-only
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    d = tmp_path / "raw"
    d.mkdir()
    torch.save(model.state_dict(), d / "model.pt")
    target = cm.import_checkpoint(str(d), "run3", step_override=0)
    assert os.path.isdir(target)
    # manifest may be missing -> invalid but exists
    assert os.path.exists(target)