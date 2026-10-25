import os
import tempfile
from uatp.colab.workflow import prepare_storage, sync_checkpoint, export_zip, find_latest_checkpoint
from uatp.checkpoint.manager import CheckpointManager
from uatp.models.toy import ToyGPT


def test_colab_sync(tmp_path):
    # durable dir
    durable = tmp_path / "drive"
    durable.mkdir()
    cm = CheckpointManager(str(tmp_path / "runs"))
    model = ToyGPT()
    state = {"model": model, "optimizer": None, "scheduler": None, "scaler": None, "tokenizer": None, "rng": None}
    path = cm.save("run1", 4, state, model_arch="ToyGPT", dataset_fingerprint="fp1")
    dst = sync_checkpoint(path, str(durable))
    assert os.path.isdir(dst)
    latest = find_latest_checkpoint(str(durable))
    assert latest is not None
    assert latest.endswith("checkpoint-4")
    # zip export
    zp = tmp_path / "ckpt.zip"
    export_zip(path, str(zp))
    assert os.path.isfile(zp)