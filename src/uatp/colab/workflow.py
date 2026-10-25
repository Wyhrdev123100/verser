"""Colab-friendly workflow helpers."""
from __future__ import annotations

import os
import shutil
import zipfile
from typing import Optional

from ..storage.manager import StorageManager
from ..checkpoint.manager import CheckpointManager


def detect_runtime() -> str:
    try:
        import google.colab  # noqa
        return "colab"
    except Exception:
        return "local"


def prepare_storage(preferred: Optional[str] = None) -> str:
    sm = StorageManager()
    if preferred and os.path.isdir(preferred):
        return preferred
    gd = StorageManager.gdrive_root()
    if gd:
        return gd
    mounted = StorageManager.mount_colab_drive()
    if mounted:
        return mounted
    return os.getcwd()


def sync_checkpoint(ckpt_path: str, durable_dir: str) -> str:
    base = os.path.basename(ckpt_path)
    dst = os.path.join(durable_dir, base)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(ckpt_path, dst)
    # verify
    from ..checkpoint.manifest import verify_manifest
    ok, errs = verify_manifest(dst)
    if not ok:
        raise RuntimeError(f"verification failed after sync: {errs}")
    return dst


def export_zip(ckpt_path: str, out_path: str) -> str:
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(ckpt_path):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, os.path.dirname(ckpt_path))
                z.write(full, arc)
    return out_path


def find_latest_checkpoint(durable_dir: str) -> Optional[str]:
    cm = CheckpointManager(durable_dir)
    return cm.find_latest_valid(root=durable_dir)
