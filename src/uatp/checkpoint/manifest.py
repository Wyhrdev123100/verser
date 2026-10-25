"""Checkpoint manifest helpers."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    run_id: str,
    global_step: int,
    model_arch: str,
    model_config: dict,
    backend: str,
    training_config: dict,
    dataset_fingerprint: str,
    files: Dict[str, str],
    has_optimizer_state: bool = True,
    checkpoint_type: str = "full",
    extra: dict | None = None,
) -> Dict[str, Any]:
    import torch, sys
    manifest = {
        "run_id": run_id,
        "global_step": int(global_step),
        "model_arch": model_arch,
        "model_config": model_config,
        "backend": backend,
        "backend_version": getattr(torch, "__version__", "unknown") if False else "torch",
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__ if "torch" in globals() else "unknown",
        },
        "dataset_fingerprint": dataset_fingerprint,
        "training_config": training_config,
        "files": files,
        "status": "incomplete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_type": checkpoint_type,
        "has_optimizer_state": bool(has_optimizer_state),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def verify_manifest(ckpt_dir: str) -> tuple[bool, list[str]]:
    errors = []
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, ["missing manifest.json"]
    try:
        m = read_manifest(manifest_path)
    except Exception as e:
        return False, [f"manifest parse error: {e}"]
    if m.get("status") != "complete":
        errors.append("checkpoint status not complete")
    files = m.get("files", {})
    for name, sha in files.items():
        fpath = os.path.join(ckpt_dir, name)
        if not os.path.isfile(fpath):
            errors.append(f"missing file: {name}")
            continue
        try:
            actual = sha256_file(fpath)
        except Exception as e:
            errors.append(f"cannot hash {name}: {e}")
            continue
        if actual != sha:
            errors.append(f"checksum mismatch: {name}")
    # required files
    if not os.path.isfile(os.path.join(ckpt_dir, "manifest.json")):
        errors.append("manifest missing")
    return len(errors) == 0, errors
