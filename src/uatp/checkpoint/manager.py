"""Checkpoint manager: save, list, inspect, validate, import, resume helpers."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional

import torch

from .manifest import (
    sha256_file, build_manifest, write_manifest, read_manifest, verify_manifest,
)


class CheckpointManager:
    def __init__(self, runs_root: str = "./runs"):
        self.runs_root = os.path.abspath(runs_root)
        os.makedirs(self.runs_root, exist_ok=True)

    # ---- path helpers ----
    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.runs_root, run_id)

    def ckpt_dir(self, run_id: str, step: int) -> str:
        return os.path.join(self.run_dir(run_id), "checkpoints", f"checkpoint-{step}")

    def checkpoints_root(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), "checkpoints")

    # ---- save ----
    def save(
        self,
        run_id: str,
        global_step: int,
        state: Dict[str, Any],
        model_arch: str = "unknown",
        model_config: dict | None = None,
        training_config: dict | None = None,
        dataset_fingerprint: str = "",
        backend: str = "torch",
        checkpoint_type: str = "full",
        has_optimizer_state: bool = True,
        extra_manifest: dict | None = None,
    ) -> str:
        target = self.ckpt_dir(run_id, global_step)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        tmp = target + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)

        # write files from state dict: keys -> (filename, object)
        # state expected: {"model": state_dict or module, "optimizer":..., ...}
        files_map = {}
        # model
        model_obj = state.get("model")
        model_sd = model_obj.state_dict() if hasattr(model_obj, "state_dict") else model_obj
        model_path = os.path.join(tmp, "model.pt")
        torch.save(model_sd, model_path)
        files_map["model.pt"] = sha256_file(model_path)

        if state.get("optimizer") is not None:
            op = os.path.join(tmp, "optimizer.pt")
            torch.save(state["optimizer"].state_dict() if hasattr(state["optimizer"], "state_dict") else state["optimizer"], op)
            files_map["optimizer.pt"] = sha256_file(op)

        if state.get("scheduler") is not None:
            sp = os.path.join(tmp, "scheduler.pt")
            torch.save(state["scheduler"].state_dict() if hasattr(state["scheduler"], "state_dict") else state["scheduler"], sp)
            files_map["scheduler.pt"] = sha256_file(sp)

        if state.get("scaler") is not None:
            scp = os.path.join(tmp, "scaler.pt")
            torch.save(state["scaler"].state_dict() if hasattr(state["scaler"], "state_dict") else state["scaler"], scp)
            files_map["scaler.pt"] = sha256_file(scp)

        if state.get("rng") is not None:
            rp = os.path.join(tmp, "rng.pt")
            torch.save(state["rng"], rp)
            files_map["rng.pt"] = sha256_file(rp)

        if state.get("tokenizer") is not None:
            tdir = os.path.join(tmp, "tokenizer")
            os.makedirs(tdir, exist_ok=True)
            tok = state["tokenizer"]
            try:
                tok.save_pretrained(tdir)
            except Exception:
                # save via pickle-ish? use torch
                torch.save(tok, os.path.join(tdir, "tokenizer.pt"))
            # record directory files
            for root, _, fs in os.walk(tdir):
                for fn in fs:
                    rel = os.path.relpath(os.path.join(root, fn), tmp)
                    files_map[rel] = sha256_file(os.path.join(root, fn))

        # config
        cfg = {
            "run_id": run_id,
            "global_step": global_step,
            "model_config": model_config or {},
            "training_config": training_config or {},
        }
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        files_map["config.json"] = sha256_file(cfg_path)

        # dataset fingerprint
        dfp = os.path.join(tmp, "dataset_fingerprint.txt")
        with open(dfp, "w") as f:
            f.write(dataset_fingerprint or "")
        files_map["dataset_fingerprint.txt"] = sha256_file(dfp)

        manifest = build_manifest(
            run_id=run_id,
            global_step=global_step,
            model_arch=model_arch,
            model_config=model_config or {},
            backend=backend,
            training_config=training_config or {},
            dataset_fingerprint=dataset_fingerprint,
            files=files_map,
            has_optimizer_state=has_optimizer_state,
            checkpoint_type=checkpoint_type,
            extra=extra_manifest,
        )
        manifest_path = os.path.join(tmp, "manifest.json")
        # status incomplete initially
        write_manifest(manifest_path, manifest)

        # mark complete
        manifest["status"] = "complete"
        manifest["files"] = files_map  # manifest does not list itself
        write_manifest(manifest_path, manifest)

        # atomic rename
        if os.path.exists(target):
            shutil.rmtree(target)
        os.replace(tmp, target)

        # update latest pointer
        latest_path = os.path.join(parent, "latest.txt")
        # verify first
        ok, errs = verify_manifest(target)
        if not ok:
            raise RuntimeError(f"checkpoint verification failed: {errs}")
        with open(latest_path, "w") as f:
            f.write(os.path.basename(target))

        return target

    # ---- list / inspect / validate ----
    def list_checkpoints(self, run_id: Optional[str] = None, root: Optional[str] = None) -> List[Dict[str, Any]]:
        search_root = root or (self.run_dir(run_id) if run_id else self.runs_root)
        results = []
        # find checkpoint-* dirs
        for dirpath, dirnames, _ in os.walk(search_root):
            for dn in dirnames:
                if dn.startswith("checkpoint-"):
                    cdir = os.path.join(dirpath, dn)
                    info = self.inspect(cdir, quick=True)
                    results.append(info)
        # sort by step
        def key(x):
            try:
                return int(x.get("global_step", 0))
            except Exception:
                return 0
        results.sort(key=key)
        return results

    def inspect(self, ckpt_path: str, quick: bool = False) -> Dict[str, Any]:
        info = {"path": os.path.abspath(ckpt_path), "exists": os.path.isdir(ckpt_path)}
        manifest_path = os.path.join(ckpt_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            info["valid"] = False
            info["errors"] = ["missing manifest.json"]
            return info
        try:
            m = read_manifest(manifest_path)
        except Exception as e:
            info["valid"] = False
            info["errors"] = [str(e)]
            return info
        info.update({
            "run_id": m.get("run_id"),
            "global_step": m.get("global_step"),
            "model_arch": m.get("model_arch"),
            "checkpoint_type": m.get("checkpoint_type"),
            "has_optimizer_state": m.get("has_optimizer_state"),
            "status": m.get("status"),
            "created_at": m.get("created_at"),
            "dataset_fingerprint": m.get("dataset_fingerprint"),
            "backend": m.get("backend"),
        })
        # size
        size = 0
        for root, _, fs in os.walk(ckpt_path):
            for fn in fs:
                size += os.path.getsize(os.path.join(root, fn))
        info["size_bytes"] = size
        if not quick:
            ok, errs = verify_manifest(ckpt_path)
            info["valid"] = ok
            info["errors"] = errs
        else:
            info["valid"] = (m.get("status") == "complete")
            info["errors"] = []
        return info

    def validate(self, ckpt_path: str) -> tuple[bool, List[str]]:
        return verify_manifest(ckpt_path)

    def find_latest_valid(self, run_id: Optional[str] = None, root: Optional[str] = None) -> Optional[str]:
        cks = self.list_checkpoints(run_id=run_id, root=root)
        valids = []
        for c in cks:
            if c.get("valid"):
                valids.append(c)
        if not valids:
            # validate each
            for c in cks:
                ok, _ = self.validate(c["path"])
                if ok:
                    valids.append(c)
        if not valids:
            return None
        valids.sort(key=lambda x: int(x.get("global_step", 0)))
        return valids[-1]["path"]

    # ---- import checkpoint from external source ----
    def import_checkpoint(self, source: str, run_id: str, step_override: Optional[int] = None) -> str:
        """Import from folder, zip, or hf repo path. Returns checkpoint dir."""
        source = os.path.abspath(source) if not source.startswith("http") else source
        # if zip
        if source.endswith(".zip") and os.path.isfile(source):
            tmp_extract = os.path.join(self.runs_root, "_import_tmp")
            if os.path.exists(tmp_extract):
                shutil.rmtree(tmp_extract)
            os.makedirs(tmp_extract, exist_ok=True)
            with zipfile.ZipFile(source) as z:
                # safe extract
                for member in z.namelist():
                    target = os.path.realpath(os.path.join(tmp_extract, member))
                    if not target.startswith(os.path.realpath(tmp_extract)):
                        raise RuntimeError("unsafe archive: path traversal detected")
                z.extractall(tmp_extract)
            # find checkpoint dir
            found = None
            for dirpath, dirnames, _ in os.walk(tmp_extract):
                for dn in dirnames:
                    if dn.startswith("checkpoint-"):
                        found = os.path.join(dirpath, dn)
                        break
                if found:
                    break
            if found is None:
                # maybe the extracted root itself is a checkpoint
                if os.path.isfile(os.path.join(tmp_extract, "manifest.json")):
                    found = tmp_extract
                else:
                    # use tmp_extract as source folder
                    found = tmp_extract
            source = found

        # if directory
        if os.path.isdir(source):
            # validate
            ok, errs = verify_manifest(source)
            if not ok:
                # maybe model-only weights; create a model-only checkpoint? We'll still copy and mark
                pass
            step = step_override
            if step is None:
                # infer from dir name or manifest
                base = os.path.basename(source)
                if base.startswith("checkpoint-"):
                    try:
                        step = int(base.split("-", 1)[1])
                    except Exception:
                        step = 0
                else:
                    try:
                        m = read_manifest(os.path.join(source, "manifest.json"))
                        step = int(m.get("global_step", 0))
                    except Exception:
                        step = 0
            step = step or 0
            target = self.ckpt_dir(run_id, step)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(source, target)
            # ensure manifest status
            mp = os.path.join(target, "manifest.json")
            if os.path.isfile(mp):
                try:
                    m = read_manifest(mp)
                    m["run_id"] = run_id
                    m["status"] = "complete"
                    write_manifest(mp, m)
                except Exception:
                    pass
            return target

        raise ValueError(f"Unsupported import source: {source}")

    # ---- export model-only ----
    def export_model_only(self, ckpt_path: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        # load model.pt
        mp = os.path.join(ckpt_path, "model.pt")
        if not os.path.isfile(mp):
            raise FileNotFoundError("model.pt not found in checkpoint")
        sd = torch.load(mp, map_location="cpu")
        torch.save(sd, os.path.join(out_dir, "model.pt"))
        # copy tokenizer if present
        tdir = os.path.join(ckpt_path, "tokenizer")
        if os.path.isdir(tdir):
            dst = os.path.join(out_dir, "tokenizer")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(tdir, dst)
        # copy config
        for fn in ("config.json",):
            if os.path.isfile(os.path.join(ckpt_path, fn)):
                shutil.copy2(os.path.join(ckpt_path, fn), os.path.join(out_dir, fn))
        return out_dir

    # ---- compatibility check ----
    def compatible(self, ckpt_path: str, model_arch: str, model_config: dict) -> tuple[bool, list[str]]:
        errs = []
        try:
            m = read_manifest(os.path.join(ckpt_path, "manifest.json"))
        except Exception as e:
            return False, [str(e)]
        if m.get("model_arch") and m.get("model_arch") != model_arch:
            errs.append(f"model_arch mismatch: {m.get('model_arch')} != {model_arch}")
        # config subset check
        mc = m.get("model_config", {}) or {}
        for k, v in (model_config or {}).items():
            if k in mc and mc[k] != v:
                errs.append(f"config mismatch {k}: {mc[k]} != {v}")
        ok, verrs = verify_manifest(ckpt_path)
        if not ok:
            errs.extend(verrs)
        return len(errs) == 0, errs
