"""Training engine."""
from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import RunConfig
from ..hardware import detect, recommend_config
from ..models import REGISTRY
from ..datasets.manager import DatasetManager
from ..runs.manager import RunManager
from ..checkpoint.manager import CheckpointManager
from ..storage.manager import StorageManager


class SimpleTextDataset(Dataset):
    def __init__(self, records, tokenizer, text_column, seq_len):
        self.records = records
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.seq_len = seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = str(self.records[idx].get(self.text_column, ""))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len + 1,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": enc.get("attention_mask", torch.ones_like(ids)).squeeze(0)}


class ToyDataset(Dataset):
    def __init__(self, n, seq_len, vocab_size):
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": ids, "labels": ids.clone()}


def _records(bundle):
    data = bundle.train
    if hasattr(data, "to_list"):
        try:
            return data.to_list()
        except Exception:
            pass
    if isinstance(data, list):
        return data
    try:
        return [data[i] for i in range(len(data))]
    except Exception:
        return []


class TrainingEngine:
    def __init__(self, runs_root: str = "./runs"):
        self.runs_root = runs_root
        self.runs = RunManager(runs_root)
        self.ckpts = CheckpointManager(runs_root)
        self.storage = StorageManager(runs_root)
        self.dm = DatasetManager()

    def _resolve_adapter(self, cfg: RunConfig):
        name = cfg.model.model_id
        model_names = {m.name for m in REGISTRY.list_models()}
        if name in model_names:
            spec = REGISTRY.get_model(name)
            return REGISTRY.get_adapter(spec.adapter), spec
        if cfg.model.source == "huggingface":
            return REGISTRY.get_adapter("huggingface"), None
        if cfg.model.source in ("toy", "local"):
            return REGISTRY.get_adapter("toy_gpt"), None
        raise ValueError(f"Cannot resolve model adapter for {name}")

    def _fill_hardware(self, cfg: RunConfig):
        hw = detect()
        if cfg.training.micro_batch_size is None or cfg.training.gradient_accumulation is None or cfg.training.sequence_length is None:
            rec = recommend_config(hw, mode=cfg.hardware.mode)
            if cfg.training.micro_batch_size is None:
                cfg.training.micro_batch_size = rec["micro_batch_size"]
            if cfg.training.gradient_accumulation is None:
                cfg.training.gradient_accumulation = rec["gradient_accumulation"]
            if cfg.training.sequence_length is None:
                cfg.training.sequence_length = rec["sequence_length"]
        if cfg.training.mixed_precision == "auto":
            if hw.bf16_supported:
                cfg.training.mixed_precision = "bf16"
            elif hw.fp16_supported and hw.device != "cpu":
                cfg.training.mixed_precision = "fp16"
            else:
                cfg.training.mixed_precision = "none"
        return hw

    def _load_data(self, cfg: RunConfig):
        dcfg = cfg.dataset
        if dcfg.source == "local":
            path = dcfg.dataset_id
            bundle = self.dm.load_local(path, text_column=dcfg.text_column, val_fraction=dcfg.validation_split, max_samples=dcfg.max_samples)
        elif dcfg.source == "huggingface":
            bundle = self.dm.load_hf(dcfg.dataset_id, config_name=dcfg.config_name, split=dcfg.split, text_column=dcfg.text_column, val_fraction=dcfg.validation_split, max_samples=dcfg.max_samples, streaming=dcfg.streaming)
        else:
            raise ValueError(f"Unknown dataset source {dcfg.source}")
        return bundle

    def train(self, cfg: RunConfig, resume_from: Optional[str] = None, max_steps_override: Optional[int] = None) -> str:
        if max_steps_override is not None:
            cfg.training.max_steps = max_steps_override

        hw = self._fill_hardware(cfg)
        device = "cpu" if cfg.hardware.force_cpu else hw.device
        if device == "mps":
            device = "cpu"  # simpler
        # model
        adapter, spec = self._resolve_adapter(cfg)
        loaded = adapter.load(cfg, device=device)
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        family = loaded.get("family", cfg.model.model_type)
        model_arch = model.__class__.__name__

        # data
        bundle = self._load_data(cfg)
        text_col = cfg.dataset.text_column
        seq_len = cfg.training.sequence_length

        if tokenizer is not None:
            records = _records(bundle)
            train_ds = SimpleTextDataset(records, tokenizer, text_col, seq_len)
            val_ds = None
            if bundle.validation is not None:
                vrecords = bundle.validation.to_list() if hasattr(bundle.validation, "to_list") else (list(bundle.validation) if not isinstance(bundle.validation, list) else bundle.validation)
                if not isinstance(vrecords, list):
                    vrecords = _records(type("B", (), {"train": bundle.validation})())
                val_ds = SimpleTextDataset(vrecords, tokenizer, text_col, seq_len)
        else:
            # toy
            n = bundle.num_train if bundle.num_train > 0 else 256
            train_ds = ToyDataset(n, seq_len, getattr(model, "vocab_size", 64))
            val_ds = None

        micro_bs = cfg.training.micro_batch_size
        accum = cfg.training.gradient_accumulation
        nw = cfg.training.dataloader_workers
        train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, num_workers=nw, drop_last=False)

        # optimizer
        lr = cfg.training.learning_rate
        wd = cfg.training.weight_decay
        opt_name = cfg.training.optimizer.lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

        # scheduler
        epochs = cfg.training.epochs
        steps_per_epoch = max(1, len(train_loader) // accum)
        total_steps = cfg.training.max_steps if cfg.training.max_steps else epochs * steps_per_epoch
        warmup = cfg.training.warmup_steps
        sched_name = cfg.training.scheduler.lower()
        if sched_name == "linear":
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warm = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup))
            decay = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, total_steps - warmup)) if total_steps > warmup else None
            if decay is not None:
                scheduler = SequentialLR(optimizer, schedulers=[warm, decay], milestones=[warmup])
            else:
                scheduler = warm
        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # mixed precision
        precision = cfg.training.mixed_precision
        use_scaler = (precision == "fp16") and (device.startswith("cuda"))
        scaler = torch.cuda.amp.GradScaler() if use_scaler else None

        # gradient checkpointing for HF
        if cfg.training.gradient_checkpointing:
            try:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
            except Exception:
                pass

        # run record
        run_id = cfg.run_id or self.runs.new_run_id()
        cfg.run_id = run_id
        config_dict = cfg.to_dict()
        resumed_from = resume_from
        parent_run_id = None
        if resume_from:
            # try to find parent run from checkpoint manifest
            try:
                from ..checkpoint.manifest import read_manifest
                m = read_manifest(os.path.join(resume_from, "manifest.json"))
                parent_run_id = m.get("run_id")
            except Exception:
                parent_run_id = None
        rr = self.runs.create(config_dict, run_id=run_id, parent_run_id=parent_run_id, resumed_from=resume_from)
        self.runs.update_status(run_id, "running", global_step=0)

        model_config = {
            "model_arch": model_arch,
            "model_type": cfg.model.model_type,
            "dtype": cfg.model.dtype,
        }
        arch_str = model_arch

        global_step = 0
        # resume
        model_only_resume = False
        if resume_from:
            ok, errs = self.ckpts.compatible(resume_from, arch_str, model_config)
            if not ok:
                # maybe model-only; try load weights anyway
                pass
            # load weights
            mp = os.path.join(resume_from, "model.pt")
            if os.path.isfile(mp):
                try:
                    sd = torch.load(mp, map_location=device)
                    model.load_state_dict(sd, strict=False)
                except Exception as e:
                    self.runs.update_status(run_id, "failed", error=f"resume load failed: {e}")
                    raise
            # optimizer/sched/scaler
            op = os.path.join(resume_from, "optimizer.pt")
            if os.path.isfile(op):
                try:
                    optimizer.load_state_dict(torch.load(op, map_location=device))
                except Exception:
                    pass
            else:
                model_only_resume = True
            sp = os.path.join(resume_from, "scheduler.pt")
            if os.path.isfile(sp):
                try:
                    scheduler.load_state_dict(torch.load(sp, map_location=device))
                except Exception:
                    pass
            scp = os.path.join(resume_from, "scaler.pt")
            if scaler is not None and os.path.isfile(scp):
                try:
                    scaler.load_state_dict(torch.load(scp, map_location=device))
                except Exception:
                    pass
            # rng
            rp = os.path.join(resume_from, "rng.pt")
            if os.path.isfile(rp):
                try:
                    rng = torch.load(rp, map_location="cpu")
                    if isinstance(rng, dict) and "torch" in rng:
                        torch.set_rng_state(rng["torch"])
                except Exception:
                    pass
            # step
            try:
                from ..checkpoint.manifest import read_manifest
                m = read_manifest(os.path.join(resume_from, "manifest.json"))
                global_step = int(m.get("global_step", 0))
            except Exception:
                global_step = 0
            if model_only_resume:
                self.runs.append_metric(run_id, {"warning": "resume from model-only checkpoint; optimizer state not restored"})

        max_grad_norm = cfg.training.max_grad_norm
        checkpoint_every = cfg.training.checkpoint_every
        log_every = cfg.training.log_every
        eval_every = cfg.training.eval_every

        def save_ckpt(step):
            state = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "tokenizer": tokenizer,
                "rng": {"torch": torch.get_rng_state()},
            }
            path = self.ckpts.save(
                run_id=run_id,
                global_step=step,
                state=state,
                model_arch=arch_str,
                model_config=model_config,
                training_config=__import__("dataclasses").asdict(cfg.training),
                dataset_fingerprint=bundle.fingerprint,
                backend="torch",
                checkpoint_type="full",
                has_optimizer_state=True,
            )
            self.runs.append_checkpoint(run_id, path)
            self.runs.update_status(run_id, "running", global_step=step)
            return path

        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        total_loss_accum = 0.0
        micro_count = 0
        try:
            for epoch in range(epochs):
                for batch in train_loader:
                    # move
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attn = batch.get("attention_mask")
                    if attn is not None:
                        attn = attn.to(device)

                    ctx = contextlib.nullcontext()
                    if precision == "fp16":
                        ctx = torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=device.startswith("cuda"))
                    elif precision == "bf16":
                        ctx = torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=True)

                    with ctx:
                        if tokenizer is not None:
                            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                        else:
                            out = model(input_ids=input_ids, labels=labels)
                        loss = out.loss
                        if loss is None:
                            # fallback
                            logits = out.logits
                            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                        loss = loss / accum

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    total_loss_accum += loss.item()
                    micro_count += 1

                    if micro_count % accum == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1

                        if log_every and global_step % log_every == 0:
                            self.runs.append_metric(run_id, {
                                "step": global_step,
                                "loss": round(total_loss_accum * accum / max(1, accum), 6),
                                "epoch": epoch,
                                "time": round(time.time() - t0, 2),
                            })
                            total_loss_accum = 0.0

                        if checkpoint_every and global_step % checkpoint_every == 0:
                            save_ckpt(global_step)

                        if eval_every and global_step % eval_every == 0 and val_ds is not None:
                            try:
                                val_loss = self._eval(model, val_ds, device, micro_bs, precision)
                                self.runs.append_metric(run_id, {"step": global_step, "val_loss": val_loss})
                            except Exception:
                                pass

                        if total_steps and global_step >= total_steps:
                            break
                if total_steps and global_step >= total_steps:
                    break
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "oom" in msg.lower():
                # attempt save
                try:
                    if global_step > 0:
                        save_ckpt(global_step)
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.runs.update_status(run_id, "failed", global_step=global_step, error=f"OOM: {msg}")
                raise RuntimeError(f"Out of memory during training: {msg}. Suggest lowering batch size / sequence length. Checkpoint saved if possible.")
            self.runs.update_status(run_id, "failed", global_step=global_step, error=msg)
            raise

        # final checkpoint if not saved at last step
        last_ckpt_step = None
        try:
            cks = self.ckpts.list_checkpoints(run_id=run_id)
            if cks:
                last_ckpt_step = cks[-1].get("global_step")
        except Exception:
            pass
        if last_ckpt_step != global_step and global_step > 0:
            save_ckpt(global_step)

        self.runs.update_status(run_id, "completed", global_step=global_step)
        return run_id

    def _eval(self, model, val_ds, device, bs, precision):
        model.eval()
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        losses = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn = batch.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                if out.loss is not None:
                    losses.append(out.loss.item())
        model.train()
        return float(sum(losses) / len(losses)) if losses else 0.0
