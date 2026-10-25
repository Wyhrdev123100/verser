"""Model adapters."""
from __future__ import annotations

from typing import Any, Dict, List

from .registry import ModelAdapter


class ToyGPTAdapter(ModelAdapter):
    name = "toy_gpt"

    def load(self, cfg, device="cpu"):
        from .toy import ToyGPT
        m = ToyGPT()
        return {"model": m, "tokenizer": None, "family": "causal_lm"}

    def supports_training_modes(self):
        return ["pretrain", "finetune"]

    def supports_peft(self):
        return False


class HuggingFaceAdapter(ModelAdapter):
    name = "huggingface"

    def load(self, cfg, device="cpu"):
        try:
            from transformers import AutoTokenizer, AutoConfig
        except Exception as e:
            raise RuntimeError("transformers not installed") from e

        mcfg = cfg.model
        model_id = mcfg.model_id
        trust = getattr(mcfg, "trust_remote_code", False)
        dtype_str = getattr(mcfg, "dtype", "auto")
        import torch
        dtype = None
        if dtype_str == "fp16":
            dtype = torch.float16
        elif dtype_str == "bf16":
            dtype = torch.bfloat16
        elif dtype_str == "fp32":
            dtype = torch.float32

        tok_id = mcfg.tokenizer_id or model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=trust, revision=mcfg.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        family = mcfg.model_type
        if family == "causal_lm":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust, revision=mcfg.revision)
        elif family == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust, revision=mcfg.revision)
        elif family == "encoder":
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust, revision=mcfg.revision)
        elif family == "vision":
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust, revision=mcfg.revision)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=trust, revision=mcfg.revision)

        model.to(device)
        # peft
        adapter = getattr(mcfg, "adapter", None)
        if adapter:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                tt = TaskType.CAUSAL_LM if family == "causal_lm" else TaskType.SEQ_2_SEQ_LM
                lcfg = LoraConfig(task_type=tt, r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear")
                model = get_peft_model(model, lcfg)
            except Exception as e:
                raise RuntimeError(f"PEFT failed: {e}")

        return {"model": model, "tokenizer": tokenizer, "family": family}

    def supports_training_modes(self):
        return ["finetune", "continued_pretrain", "peft"]

    def supports_peft(self):
        return True
