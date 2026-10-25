"""Dataset manager."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import datasets as hf_datasets
except Exception:
    hf_datasets = None


@dataclass
class DatasetBundle:
    train: Any
    validation: Optional[Any] = None
    fingerprint: str = ""
    source: str = ""
    num_train: int = 0
    num_val: int = 0
    text_column: str = "text"
    schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict_summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "fingerprint": self.fingerprint,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "text_column": self.text_column,
            "schema": self.schema,
        }


class DatasetManager:
    def __init__(self):
        pass

    def fingerprint_of(self, obj: Any, extra: str = "") -> str:
        try:
            s = str(obj) + extra
        except Exception:
            s = extra
        return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:16]

    def load_local(self, path: str, text_column: str = "text", val_fraction: Optional[float] = None, max_samples: Optional[int] = None) -> DatasetBundle:
        path = os.path.abspath(path)
        rows: List[Dict[str, Any]] = []
        if path.endswith(".jsonl") or path.endswith(".json"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif path.endswith(".csv"):
            import csv
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(dict(r))
        else:
            raise ValueError(f"Unsupported local dataset format: {path}")

        # clean empty
        rows = [r for r in rows if str(r.get(text_column, "")).strip()]
        # dedup
        seen = set()
        deduped = []
        for r in rows:
            key = str(r.get(text_column, "")).strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        rows = deduped

        if max_samples is not None:
            rows = rows[:max_samples]

        # split
        train_rows = rows
        val_rows = None
        if val_fraction and 0 < val_fraction < 1:
            n_val = max(1, int(len(rows) * val_fraction))
            val_rows = rows[-n_val:]
            train_rows = rows[:-n_val]

        fp = self.fingerprint_of({"path": path, "n": len(rows), "col": text_column})
        schema = {"columns": list(rows[0].keys()) if rows else [], "sample": rows[0] if rows else None}
        bundle = DatasetBundle(
            train=train_rows,
            validation=val_rows,
            fingerprint=fp,
            source=f"local:{path}",
            num_train=len(train_rows),
            num_val=len(val_rows) if val_rows else 0,
            text_column=text_column,
            schema=schema,
        )
        return bundle

    def load_hf(self, dataset_id: str, config_name: Optional[str] = None, split: str = "train", text_column: str = "text",
               val_fraction: Optional[float] = None, max_samples: Optional[int] = None, streaming: bool = False) -> DatasetBundle:
        if hf_datasets is None:
            raise RuntimeError("datasets library not installed")
        ds = hf_datasets.load_dataset(dataset_id, config_name, split=split, streaming=streaming)
        if not streaming:
            # clean empty
            def _keep(ex):
                return str(ex.get(text_column, "")).strip() != ""
            try:
                ds = ds.filter(_keep)
            except Exception:
                pass
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))
        # split
        train_ds = ds
        val_ds = None
        if (not streaming) and val_fraction and 0 < val_fraction < 1 and len(ds) > 1:
            split_out = ds.train_test_split(test_size=val_fraction, seed=42)
            train_ds = split_out["train"]
            val_ds = split_out["test"]

        fp = self.fingerprint_of({"id": dataset_id, "cfg": config_name, "split": split, "n": (len(train_ds) if not streaming else "stream")})
        schema = {}
        try:
            schema = {"columns": list(train_ds.column_names), "features": str(train_ds.features)}
        except Exception:
            pass
        bundle = DatasetBundle(
            train=train_ds,
            validation=val_ds,
            fingerprint=fp,
            source=f"hf:{dataset_id}",
            num_train=(len(train_ds) if not streaming else 0),
            num_val=(len(val_ds) if (val_ds is not None and not streaming) else 0),
            text_column=text_column,
            schema=schema,
        )
        return bundle

    def inspect(self, bundle: DatasetBundle) -> Dict[str, Any]:
        return bundle.to_dict_summary()
