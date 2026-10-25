import json, os
from uatp.datasets.manager import DatasetManager


def test_local_load(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"text": "hello"}, {"text": ""}, {"text": "world"}, {"text": "hello"}]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dm = DatasetManager()
    b = dm.load_local(str(p), text_column="text")
    assert b.num_train == 2  # dedup + clean
    assert b.fingerprint
    assert b.schema["columns"] == ["text"]


def test_local_split(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"text": f"t{i}"} for i in range(10)]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dm = DatasetManager()
    b = dm.load_local(str(p), val_fraction=0.2)
    assert b.num_val == 2
    assert b.num_train == 8