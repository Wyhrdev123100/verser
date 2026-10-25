"""Wynr-1.0 tokenizer pipeline: deterministic character vocabulary with BOS/EOS/PAD/UNK."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

SPECIAL_ORDER = ["<bos>", "<eos>", "<pad>", "<unk>"]
SPECIAL_IDS = {"bos": 1, "eos": 2, "pad": 3, "unk": 4}
TOKENIZER_VERSION = 1


class WynrTokenizer:
    """Character-level tokenizer trained on Wynr-1.0 project data.

    Vocabulary is derived deterministically from the characters observed in the
    training corpus plus four reserved special tokens (BOS/EOS/PAD/UNK).
    Special ids: BOS=1, EOS=2, PAD=3, UNK=4; real characters start at id 5.
    """

    def __init__(
        self,
        char_to_id: Dict[str, int],
        id_to_char: Dict[int, str],
        source: str,
        version: int = TOKENIZER_VERSION,
        hash_value: str = "",
        vocab_size: Optional[int] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.char_to_id = dict(char_to_id)
        self.id_to_char = dict(id_to_char)
        self.source = source
        self.version = int(version)
        self.hash_value = hash_value or WynrTokenizer._compute_hash(char_to_id, source)
        self._vocab_size = int(vocab_size) if vocab_size is not None else max(self.char_to_id.values(), default=4) + 1
        self.model_dir = Path(model_dir) if model_dir is not None else None

    # ------------------------------------------------------------------ #
    # Training / loading / saving
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        texts: list,
        model_dir: Path,
        vocab_size: Optional[int] = None,
        source: str = "synthetic_v1",
    ) -> "WynrTokenizer":
        chars: set = set()
        for t in texts:
            for ch in str(t):
                chars.add(ch)
        char_list = sorted(chars)
        char_to_id = {ch: cid for cid, ch in enumerate(char_list, start=5)}
        id_to_char = {cid: ch for ch, cid in char_to_id.items()}
        vocab_size = len(char_to_id) + 5
        tok = cls(
            char_to_id=char_to_id,
            id_to_char=id_to_char,
            source=source,
            version=TOKENIZER_VERSION,
            hash_value=cls._compute_hash(char_to_id, source),
            vocab_size=vocab_size,
            model_dir=Path(model_dir),
        )
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(json.dumps(tok._vocab_payload(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return tok

    @classmethod
    def load(cls, directory: Path) -> "WynrTokenizer":
        directory = Path(directory)
        jp = directory / "tokenizer.json"
        if not jp.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {jp}")
        vocab = json.loads(jp.read_text(encoding="utf-8"))
        char_to_id = {ch: int(cid) for ch, cid in vocab.get("char_to_id", {}).items()}
        id_to_char = {int(k): v for k, v in vocab.get("id_to_char", {}).items()}
        return cls(
            char_to_id=char_to_id,
            id_to_char=id_to_char,
            source=str(vocab.get("source") or "unknown"),
            version=int(vocab.get("version") or TOKENIZER_VERSION),
            hash_value=str(vocab.get("hash") or ""),
            vocab_size=int(vocab.get("vocab_size") or 0) or max(char_to_id.values(), default=4) + 1,
            model_dir=directory,
        )

    def save(self, directory: Path) -> Dict[str, object]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        vocab = self._vocab_payload()
        jp = directory / "tokenizer.json"
        jp.write_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return {
            "path": str(jp),
            "version": self.version,
            "hash": self.hash_value,
            "vocab_size": self.vocab_size,
            "source": self.source,
        }

    def _vocab_payload(self) -> dict:
        char_to_id = {ch: cid for ch, cid in sorted(self.char_to_id.items(), key=lambda kv: kv[1])}
        id_to_char = {str(k): v for k, v in sorted(self.id_to_char.items())}
        return {
            "version": self.version,
            "source": self.source,
            "special_tokens": list(SPECIAL_ORDER),
            "char_to_id": char_to_id,
            "id_to_char": id_to_char,
            "vocab_size": self.vocab_size,
        }

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    @property
    def bos_id(self) -> int:
        return SPECIAL_IDS["bos"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_IDS["eos"]

    @property
    def pad_id(self) -> int:
        return SPECIAL_IDS["pad"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_IDS["unk"]

    @property
    def vocab_size(self) -> int:
        return max(self.char_to_id.values(), default=4) + 1

    def encode(self, text: str) -> List[int]:
        ids = [self.bos_id]
        for ch in str(text):
            cid = self.char_to_id.get(ch, self.unk_id)
            ids.append(cid)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        skip = {self.bos_id, self.eos_id, self.pad_id, self.unk_id}
        out: List[str] = []
        for i in ids:
            if int(i) in skip:
                continue
            ch = self.id_to_char.get(int(i))
            if ch is None:
                continue
            out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_hash(char_to_id: Dict[str, int], source: str) -> str:
        payload = json.dumps(
            {"char_to_id": char_to_id, "source": source, "version": TOKENIZER_VERSION},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
